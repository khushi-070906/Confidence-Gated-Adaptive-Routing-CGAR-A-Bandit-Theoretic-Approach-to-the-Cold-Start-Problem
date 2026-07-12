"""
simulator.py
------------
Runs the gateway simulation for a given router and records:
  - Per-request: latency, error, SLA violation, reward, regret
  - Per-churn-event: latency window around the event (for P99 analysis)
  - Cumulative regret over time (the paper's primary figure)

Usage:
    from core.simulator import run_simulation
    results = run_simulation(router, pool, n_requests=5000)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List


@dataclass
class SimResult:
    """All metrics collected from one simulation run."""
    router_name: str
    latencies: List[float] = field(default_factory=list)
    errors: List[bool] = field(default_factory=list)
    sla_violations: List[bool] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    regrets: List[float] = field(default_factory=list)
    cumulative_regret: List[float] = field(default_factory=list)
    churn_timesteps: List[int] = field(default_factory=list)

    # For P99 analysis around churn events
    # List of (relative_t, latency) pairs where relative_t=0 is the churn event
    churn_window_latencies: List[tuple] = field(default_factory=list)

    def p99_latency(self) -> float:
        if not self.latencies:
            return 0.0
        return float(np.percentile(self.latencies, 99))

    def p95_latency(self) -> float:
        if not self.latencies:
            return 0.0
        return float(np.percentile(self.latencies, 95))

    def p50_latency(self) -> float:
        if not self.latencies:
            return 0.0
        return float(np.percentile(self.latencies, 50))

    def sla_violation_rate(self) -> float:
        if not self.sla_violations:
            return 0.0
        return float(np.mean(self.sla_violations))

    def total_regret(self) -> float:
        return float(sum(self.regrets))


def compute_reward(latency: float, error: bool, sla_violation: bool,
                   alpha: float = 0.01, beta: float = 5.0, gamma: float = 10.0) -> float:
    """
    Reward function: Equation (5) in the paper.
    r(t) = -alpha*latency - beta*error_rate - gamma*sla_violation
    Returns a scalar reward (more negative = worse).
    """
    return -alpha * latency - beta * float(error) - gamma * float(sla_violation)


def run_simulation(router, pool, n_requests: int = 5000,
                   churn_window: int = 200,
                   alpha: float = 0.01, beta: float = 5.0, gamma: float = 10.0,
                   router_name: str = "Router") -> SimResult:
    """
    Run the full simulation.

    Parameters
    ----------
    router       : any router with .route(), .update(), .notify_pool_change()
    pool         : BackendPool instance
    n_requests   : total number of requests to simulate
    churn_window : number of requests before/after a churn event to log for P99 analysis
    router_name  : label for results
    """
    result = SimResult(router_name=router_name)
    churn_events_seen = set()
    cumulative_reg = 0.0

    # Track churn event times from pool schedule for window analysis
    scheduled_churns = {t for t, _ in pool.churn_schedule}

    for t in range(n_requests):

        # ── 1. Check for churn events ────────────────────────────────────
        active_before = {b.id for b in pool.active_backends}
        churn_fired = pool.step_churn(t)
        active_after = {b.id for b in pool.active_backends}

        if churn_fired:
            new_ids = list(active_after - active_before)
            removed_ids = list(active_before - active_after)
            router.notify_pool_change(new_ids, removed_ids)
            result.churn_timesteps.append(t)

        # ── 2. Get active backends ────────────────────────────────────────
        active = pool.active_backends
        if not active:
            continue

        # ── 3. Oracle: what's the best possible reward this step? ─────────
        oracle_idx = pool.oracle_best()
        oracle_result = active[oracle_idx].serve()
        oracle_reward = compute_reward(
            oracle_result["latency"], oracle_result["error"],
            oracle_result["sla_violation"] if "sla_violation" in oracle_result
            else oracle_result["latency"] > pool.sla_threshold,
            alpha, beta, gamma
        )
        # NOTE (fix): Backend.serve() already increments AND decrements
        # queue_depth internally before returning (see environment.py),
        # so it is already queue-neutral. The line that used to be here
        # subtracted queue_depth by an EXTRA 1 for whichever backend is
        # currently the oracle (lowest base_latency) -- on every single
        # timestep. That artificially deflated the true-best backend's
        # queue depth for the real routing decision that follows,
        # systematically favoring Least-Connections-style logic (and
        # CGAR's H_i = 1/(queue_depth+1) heuristic term) whenever they
        # happened to route to that backend. No correction is needed
        # for queue_depth; serve() is already fair on that front.
        #
        # NOTE (fix): serve() does NOT recompute self.cpu after resetting
        # queue_depth back down -- cpu is left at the momentary, elevated
        # "mid-service" value it had right after the increment. Left
        # uncorrected, that stale value leaks into the router's state
        # features (b.cpu) for whichever backend is currently the oracle,
        # making it look artificially more loaded than it truly is on the
        # very step that follows. Recompute it from the restored
        # queue_depth so the ghost serve has zero lasting side effects.
        _oracle_backend = active[oracle_idx]
        _oracle_backend.cpu = min(1.0, _oracle_backend.queue_depth / _oracle_backend.capacity)

        # ── 4. Router makes its decision ──────────────────────────────────
        action = router.route(active)
        action = min(action, len(active) - 1)

        # ── 5. Execute the routing decision ───────────────────────────────
        result_dict = pool.route_to(action)
        latency = result_dict["latency"]
        error = result_dict["error"]
        sla_v = result_dict["sla_violation"]

        reward = compute_reward(latency, error, sla_v, alpha, beta, gamma)
        regret = max(0.0, oracle_reward - reward)   # regret = how much worse than oracle

        # ── 6. Update router ──────────────────────────────────────────────
        router.update(active, action, reward, result_dict)

        # ── 7. Record metrics ─────────────────────────────────────────────
        result.latencies.append(latency)
        result.errors.append(error)
        result.sla_violations.append(sla_v)
        result.rewards.append(reward)
        result.regrets.append(regret)
        cumulative_reg += regret
        result.cumulative_regret.append(cumulative_reg)

        # Log latency relative to nearest churn event (for P99 churn analysis)
        for churn_t in result.churn_timesteps:
            rel = t - churn_t
            if -churn_window <= rel <= churn_window:
                result.churn_window_latencies.append((rel, latency))

    return result