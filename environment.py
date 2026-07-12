"""
environment.py
--------------
Simulates a backend pool with realistic latency, queue depth,
CPU utilisation, and churn events (autoscale add/remove).

Latency model:
  base_latency + queue_penalty * queue_depth + noise
  When a backend is overloaded (cpu > 0.85), latency spikes.
"""

import numpy as np


class Backend:
    """Represents one backend instance in the pool."""

    def __init__(self, backend_id: int, base_latency: float, capacity: int, rng: np.random.Generator):
        self.id = backend_id
        self.base_latency = base_latency   # ms, true quality of this backend
        self.capacity = capacity           # max concurrent requests before overload
        self.rng = rng

        # Runtime state
        self.queue_depth = 0
        self.cpu = 0.0
        self.total_served = 0
        self.alive = True

    def serve(self) -> dict:
        """
        Simulate serving one request.
        Returns a dict with latency, error, and current state.
        """
        if not self.alive:
            return {"latency": 9999.0, "error": True, "queue": 0, "cpu": 0.0}

        self.queue_depth += 1
        self.cpu = min(1.0, self.queue_depth / self.capacity)

        # Latency increases with load; spikes when overloaded
        queue_penalty = 5.0 * self.queue_depth
        spike = 150.0 if self.cpu > 0.85 else 0.0
        noise = self.rng.normal(0, 5.0)
        latency = max(1.0, self.base_latency + queue_penalty + spike + noise)

        # Error probability rises with overload
        error_prob = 0.01 + 0.15 * max(0.0, self.cpu - 0.8)
        error = self.rng.random() < error_prob

        # Queue drains after serving (simplified: instant service)
        self.queue_depth = max(0, self.queue_depth - 1)
        self.total_served += 1

        return {
            "latency": latency,
            "error": error,
            "queue": self.queue_depth,
            "cpu": self.cpu,
        }

    def reset_stats(self):
        """Called when a backend re-enters the pool after churn."""
        self.queue_depth = 0
        self.cpu = 0.0
        self.total_served = 0


class BackendPool:
    """
    Manages a dynamic pool of backends with churn events.

    Churn events occur at scheduled time steps:
      - 'add':    add a new backend instance
      - 'remove': remove an existing backend instance
      - 'replace': remove one and add a fresh one (rolling deploy)
    """

    def __init__(self, n_initial: int, churn_schedule: list, sla_threshold_ms: float, seed: int = 42):
        """
        Parameters
        ----------
        n_initial       : initial number of backends
        churn_schedule  : list of (timestep, event_type) tuples
                          event_type in {'add', 'remove', 'replace'}
        sla_threshold_ms: latency above which an SLA violation is counted
        seed            : random seed
        """
        self.rng = np.random.default_rng(seed)
        self.sla_threshold = sla_threshold_ms
        self.churn_schedule = sorted(churn_schedule, key=lambda x: x[0])
        self._churn_index = 0
        self._next_id = n_initial

        # Create initial backends with varied quality (base latency)
        self.backends = []
        for i in range(n_initial):
            bl = self.rng.uniform(40, 110)   # ms — heterogeneous backends
            cap = int(self.rng.integers(5, 15))
            self.backends.append(Backend(i, bl, cap, self.rng))

        # Track which timesteps had churn (for analysis)
        self.churn_events_log = []

    @property
    def active_backends(self):
        return [b for b in self.backends if b.alive]

    def step_churn(self, t: int) -> bool:
        """
        Apply any churn events scheduled at timestep t.
        Returns True if a churn event fired.
        """
        fired = False
        while (self._churn_index < len(self.churn_schedule) and
               self.churn_schedule[self._churn_index][0] <= t):
            _, event = self.churn_schedule[self._churn_index]
            self._churn_index += 1
            self._apply_churn(event, t)
            fired = True
        return fired

    def _apply_churn(self, event: str, t: int):
        active = self.active_backends
        if event in ('remove', 'replace') and len(active) > 1:
            # Remove a random backend
            victim = self.rng.choice(active)
            victim.alive = False
            self.churn_events_log.append((t, 'remove', victim.id))

        if event in ('add', 'replace'):
            # Add a fresh backend (unknown quality — triggers cold start)
            bl = self.rng.uniform(40, 110)
            cap = int(self.rng.integers(5, 15))
            new_b = Backend(self._next_id, bl, cap, self.rng)
            self._next_id += 1
            self.backends.append(new_b)
            self.churn_events_log.append((t, 'add', new_b.id))

    def route_to(self, backend_idx: int) -> dict:
        """Route a request to the backend at position backend_idx in active list."""
        active = self.active_backends
        if backend_idx >= len(active):
            backend_idx = 0   # safety fallback
        b = active[backend_idx]
        result = b.serve()
        result["sla_violation"] = result["latency"] > self.sla_threshold
        result["backend_id"] = b.id
        return result

    def oracle_best(self) -> int:
        """
        Oracle: index of the currently best backend (lowest base latency).
        Used only for regret computation — routers do not see this.
        """
        active = self.active_backends
        best = min(range(len(active)), key=lambda i: active[i].base_latency)
        return best
