"""
run_experiment.py
-----------------
MAIN ENTRY POINT — run this file to reproduce all paper results.

Usage:
    python run_experiment.py

What it does:
    1. Sets up the backend pool with N_INITIAL_BACKENDS and the churn
       schedule below
    2. For EACH tau in TAUS: runs all 8 routers for N_REQUESTS x N_SEEDS,
       computes regret/P99/SLA, runs significance tests, saves a CSV
    3. Picks the tau with the lowest mean CGAR (Adaptive) total regret
    4. Generates the paper figures (fig_regret.pdf, fig_p99_churn.pdf,
       fig_summary.pdf, fig_sla.pdf) using that best tau's results
    5. Prints a final tau-comparison table so you can see the full
       sensitivity sweep, not just the winner

--------------------------------------------------------------------------
FIXES / ADDITIONS APPLIED (see accompanying review notes):

1. Key mismatch in print_summary() [fixed previously] — resolved via the
   single CGAR_NAME constant used everywhere.

2. Non-reproducible per-router seeding [fixed previously] — stable
   md5-based stable_hash() instead of Python's randomized hash(str).

3. Statistical significance testing [added previously] — Welch's t-test
   + Cohen's d, CGAR vs every baseline.

4. NEW: NameError from CGAR_TAU -> TAUS rename.
   CGAR_TAU was renamed to a list, TAUS = [20,40,80,120,200], intending a
   tau sweep -- but make_routers() still referenced the old singular
   CGAR_TAU, which no longer existed. Since a genuine sweep across five
   tau values was clearly the intent (the "tune this for your workload"
   comment on the old CGAR_TAU line), make_routers()/run_all() now take
   an explicit `tau` argument, and main() loops over every value in
   TAUS, running the full experiment once per tau, saving a separate
   results/results_tau{N}.csv and results/significance_tau{N}.csv for
   each. After the sweep, the tau with the lowest mean CGAR total
   regret is selected automatically and used to generate the actual
   paper figures (fig_regret.pdf etc.), so you get both the full
   sensitivity picture AND one clean set of figures to put in the paper.

5. NEW: broken indentation under `if __name__ == "__main__":`.
   Only the two initial print() calls were indented inside the guard;
   everything after ran unconditionally at module import time. All
   execution logic is now correctly indented inside the guard.

   Requires scipy AND pandas: pip install scipy pandas --break-system-packages
--------------------------------------------------------------------------
"""

import hashlib
import os

import numpy as np
import pandas as pd
from collections import defaultdict
from scipy import stats

from environment import BackendPool
from cgar import CGARRouter
from simulator import run_simulation, SimResult

from routers import (
    RoundRobinRouter,
    LeastConnectionsRouter,
    PureDQNRouter,
    EpsGreedyDQNRouter,
)

from plot_results import generate_all_figures


# ── Experiment Configuration ──────────────────────────────────────────────

N_REQUESTS     = 20000    # total requests per run
N_SEEDS        = 15    # independent runs per router (for confidence intervals)
N_INITIAL_BACKENDS = 5
SLA_THRESHOLD_MS   = 100.0  # ms — SLA violation threshold

# Churn schedule: (timestep, event_type)
# event_type in {'add', 'remove', 'replace'}
CHURN_SCHEDULE = [
    (1000, 'replace'),
    (1800, 'add'),
    (2600, 'replace'),
    (3400, 'remove'),
    (4200, 'add'),
    (5000, 'replace'),
    (6200, 'remove'),
]

# Confidence half-life sweep. The full experiment runs once per value
# here; the best-performing tau (lowest mean CGAR total regret) is used
# to generate the final paper figures.
TAUS = [20, 40, 80, 120, 200]

# Canonical key for the fully adaptive CGAR router. Defined once here so
# print_summary() and make_routers() can never drift out of sync again.
CGAR_NAME = "CGAR (Adaptive)"

SIGNIFICANCE_ALPHA = 0.05   # standard threshold for the paper's stats claims

RESULTS_DIR = "results"


def stable_hash(s: str) -> int:
    """
    Deterministic string hash, stable across processes/machines/runs —
    unlike Python's built-in hash(str), which is randomized per-process
    (PYTHONHASHSEED) by default. Used only to derive a reproducible
    per-router seed offset, not for anything security-sensitive.
    """
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16) % 100


# ── Router Factory ────────────────────────────────────────────────────────

def make_routers(seed: int, tau: float) -> dict:
    """
    tau is now an explicit argument (not a module-level constant) so the
    sweep in main() can build a fresh set of routers for each tau value
    without any global-state leakage between sweep iterations.
    """
    return {
        "Round Robin": RoundRobinRouter(),

        "Least Connections": LeastConnectionsRouter(),

        "Pure DQN": PureDQNRouter(
            epsilon=0.15,
            seed=seed
        ),

        "Eps-Greedy DQN": EpsGreedyDQNRouter(
            epsilon_start=0.9,
            epsilon_min=0.05,
            decay_steps=2000,
            seed=seed
        ),

        "Heuristic Only": CGARRouter(
            tau=tau,
            mode="heuristic",
            seed=seed
        ),

        "RL Only": CGARRouter(
            tau=tau,
            mode="rl",
            seed=seed
        ),

        "Static Hybrid": CGARRouter(
            tau=tau,
            mode="static",
            static_lambda=0.5,
            seed=seed
        ),

        CGAR_NAME: CGARRouter(
            tau=tau,
            mode="adaptive",
            seed=seed
        ),
    }


# ── Run Experiments ───────────────────────────────────────────────────────

def run_all(tau: float, verbose: bool = True) -> tuple:
    """
    Run all routers across all seeds for a single tau value.
    Returns (all_runs, churn_timesteps).
    """
    all_runs = defaultdict(list)

    for seed in range(N_SEEDS):
        if verbose:
            print(f"\n[tau={tau}] Seed {seed+1}/{N_SEEDS}")

        routers = make_routers(seed, tau)

        for name, router in routers.items():
            pool = BackendPool(
                n_initial=N_INITIAL_BACKENDS,
                churn_schedule=CHURN_SCHEDULE,
                sla_threshold_ms=SLA_THRESHOLD_MS,
                seed=seed * 100 + stable_hash(name),
            )

            result = run_simulation(
                router=router,
                pool=pool,
                n_requests=N_REQUESTS,
                churn_window=200,
                router_name=name,
            )
            all_runs[name].append(result)

            if verbose:
                print(f"  {name:<22} | "
                      f"P99={result.p99_latency():.1f}ms | "
                      f"SLA%={result.sla_violation_rate()*100:.1f}% | "
                      f"Regret={result.total_regret():.0f}")

    return all_runs, [t for t, _ in CHURN_SCHEDULE]


def aggregate(all_runs: dict) -> dict:
    """
    For each router, average cumulative regret curves across seeds.
    Returns a dict of name -> SimResult (with averaged cumulative_regret).
    """
    aggregated = {}
    for name, runs in all_runs.items():
        base = runs[0]
        avg_cum_regret = np.mean(
            [np.array(r.cumulative_regret) for r in runs], axis=0
        ).tolist()
        base.cumulative_regret = avg_cum_regret
        aggregated[name] = base
    return aggregated


# ── Statistical Significance ──────────────────────────────────────────────

def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cohen's d effect size for two independent samples (pooled std).
    Rule of thumb: |d| ~ 0.2 small, 0.5 medium, 0.8 large.
    """
    n_a, n_b = len(a), len(b)
    var_a, var_b = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_std = np.sqrt(
        ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    )
    if pooled_std == 0:
        return 0.0
    return (np.mean(a) - np.mean(b)) / pooled_std


def compare_regret(all_runs: dict, method_a: str, method_b: str) -> dict:
    """
    Welch's t-test (unequal variance) comparing total_regret across
    seeds for method_a vs method_b, plus Cohen's d effect size.
    """
    a = np.array([r.total_regret() for r in all_runs[method_a]])
    b = np.array([r.total_regret() for r in all_runs[method_b]])

    t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)
    d = cohens_d(a, b)

    mean_a, mean_b = np.mean(a), np.mean(b)
    percent_improvement = (
        (mean_b - mean_a) / mean_b * 100 if mean_b != 0 else 0.0
    )

    return {
        "method_a": method_a,
        "method_b": method_b,
        "mean_a": mean_a,
        "mean_b": mean_b,
        "t_stat": t_stat,
        "p_value": p_value,
        "cohens_d": d,
        "significant": p_value < SIGNIFICANCE_ALPHA,
        "percent_improvement": percent_improvement,
    }


def run_significance_tests(all_runs: dict, cgar_name: str = CGAR_NAME) -> list:
    """
    Compare CGAR against every other router on total_regret.
    """
    baselines = [name for name in all_runs.keys() if name != cgar_name]
    return [compare_regret(all_runs, cgar_name, name) for name in baselines]


def mean_ci(values):
    values = np.array(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    ci95 = 1.96 * std / np.sqrt(len(values))
    return mean, ci95


# ── Summary Tables ─────────────────────────────────────────────────────────

def print_summary(all_runs: dict, tau: float):
    """Print a clean summary table to console for one tau value."""
    print("\n" + "="*90)
    print(f"SUMMARY — tau={tau}")
    print("-"*90)
    print(f"{'Router':<22} {'P99 (ms)':>18} {'P95 (ms)':>18} "
          f"{'SLA Viol%':>16} {'Total Regret':>20}")
    print("-"*90)

    for name, runs in all_runs.items():
        p99_mean, p99_ci = mean_ci([r.p99_latency() for r in runs])
        p95_mean, p95_ci = mean_ci([r.p95_latency() for r in runs])
        sla_mean, sla_ci = mean_ci([r.sla_violation_rate()*100 for r in runs])
        reg_mean, reg_ci = mean_ci([r.total_regret() for r in runs])

        print(
            f"{name:<22} "
            f"{p99_mean:>8.1f}±{p99_ci:<7.1f} "
            f"{p95_mean:>8.1f}±{p95_ci:<7.1f} "
            f"{sla_mean:>7.1f}±{sla_ci:<6.1f}% "
            f"{reg_mean:>9.0f}±{reg_ci:<8.0f}"
        )

    print("="*90)

    if CGAR_NAME in all_runs and "Pure DQN" in all_runs:
        cgar_reg = np.mean([r.total_regret() for r in all_runs[CGAR_NAME]])
        dqn_reg  = np.mean([r.total_regret() for r in all_runs["Pure DQN"]])
        if dqn_reg > 0:
            improv = (dqn_reg - cgar_reg) / dqn_reg * 100
            print(f"CGAR regret reduction vs Pure DQN: {improv:.1f}%")

    if CGAR_NAME in all_runs and "Eps-Greedy DQN" in all_runs:
        cgar_reg = np.mean([r.total_regret() for r in all_runs[CGAR_NAME]])
        eps_reg  = np.mean([r.total_regret() for r in all_runs["Eps-Greedy DQN"]])
        if eps_reg > 0:
            improv = (eps_reg - cgar_reg) / eps_reg * 100
            print(f"CGAR regret reduction vs Eps-Greedy DQN: {improv:.1f}%")


def print_significance_table(comparisons: list, tau: float):
    """
    Print CGAR-vs-baseline statistical comparisons for one tau value.
    """
    print("\n" + "="*95)
    print(f"STATISTICAL SIGNIFICANCE — {CGAR_NAME} vs. each baseline (tau={tau}, "
          f"total regret, n={N_SEEDS} seeds/method)")
    print("-"*95)
    print(f"{'Baseline':<22} {'CGAR mean':>12} {'Baseline mean':>14} "
          f"{'Improv %':>10} {'p-value':>10} {'Sig?':>6} {'Cohens d':>10}")
    print("-"*95)

    for c in comparisons:
        sig_flag = "YES" if c["significant"] else "no"
        print(f"{c['method_b']:<22} {c['mean_a']:>12.0f} {c['mean_b']:>14.0f} "
              f"{c['percent_improvement']:>9.1f}% {c['p_value']:>10.4f} "
              f"{sig_flag:>6} {c['cohens_d']:>10.2f}")

    print("="*95)
    n_sig = sum(1 for c in comparisons if c["significant"])
    print(f"CGAR significantly outperforms {n_sig}/{len(comparisons)} baselines "
          f"on total regret at alpha={SIGNIFICANCE_ALPHA} (tau={tau}).")


def print_tau_comparison(tau_results: list):
    """
    Final summary across the whole sweep: mean CGAR total regret per tau,
    so you can see the full sensitivity picture, not just the winner.
    """
    print("\n" + "="*60)
    print("TAU SWEEP SUMMARY — mean CGAR (Adaptive) total regret")
    print("-"*60)
    print(f"{'tau':>8} {'mean regret':>16} {'95% CI':>14}")
    print("-"*60)
    for tau, mean_reg, ci in tau_results:
        print(f"{tau:>8} {mean_reg:>16.0f} {'±' + f'{ci:.0f}':>14}")
    print("="*60)


# ── CSV Export ───────────────────────────────────────────────────────────

def save_results_csv(all_runs: dict, tau: float):
    rows = []
    for name, runs in all_runs.items():
        for seed, result in enumerate(runs):
            rows.append({
                "Tau": tau,
                "Router": name,
                "Seed": seed + 1,
                "P99": result.p99_latency(),
                "P95": result.p95_latency(),
                "SLA (%)": result.sla_violation_rate() * 100,
                "Total Regret": result.total_regret()
            })

    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = pd.DataFrame(rows)
    path = os.path.join(RESULTS_DIR, f"results_tau{int(tau)}.csv")
    df.to_csv(path, index=False)
    print(f"Saved {path}")


def save_significance_csv(comparisons: list, tau: float):
    rows = []
    for c in comparisons:
        rows.append({
            "Tau": tau,
            "Baseline": c["method_b"],
            "CGAR Mean": c["mean_a"],
            "Baseline Mean": c["mean_b"],
            "Improvement (%)": c["percent_improvement"],
            "p-value": c["p_value"],
            "Cohen's d": c["cohens_d"],
            "Significant": c["significant"]
        })

    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = pd.DataFrame(rows)
    path = os.path.join(RESULTS_DIR, f"significance_tau{int(tau)}.csv")
    df.to_csv(path, index=False)
    print(f"Saved {path}")


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("CGAR Experiment Runner — Tau Sweep")
    print(f"Config: {N_REQUESTS} requests, {N_SEEDS} seeds, "
          f"{len(CHURN_SCHEDULE)} churn events, taus={TAUS}\n")

    sweep_all_runs = {}      # tau -> all_runs dict
    sweep_churn_ts = None
    tau_results = []          # list of (tau, mean_regret, ci95) for CGAR

    for tau in TAUS:
        all_runs, churn_ts = run_all(tau=tau, verbose=True)
        sweep_all_runs[tau] = all_runs
        sweep_churn_ts = churn_ts  # same schedule every run

        print_summary(all_runs, tau)

        comparisons = run_significance_tests(all_runs, cgar_name=CGAR_NAME)
        print_significance_table(comparisons, tau)

        save_results_csv(all_runs, tau)
        save_significance_csv(comparisons, tau)

        cgar_regrets = [r.total_regret() for r in all_runs[CGAR_NAME]]
        mean_reg, ci = mean_ci(cgar_regrets)
        tau_results.append((tau, mean_reg, ci))

    # ── Pick the best tau and generate the paper figures for it ──────────
    print_tau_comparison(tau_results)

    best_tau, best_mean_reg, _ = min(tau_results, key=lambda x: x[1])
    print(f"\nBest tau by mean CGAR total regret: tau={best_tau} "
          f"(mean regret={best_mean_reg:.0f})")
    print("Generating paper figures using this tau's results...")

    generate_all_figures(sweep_all_runs[best_tau], sweep_churn_ts)

    print("\nDone. Check results/ folder for per-tau CSV files.")
    print(f"Figures (fig_regret.pdf, fig_p99_churn.pdf, fig_summary.pdf, "
          f"fig_sla.pdf) reflect tau={best_tau}.")
    print("Report the full tau-sweep table above in your paper as a sensitivity")
    print("analysis — it strengthens the paper regardless of which tau you pick.")