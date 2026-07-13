"""
run_experiment.py  —  CGAR v4.0  Final Experiment Runner
=========================================================
Run this to reproduce all paper results:

    python run_experiment.py

Outputs
-------
  Console : summary table with 95% CI and Wilcoxon p-values
  results/ : fig_regret.pdf, fig_p99_churn.pdf, fig_sla.pdf,
             fig_summary.pdf, fig_churn_decomp.pdf

Runtime: ~5-8 minutes on a laptop (numpy MLP, 20 seeds x 8 routers)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from collections import defaultdict
from scipy import stats

from environment import BackendPool
from cgar        import CGARRouter
from simulator   import run_simulation
from routers     import (RoundRobinRouter, LeastConnectionsRouter,
                         PureDQNRouter, EpsGreedyDQNRouter)
from plot_results import generate_all_figures


# ══════════════════════════════════════════════════════════════════════════
# Configuration  (tuned via systematic parameter sweep)
# ══════════════════════════════════════════════════════════════════════════

N_REQUESTS         = 5000
N_SEEDS            = 20
N_INITIAL_BACKENDS = 5
SLA_THRESHOLD_MS   = 100.0

CHURN_SCHEDULE = [
    (1000, 'replace'),
    (2000, 'add'),
    (3000, 'replace'),
    (4000, 'remove'),
]

# Best CGAR hyperparameters (found via grid search in parameter_sweep.py)
CGAR_TAU          = 280.0
CGAR_LAMBDA_FLOOR = 0.85
CGAR_LR           = 0.01


# ══════════════════════════════════════════════════════════════════════════
# Router factory
# ══════════════════════════════════════════════════════════════════════════

def make_routers(seed: int) -> dict:
    return {
        "Round Robin":       RoundRobinRouter(),
        "Least Connections": LeastConnectionsRouter(),
        "Pure DQN":          PureDQNRouter(epsilon=0.15, seed=seed),
        "Eps-Greedy DQN":    EpsGreedyDQNRouter(
                                epsilon_start=0.9, epsilon_min=0.05,
                                decay_steps=2000, seed=seed),
        # ── Ablations ──────────────────────────────────────────────────
        "Heuristic Only":    CGARRouter(
                                tau=CGAR_TAU, lambda_floor=CGAR_LAMBDA_FLOOR,
                                mode="heuristic", lr=CGAR_LR, seed=seed),
        "RL Only":           CGARRouter(
                                tau=CGAR_TAU, lambda_floor=CGAR_LAMBDA_FLOOR,
                                mode="rl", lr=CGAR_LR, seed=seed),
        "Static Hybrid":     CGARRouter(
                                tau=CGAR_TAU, lambda_floor=CGAR_LAMBDA_FLOOR,
                                mode="static", static_lambda=CGAR_LAMBDA_FLOOR,
                                lr=CGAR_LR, seed=seed),
        # ── Proposed method ────────────────────────────────────────────
        "CGAR (Adaptive)":   CGARRouter(
                                tau=CGAR_TAU, lambda_floor=CGAR_LAMBDA_FLOOR,
                                mode="adaptive", lr=CGAR_LR, seed=seed),
    }


# ══════════════════════════════════════════════════════════════════════════
# Run
# ══════════════════════════════════════════════════════════════════════════

def run_all(verbose=True):
    all_runs = defaultdict(list)

    for seed in range(N_SEEDS):
        if verbose:
            print(f"\n── Seed {seed+1}/{N_SEEDS} " + "─"*38)
        routers = make_routers(seed)

        for name, router in routers.items():
            pool = BackendPool(
                n_initial=N_INITIAL_BACKENDS,
                churn_schedule=CHURN_SCHEDULE,
                sla_threshold_ms=SLA_THRESHOLD_MS,
                seed=seed * 100 + abs(hash(name)) % 97,
            )
            result = run_simulation(
                router=router, pool=pool,
                n_requests=N_REQUESTS, churn_window=300,
                router_name=name,
            )
            all_runs[name].append(result)

            if verbose:
                print(f"  {name:<20} | P99={result.p99_latency():6.1f}ms | "
                      f"SLA={result.sla_violation_rate()*100:5.1f}% | "
                      f"Regret={result.total_regret():7.0f}")

    return all_runs, [t for t, _ in CHURN_SCHEDULE]


# ══════════════════════════════════════════════════════════════════════════
# Churn-window regret decomposition (paper Section 7 primary result)
# ══════════════════════════════════════════════════════════════════════════

def decompose_regret(result, churn_ts, window=300):
    """
    Split total regret into:
      R_churn  : regret in [t, t+window] after each churn event
      R_steady : regret outside all churn windows
    """
    regs = np.array(result.regrets)
    n    = len(regs)
    mask = np.zeros(n, dtype=bool)
    for ct in churn_ts:
        mask[ct:min(ct + window, n)] = True
    return regs[mask].sum(), regs[~mask].sum()


# ══════════════════════════════════════════════════════════════════════════
# Summary table with statistics
# ══════════════════════════════════════════════════════════════════════════

def print_summary(all_runs, churn_ts):
    print("\n" + "="*90)
    print(f"{'Router':<22} {'P99':>7} {'P95':>7} {'SLA%':>6} "
          f"{'Regret':>9} {'95% CI':>22} {'R_churn':>9} {'R_steady':>9}")
    print("-"*90)

    for name, runs in all_runs.items():
        p99s  = [r.p99_latency()          for r in runs]
        p95s  = [r.p95_latency()          for r in runs]
        slas  = [r.sla_violation_rate()   for r in runs]
        regs  = [r.total_regret()         for r in runs]
        cw    = [decompose_regret(r, churn_ts)[0] for r in runs]
        ss    = [decompose_regret(r, churn_ts)[1] for r in runs]

        ci = stats.t.interval(0.95, len(regs)-1,
                               loc=np.mean(regs), scale=stats.sem(regs))
        print(f"{name:<22} {np.mean(p99s):>7.1f} {np.mean(p95s):>7.1f} "
              f"{np.mean(slas)*100:>5.1f}% {np.mean(regs):>9.0f} "
              f"[{ci[0]:>9.0f},{ci[1]:>9.0f}] "
              f"{np.mean(cw):>9.0f} {np.mean(ss):>9.0f}")

    print("="*90)

    # Wilcoxon signed-rank tests vs CGAR
    print("\nStatistical significance (Wilcoxon signed-rank, two-sided):")
    cgar_regs = [r.total_regret() for r in all_runs["CGAR (Adaptive)"]]
    baselines = ["Round Robin", "Least Connections", "Pure DQN",
                 "Eps-Greedy DQN", "Static Hybrid", "RL Only"]
    for b in baselines:
        if b not in all_runs:
            continue
        b_regs = [r.total_regret() for r in all_runs[b]]
        _, p   = stats.wilcoxon(cgar_regs, b_regs)
        d      = (np.mean(b_regs) - np.mean(cgar_regs)) / max(np.mean(b_regs), 1) * 100
        sig    = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        result = "CGAR wins" if d > 0 else "baseline wins"
        print(f"  CGAR vs {b:<22}: {abs(d):5.1f}% {result:<14} p={p:.4f} {sig}")


# ══════════════════════════════════════════════════════════════════════════
# Aggregation for plotting
# ══════════════════════════════════════════════════════════════════════════

def aggregate(all_runs):
    aggregated = {}
    for name, runs in all_runs.items():
        base   = runs[0]
        curves = [np.array(r.cumulative_regret) for r in runs]
        min_len = min(len(c) for c in curves)
        base.cumulative_regret     = np.mean([c[:min_len] for c in curves], axis=0).tolist()
        base.cumulative_regret_std = np.std( [c[:min_len] for c in curves], axis=0).tolist()
        aggregated[name] = base
    return aggregated


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  CGAR v4.0  —  Final Paper Experiment Runner            ║")
    print(f"║  {N_REQUESTS} req  ·  {N_SEEDS} seeds  ·  {len(CHURN_SCHEDULE)} churn events  ·  tau={CGAR_TAU:.0f}  ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    all_runs, churn_ts = run_all(verbose=True)
    print_summary(all_runs, churn_ts)

    aggregated = aggregate(all_runs)
    generate_all_figures(all_runs, aggregated, churn_ts)

    print("\n✓ Complete. All figures saved to results/")
    print("  Replace placeholder figures in your paper with these real results.")