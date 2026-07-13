"""
plot_results.py  —  Publication-quality figures for CGAR paper
==============================================================
Generates 5 figures:
  fig_regret.pdf       — cumulative regret + 95% CI bands  (primary result)
  fig_p99_churn.pdf    — P99 latency around churn events
  fig_sla.pdf          — rolling SLA violation rate
  fig_summary.pdf      — bar chart of all metrics
  fig_churn_decomp.pdf — churn-window vs steady-state regret (key paper figure)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "legend.fontsize":   9,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "figure.dpi":        180,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linestyle":    "--",
})

STYLE = {
    "Round Robin":       ("#999999", "--",  1.4),
    "Least Connections": ("#e67e22", ":",   1.6),
    "Pure DQN":          ("#c0392b", "-",   1.4),
    "Eps-Greedy DQN":    ("#8e44ad", "-.",  1.4),
    "Heuristic Only":    ("#27ae60", "--",  1.2),
    "RL Only":           ("#e74c3c", ":",   1.2),
    "Static Hybrid":     ("#f39c12", "-.",  1.2),
    "CGAR (Adaptive)":   ("#1a5276", "-",   2.8),
}

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)


# ── Figure 1: Cumulative Regret with 95% CI ───────────────────────────────

def plot_cumulative_regret(aggregated, churn_ts, n_seeds=20):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    for name, res in aggregated.items():
        c, ls, lw = STYLE.get(name, ("#555", "-", 1.4))
        y = np.array(res.cumulative_regret)
        x = np.arange(len(y))
        ax.plot(x, y, label=name, color=c, linestyle=ls, linewidth=lw)
        if hasattr(res, "cumulative_regret_std"):
            std    = np.array(res.cumulative_regret_std)
            margin = 1.96 * std / np.sqrt(n_seeds)
            ax.fill_between(x, y - margin, y + margin, color=c, alpha=0.08)

    ymax = ax.get_ylim()[1]
    for i, ct in enumerate(churn_ts):
        ax.axvline(ct, color="#aaaaaa", linewidth=0.9, linestyle=":")
        ax.text(ct + 30, ymax * 0.04, f"C{i+1}",
                fontsize=8, color="#888888", va="bottom")

    ax.set_xlabel("Request index $t$")
    ax.set_ylabel("Cumulative regret $R(T)$")
    ax.set_title("Cumulative Regret vs. Request Index\n"
                 f"(mean ± 95% CI, {n_seeds} seeds; C1–C4 = churn events)")
    ax.legend(loc="upper left", ncol=2, framealpha=0.9)
    plt.tight_layout()
    path = os.path.join(OUT, "fig_regret.pdf")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


# ── Figure 2: P99 Latency Around Churn Events ────────────────────────────

def plot_p99_churn(all_runs, window=300):
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    bins    = np.arange(-window, window + 1, 30)
    centers = (bins[:-1] + bins[1:]) / 2

    for name, runs in all_runs.items():
        c, ls, lw = STYLE.get(name, ("#555", "-", 1.4))
        all_rel, all_lat = [], []
        for r in runs:
            for rel, lat in r.churn_window_latencies:
                all_rel.append(rel)
                all_lat.append(lat)
        if not all_rel:
            continue
        all_rel = np.array(all_rel)
        all_lat = np.array(all_lat)
        p99_bins = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (all_rel >= lo) & (all_rel < hi)
            p99_bins.append(np.percentile(all_lat[mask], 99)
                            if mask.sum() > 5 else np.nan)
        ax.plot(centers, p99_bins, label=name,
                color=c, linestyle=ls, linewidth=lw)

    ax.axvline(0, color="black", linewidth=1.2)
    ax.text(10, ax.get_ylim()[1] * 0.97,
            "churn\nevent", fontsize=8, va="top", color="#333")
    ax.set_xlabel("Requests relative to churn event ($t = 0$)")
    ax.set_ylabel("P99 latency (ms)")
    ax.set_title("P99 Latency Around Backend Pool Churn Events\n"
                 "(aggregated across all churn events and seeds)")
    ax.legend(loc="upper right", ncol=2, framealpha=0.9)
    plt.tight_layout()
    path = os.path.join(OUT, "fig_p99_churn.pdf")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


# ── Figure 3: Rolling SLA Violation Rate ─────────────────────────────────

def plot_sla(aggregated, churn_ts, window=150):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for name, res in aggregated.items():
        c, ls, lw = STYLE.get(name, ("#555", "-", 1.4))
        sla     = np.array(res.sla_violations, dtype=float)
        rolling = np.convolve(sla, np.ones(window) / window, mode="valid")
        ax.plot(rolling * 100, label=name, color=c, linestyle=ls, linewidth=lw)
    for ct in churn_ts:
        ax.axvline(ct, color="#aaaaaa", linewidth=0.9, linestyle=":")
    ax.set_xlabel("Request index $t$")
    ax.set_ylabel(f"SLA violation rate (%, {window}-req rolling avg)")
    ax.set_title("Rolling SLA Violation Rate Over Time")
    ax.legend(loc="upper right", ncol=2, framealpha=0.9)
    plt.tight_layout()
    path = os.path.join(OUT, "fig_sla.pdf")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


# ── Figure 4: Summary Bar Chart ───────────────────────────────────────────

def plot_summary(all_runs):
    names  = list(all_runs.keys())
    colors = [STYLE.get(n, ("#555", "-", 1))[0] for n in names]

    p99s = [np.mean([r.p99_latency()        for r in all_runs[n]]) for n in names]
    slas = [np.mean([r.sla_violation_rate() for r in all_runs[n]]) * 100 for n in names]
    regs = [np.mean([r.total_regret()       for r in all_runs[n]]) / 1000 for n in names]
    p99e = [np.std([r.p99_latency()         for r in all_runs[n]]) / np.sqrt(len(all_runs[n])) for n in names]
    slae = [np.std([r.sla_violation_rate()  for r in all_runs[n]]) / np.sqrt(len(all_runs[n])) * 100 for n in names]
    rege = [np.std([r.total_regret()        for r in all_runs[n]]) / np.sqrt(len(all_runs[n])) / 1000 for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    x = np.arange(len(names))

    for ax, vals, errs, title, ylabel in zip(
        axes,
        [p99s, slas, regs],
        [p99e, slae, rege],
        ["P99 Latency", "SLA Violation Rate", "Total Regret (×10³)"],
        ["ms", "%", "regret units"],
    ):
        bars = ax.bar(x, vals, yerr=errs, color=colors,
                      edgecolor="white", linewidth=0.5,
                      error_kw=dict(ecolor="#555", capsize=3, lw=1.2))
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=32, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=9)
        for bar, val, err in zip(bars, vals, errs):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + err + max(vals) * 0.01,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=7.5)

    plt.suptitle(f"Summary Metrics — Mean ± SE across {len(list(all_runs.values())[0])} Seeds",
                 fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(OUT, "fig_summary.pdf")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


# ── Figure 5: Churn-Window vs Steady-State Regret Decomposition ──────────
# This is the KEY figure for the paper — it shows CGAR's specific advantage
# around churn events, which is the paper's central claim.

def decompose_regret(result, churn_ts, window=300):
    regs = np.array(result.regrets)
    n    = len(regs)
    mask = np.zeros(n, dtype=bool)
    for ct in churn_ts:
        mask[ct:min(ct + window, n)] = True
    return regs[mask].sum(), regs[~mask].sum()


def plot_churn_decomposition(all_runs, churn_ts):
    names  = list(all_runs.keys())
    colors = [STYLE.get(n, ("#555", "-", 1))[0] for n in names]

    churn_regs  = []
    steady_regs = []
    churn_errs  = []
    steady_errs = []

    for name in names:
        cw = [decompose_regret(r, churn_ts)[0] / 1000 for r in all_runs[name]]
        ss = [decompose_regret(r, churn_ts)[1] / 1000 for r in all_runs[name]]
        n  = len(cw)
        churn_regs.append(np.mean(cw))
        steady_regs.append(np.mean(ss))
        churn_errs.append(np.std(cw) / np.sqrt(n))
        steady_errs.append(np.std(ss) / np.sqrt(n))

    x  = np.arange(len(names))
    w  = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))

    b1 = ax.bar(x - w/2, churn_regs,  w, yerr=churn_errs,
                label="Churn-window regret $R_{\\mathrm{churn}}$",
                color=colors, alpha=0.95,
                error_kw=dict(ecolor="#333", capsize=3, lw=1.2))
    b2 = ax.bar(x + w/2, steady_regs, w, yerr=steady_errs,
                label="Steady-state regret $R_{\\mathrm{steady}}$",
                color=colors, alpha=0.45, hatch="//",
                error_kw=dict(ecolor="#333", capsize=3, lw=1.2))

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Regret (×10³)")
    ax.set_title("Churn-Window vs. Steady-State Regret Decomposition\n"
                 "(solid = churn-window  ///  = steady-state)")
    ax.legend(loc="upper right", framealpha=0.9)

    for i, (cv, sv) in enumerate(zip(churn_regs, steady_regs)):
        ax.text(i - w/2, cv + churn_errs[i] + 0.2,
                f"{cv:.1f}", ha="center", va="bottom", fontsize=7.5)
        ax.text(i + w/2, sv + steady_errs[i] + 0.2,
                f"{sv:.1f}", ha="center", va="bottom", fontsize=7.5)

    plt.tight_layout()
    path = os.path.join(OUT, "fig_churn_decomp.pdf")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


# ── Master call ───────────────────────────────────────────────────────────

def generate_all_figures(all_runs, aggregated, churn_ts):
    n_seeds = len(list(all_runs.values())[0])
    print(f"\nGenerating publication figures ({n_seeds} seeds)...")
    plot_cumulative_regret(aggregated, churn_ts, n_seeds)
    plot_p99_churn(all_runs, window=300)
    plot_sla(aggregated, churn_ts)
    plot_summary(all_runs)
    plot_churn_decomposition(all_runs, churn_ts)
    print(f"All 5 figures saved to results/")