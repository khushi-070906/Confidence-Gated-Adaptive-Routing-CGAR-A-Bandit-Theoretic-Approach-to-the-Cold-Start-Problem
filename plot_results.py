"""
plot_results.py
---------------
Generates all figures for the paper from real simulation results.
Run this after run_experiment.py produces results.

Figures produced:
  fig_regret.pdf       — cumulative regret over time (primary result)
  fig_p99_churn.pdf    — P99 latency around churn events
  fig_sla.pdf          — SLA violation rate comparison
  fig_summary.pdf      — Summary bar chart of all metrics
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "legend.fontsize": 9.5,
    "figure.dpi": 150,
})

COLORS = {
    "Round Robin": "#888888",
    "Least Connections": "#e67e22",
    "Pure DQN": "#c0392b",
    "Eps-Greedy DQN": "#8e44ad",
    "Heuristic Only": "#16a085",
    "RL Only": "#d35400",
    "Static Hybrid": "#2980b9",
    "CGAR (Adaptive)": "#1a5fa8",
}

STYLES = {
    "Round Robin": ("--", 1.4),
    "Least Connections": (":", 1.4),
    "Pure DQN": ("-", 1.4),
    "Eps-Greedy DQN": ("-.", 1.4),
    "Heuristic Only": ("--", 1.5),
    "RL Only": ("-.", 1.5),
    "Static Hybrid": (":", 1.7),
    "CGAR (Adaptive)": ("-", 2.8),
}

OUTPUT_DIR = os.path.dirname(__file__)


def plot_cumulative_regret(all_results, churn_timesteps, save_path):
    fig, ax = plt.subplots(figsize=(7, 4.2))

    for name, runs in all_results.items():
        curves = np.array([r.cumulative_regret for r in runs])

        mean_curve = np.mean(curves, axis=0)
        std_curve = np.std(curves, axis=0)
        ci = 1.96 * std_curve / np.sqrt(len(runs))

        ls, lw = STYLES.get(name, ("-", 1.5))
        color = COLORS.get(name, "black")

        ax.plot(mean_curve, label=name, color=color,
                linestyle=ls, linewidth=lw)

        ax.fill_between(
            range(len(mean_curve)),
            mean_curve - ci,
            mean_curve + ci,
            color=color,
            alpha=0.15
        )

    for ct in churn_timesteps:
        ax.axvline(ct, color="#bbbbbb", linewidth=0.9, linestyle=":")

    ax.set_xlabel("Request index")
    ax.set_ylabel("Cumulative regret")
    ax.set_title("Cumulative Regret with 95% Confidence Interval")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_p99_churn(all_results, save_path):
    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    names = list(all_results.keys())
    p99s = []

    for name, runs in all_results.items():
        vals = [r.p99_latency() for r in runs]
        p99s.append(np.mean(vals))

    x = np.arange(len(names))
    colors = [COLORS.get(n, "#555555") for n in names]

    bars = ax.bar(x, p99s, color=colors)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("P99 Latency (ms)")
    ax.set_title("P99 Latency Comparison")

    for bar, val in zip(bars, p99s):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height(),
                f"{val:.1f}",
                ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_summary_bars(all_results, save_path):
    names = list(all_results.keys())

    p99s = [np.mean([r.p99_latency() for r in runs]) for runs in all_results.values()]
    slas = [np.mean([r.sla_violation_rate()*100 for r in runs]) for runs in all_results.values()]
    regrets = [np.mean([r.total_regret()/1000 for r in runs]) for runs in all_results.values()]

    x = np.arange(len(names))
    colors = [COLORS.get(n, "#555555") for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    metrics = [
        (p99s, "P99 Latency", "ms"),
        (slas, "SLA Violation Rate", "%"),
        (regrets, "Total Regret", "×10³")
    ]

    for ax, (vals, title, ylabel) in zip(axes, metrics):
        bars = ax.bar(x, vals, color=colors)

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_sla_over_time(all_results, window_size=100, save_path="fig_sla.pdf"):
    fig, ax = plt.subplots(figsize=(7, 4))

    for name, runs in all_results.items():
        curves = []

        for r in runs:
            sla = np.array(r.sla_violations, dtype=float)
            rolling = np.convolve(sla, np.ones(window_size)/window_size, mode='valid')
            curves.append(rolling)

        curves = np.array(curves)
        mean_curve = np.mean(curves, axis=0)

        ls, lw = STYLES.get(name, ("-", 1.5))
        ax.plot(mean_curve*100,
                label=name,
                color=COLORS.get(name, "black"),
                linestyle=ls,
                linewidth=lw)

    ax.set_xlabel("Request index")
    ax.set_ylabel("SLA violation rate (%)")
    ax.set_title("Rolling SLA Violation Rate")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def generate_all_figures(all_results, churn_timesteps):
    out = OUTPUT_DIR

    print("\nGenerating figures...")

    plot_cumulative_regret(
        all_results,
        churn_timesteps,
        save_path=os.path.join(out, "fig_regret.pdf")
    )

    plot_p99_churn(
        all_results,
        save_path=os.path.join(out, "fig_p99_churn.pdf")
    )

    plot_summary_bars(
        all_results,
        save_path=os.path.join(out, "fig_summary.pdf")
    )

    plot_sla_over_time(
        all_results,
        save_path=os.path.join(out, "fig_sla.pdf")
    )

    print("All figures saved.")