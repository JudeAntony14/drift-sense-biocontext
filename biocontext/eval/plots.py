"""Generates the benchmark comparison figures from the CSV outputs of
`biocontext.eval.benchmark`.

Usage:
    python -m biocontext.eval.plots --raw results/tables/benchmark_raw.csv --out-dir results/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

METHOD_ORDER = ["baseline", "retina_only", "context_only", "biocontext"]
METHOD_LABELS = {
    "baseline": "Classical\nBaseline",
    "retina_only": "Retina\n(center-surround)",
    "context_only": "Ant-Context\n(surround verify)",
    "biocontext": "BioContext\n(combined)",
}
METHOD_COLORS = {
    "baseline": "#9aa0a6",
    "retina_only": "#4c8bf5",
    "context_only": "#f5a623",
    "biocontext": "#2ecc71",
}


def _order(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["method"] = pd.Categorical(df["method"], categories=METHOD_ORDER, ordered=True)
    return df.sort_values("method")


def plot_overall_bars(raw: pd.DataFrame, out_dir: Path) -> None:
    summary = raw.groupby("method").agg(mean_error=("error_px", "mean"),
                                         success_rate=("success", "mean")).reset_index()
    summary = _order(summary)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = [METHOD_COLORS[m] for m in summary["method"]]
    labels = [METHOD_LABELS[m] for m in summary["method"]]

    axes[0].bar(labels, summary["mean_error"], color=colors)
    axes[0].set_ylabel("Mean localization error (px)")
    axes[0].set_title("Mean error - lower is better")
    for i, v in enumerate(summary["mean_error"]):
        axes[0].text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)

    axes[1].bar(labels, summary["success_rate"] * 100, color=colors)
    axes[1].set_ylabel("Success rate (%) @ 15px")
    axes[1].set_title("Success rate - higher is better")
    axes[1].set_ylim(0, 100)
    for i, v in enumerate(summary["success_rate"] * 100):
        axes[1].text(i, v, f"{v:.0f}%", ha="center", va="bottom", fontsize=9)

    fig.suptitle("BioContext Benchmark - Overall Method Comparison", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "overall_comparison.png", dpi=150)
    plt.close(fig)


def plot_by_factor(raw: pd.DataFrame, out_dir: Path) -> None:
    factors = [f for f in raw["factor"].unique() if f != "other"]
    n = len(factors)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for ax, factor in zip(axes, factors):
        sub = raw[raw["factor"] == factor]
        agg = sub.groupby("method")["error_px"].mean().reindex(METHOD_ORDER)
        colors = [METHOD_COLORS[m] for m in agg.index]
        ax.bar([METHOD_LABELS[m] for m in agg.index], agg.values, color=colors)
        ax.set_title(f"Stress factor: {factor}")
        ax.set_ylabel("Mean error (px)")
        ax.tick_params(axis="x", labelsize=8)

    for ax in axes[len(factors):]:
        ax.axis("off")

    fig.suptitle("BioContext Benchmark - Error by Stress Factor", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "error_by_factor.png", dpi=150)
    plt.close(fig)


def plot_success_by_factor(raw: pd.DataFrame, out_dir: Path) -> None:
    factors = [f for f in raw["factor"].unique() if f != "other"]
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.2
    x = range(len(factors))
    for i, method in enumerate(METHOD_ORDER):
        rates = []
        for factor in factors:
            sub = raw[(raw["factor"] == factor) & (raw["method"] == method)]
            rates.append(sub["success"].mean() * 100 if len(sub) else 0)
        ax.bar([xi + i * width for xi in x], rates, width=width, label=METHOD_LABELS[method].replace("\n", " "),
               color=METHOD_COLORS[method])
    ax.set_xticks([xi + 1.5 * width for xi in x])
    ax.set_xticklabels(factors, rotation=20, ha="right")
    ax.set_ylabel("Success rate (%) @ 15px")
    ax.set_title("Success Rate by Stress Factor and Method")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "success_by_factor.png", dpi=150)
    plt.close(fig)


def plot_runtime(raw: pd.DataFrame, out_dir: Path) -> None:
    agg = raw.groupby("method")["runtime_s"].mean().reindex(METHOD_ORDER)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = [METHOD_COLORS[m] for m in agg.index]
    ax.bar([METHOD_LABELS[m] for m in agg.index], agg.values * 1000, color=colors)
    ax.set_ylabel("Mean runtime (ms) per case")
    ax.set_title("Runtime Comparison")
    for i, v in enumerate(agg.values * 1000):
        ax.text(i, v, f"{v:.0f}ms", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "runtime_comparison.png", dpi=150)
    plt.close(fig)


def plot_error_distribution(raw: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    data = [raw[raw["method"] == m]["error_px"].clip(upper=500) for m in METHOD_ORDER]
    bp = ax.boxplot(data, labels=[METHOD_LABELS[m].replace("\n", " ") for m in METHOD_ORDER],
                     patch_artist=True, showfliers=False)
    for patch, m in zip(bp["boxes"], METHOD_ORDER):
        patch.set_facecolor(METHOD_COLORS[m])
        patch.set_alpha(0.7)
    ax.set_ylabel("Localization error (px, clipped at 500)")
    ax.set_title("Error Distribution Across All Benchmark Cases")
    fig.tight_layout()
    fig.savefig(out_dir / "error_distribution.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=str, default="results/tables/benchmark_raw.csv")
    ap.add_argument("--out-dir", type=str, default="results/figures")
    args = ap.parse_args()

    raw = pd.read_csv(args.raw)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_overall_bars(raw, out_dir)
    plot_by_factor(raw, out_dir)
    plot_success_by_factor(raw, out_dir)
    plot_runtime(raw, out_dir)
    plot_error_distribution(raw, out_dir)
    print(f"Saved figures to {out_dir}")


if __name__ == "__main__":
    main()
