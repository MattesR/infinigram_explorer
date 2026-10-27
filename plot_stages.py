"""
Plot recall/nDCG progression across pipeline stages.

Usage:
    from plot_stages import plot_stage_progression

    plot_stage_progression("beir/fever/metrics.csv", output_path="beir/fever/stage_progression.pdf")

    # Multiple datasets
    plot_stage_comparison(["beir/fever/metrics.csv", "beir/fiqa/metrics.csv"])
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.6,
})

STAGE_ORDER = ["pool", "biencoder", "crossencoder"]
STAGE_LABELS = {"pool": "Retrieval Pool", "biencoder": "Bi-Encoder", "crossencoder": "Cross-Encoder"}
STAGE_COLORS = {"pool": "#2ecc71", "biencoder": "#3498db", "crossencoder": "#e74c3c"}


def plot_stage_progression(
    csv_path: str,
    output_path: str = None,
    metric: str = "ndcg",
    k: int = 10,
    figsize: tuple = (5, 3.5),
):
    """
    Plot metric progression across stages as box plots + connected means.

    Args:
        csv_path: Path to per_query metrics CSV.
        metric: "ndcg", "recall", or "precision".
        k: Cutoff (10, 100, 1000).
    """
    df = pd.read_csv(csv_path)
    run_name = df["run"].iloc[0] if "run" in df.columns else "run"
    col = f"{metric}_{k}"

    if col not in df.columns:
        print(f"Column {col} not found in {csv_path}")
        return

    stages_present = [s for s in STAGE_ORDER if s in df["stage"].values]

    fig, ax = plt.subplots(figsize=figsize)

    # Box plots
    box_data = []
    positions = []
    colors = []
    for i, stage in enumerate(stages_present):
        sdf = df[df["stage"] == stage]
        vals = sdf[col].dropna().values
        box_data.append(vals)
        positions.append(i)
        colors.append(STAGE_COLORS.get(stage, "#888"))

    bp = ax.boxplot(box_data, positions=positions, patch_artist=True,
                     showfliers=False, widths=0.5,
                     medianprops=dict(color='white', linewidth=1.5))

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Connected means
    means = [np.mean(d) for d in box_data]
    ax.plot(positions, means, 'o-', color='#2c3e50', linewidth=2,
            markersize=6, zorder=5, label='Mean')

    # Annotate means
    for i, (pos, mean) in enumerate(zip(positions, means)):
        ax.annotate(f'{mean:.3f}', xy=(pos, mean),
                   xytext=(8, 5), textcoords='offset points',
                   fontsize=8, color='#2c3e50', fontweight='bold')

    ax.set_xticks(positions)
    ax.set_xticklabels([STAGE_LABELS.get(s, s) for s in stages_present])
    ax.set_ylabel(f'{metric.upper()}@{k}')
    ax.set_title(f'{run_name}: {metric.upper()}@{k} across stages')
    ax.set_ylim(-0.05, 1.05)

    if output_path is None:
        output_path = csv_path.replace(".csv", f"_stage_{metric}{k}.pdf")

    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_stage_multi_metric(
    csv_path: str,
    output_path: str = None,
    metrics: list = None,
    figsize: tuple = (7, 4),
):
    """
    Grouped bar chart showing multiple metrics across stages.
    """
    if metrics is None:
        metrics = [("ndcg", 10), ("recall", 10), ("recall", 100), ("recall", 1000)]

    df = pd.read_csv(csv_path)
    run_name = df["run"].iloc[0] if "run" in df.columns else "run"

    stages_present = [s for s in STAGE_ORDER if s in df["stage"].values]
    n_stages = len(stages_present)
    n_metrics = len(metrics)

    fig, ax = plt.subplots(figsize=figsize)

    bar_width = 0.7 / n_metrics
    metric_colors = ['#2c3e50', '#e74c3c', '#3498db', '#2ecc71', '#e67e22']

    for j, (metric, k) in enumerate(metrics):
        col = f"{metric}_{k}"
        if col not in df.columns:
            continue

        means = []
        for stage in stages_present:
            sdf = df[df["stage"] == stage]
            means.append(sdf[col].mean())

        x = np.arange(n_stages) + j * bar_width - (n_metrics - 1) * bar_width / 2
        bars = ax.bar(x, means, bar_width * 0.9,
                       color=metric_colors[j % len(metric_colors)],
                       alpha=0.8, label=f'{metric.upper()}@{k}',
                       edgecolor='white', linewidth=0.3)

        for bar, mean in zip(bars, means):
            if mean > 0.02:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f'{mean:.3f}', ha='center', va='bottom', fontsize=6.5)

    ax.set_xticks(np.arange(n_stages))
    ax.set_xticklabels([STAGE_LABELS.get(s, s) for s in stages_present])
    ax.set_ylabel('Score')
    ax.set_title(f'{run_name}: Metrics across pipeline stages')
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper left', framealpha=0.9, edgecolor='none')

    if output_path is None:
        output_path = csv_path.replace(".csv", "_stage_multi.pdf")

    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_recall_flow(
    csv_path: str,
    output_path: str = None,
    figsize: tuple = (5, 3.5),
):
    """
    Sankey-style flow showing how recall changes across stages.
    Shows pool_recall → biencoder R@1000 → crossencoder R@100.
    """
    df = pd.read_csv(csv_path)
    run_name = df["run"].iloc[0] if "run" in df.columns else "run"

    stages_present = [s for s in STAGE_ORDER if s in df["stage"].values]

    # Define which metric represents each stage's "output"
    stage_metric = {
        "pool": "pool_recall",
        "biencoder": "recall_1000",
        "crossencoder": "recall_100",
    }

    fig, ax = plt.subplots(figsize=figsize)

    # Per-query lines (thin, transparent)
    qids = df["qid"].unique()
    for qid in qids:
        values = []
        for stage in stages_present:
            sdf = df[(df["stage"] == stage) & (df["qid"] == qid)]
            col = stage_metric.get(stage, "recall_10")
            if not sdf.empty and col in sdf.columns:
                values.append(sdf[col].iloc[0])
            else:
                values.append(None)

        # Plot if we have values for all stages
        if all(v is not None for v in values):
            ax.plot(range(len(stages_present)), values, '-', color='#3498db',
                    alpha=0.05, linewidth=0.5)

    # Mean line
    mean_values = []
    for stage in stages_present:
        sdf = df[df["stage"] == stage]
        col = stage_metric.get(stage, "recall_10")
        if col in sdf.columns:
            mean_values.append(sdf[col].mean())
        else:
            mean_values.append(0)

    ax.plot(range(len(stages_present)), mean_values, 'o-', color='#2c3e50',
            linewidth=2.5, markersize=8, zorder=5)

    for i, (stage, val) in enumerate(zip(stages_present, mean_values)):
        metric_name = stage_metric.get(stage, "?")
        ax.annotate(f'{val:.3f}\n({metric_name})',
                   xy=(i, val), xytext=(0, 12), textcoords='offset points',
                   ha='center', fontsize=8, color='#2c3e50', fontweight='bold')

    ax.set_xticks(range(len(stages_present)))
    ax.set_xticklabels([STAGE_LABELS.get(s, s) for s in stages_present])
    ax.set_ylabel('Recall')
    ax.set_title(f'{run_name}: Recall across pipeline stages')
    ax.set_ylim(-0.05, 1.1)

    if output_path is None:
        output_path = csv_path.replace(".csv", "_recall_flow.pdf")

    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_stage_comparison(
    csv_paths: list,
    output_path: str = "stage_comparison.pdf",
    metric: str = "ndcg",
    k: int = 10,
    figsize: tuple = (7, 4),
):
    """
    Compare metric across stages for multiple datasets side by side.
    """
    all_dfs = []
    for path in csv_paths:
        df = pd.read_csv(path)
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    col = f"{metric}_{k}"

    runs = combined["run"].unique()
    stages_present = [s for s in STAGE_ORDER if s in combined["stage"].values]
    n_runs = len(runs)
    n_stages = len(stages_present)

    fig, ax = plt.subplots(figsize=figsize)

    bar_width = 0.7 / n_stages

    for j, stage in enumerate(stages_present):
        means = []
        for run in runs:
            sdf = combined[(combined["run"] == run) & (combined["stage"] == stage)]
            means.append(sdf[col].mean() if col in sdf.columns and not sdf.empty else 0)

        x = np.arange(n_runs) + j * bar_width - (n_stages - 1) * bar_width / 2
        color = STAGE_COLORS.get(stage, '#888')
        ax.bar(x, means, bar_width * 0.9, color=color, alpha=0.8,
               label=STAGE_LABELS.get(stage, stage), edgecolor='white', linewidth=0.3)

    ax.set_xticks(np.arange(n_runs))
    ax.set_xticklabels(runs, rotation=30, ha='right')
    ax.set_ylabel(f'{metric.upper()}@{k}')
    ax.set_title(f'{metric.upper()}@{k} across datasets and stages')
    ax.legend(loc='upper right', framealpha=0.9, edgecolor='none')
    ax.set_ylim(0, 1.1)

    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_all_stages(csv_path, output_dir=None):
    """Generate all stage progression plots."""
    if output_dir is None:
        output_dir = os.path.dirname(csv_path)

    plot_stage_progression(csv_path, os.path.join(output_dir, "stage_ndcg10.pdf"),
                           metric="ndcg", k=10)
    plot_stage_progression(csv_path, os.path.join(output_dir, "stage_recall100.pdf"),
                           metric="recall", k=100)
    plot_stage_multi_metric(csv_path, os.path.join(output_dir, "stage_multi.pdf"))
    plot_recall_flow(csv_path, os.path.join(output_dir, "stage_recall_flow.pdf"))