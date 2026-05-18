"""
Paper-ready plots for CNF query pipeline analysis (300+ topics).

Usage:
    from query_plots import plot_all
    plot_all(curves, output_dir="./plots/")
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

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
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
})

STEP_COLORS = {
    '3way_and': '#c0392b',
    '2way_and': '#2980b9',
    'standalone': '#7f8c8d',
}
STEP_LABELS = {
    '3way_and': '3-Term AND',
    '2way_and': '2-Term AND',
    'standalone': 'Standalone',
}


def _classify_step(query_desc):
    if " AND " not in query_desc:
        return "standalone"
    parts = query_desc.split(" AND ")
    return "3way_and" if len(parts) >= 3 else "2way_and"


def plot_recall_vs_engine(curves, output_path, figsize=(4.5, 3.2), max_engine=50000):
    """
    Recall vs cumulative engine count.
    Median line + IQR shading + 10/90 percentile.
    """
    fig, ax = plt.subplots(figsize=figsize)

    engine_grid = np.linspace(0, max_engine, 500)
    recall_interps = []

    for qid, data in curves.items():
        steps = data["steps"]
        if not steps:
            continue
        engines = [0] + [s["cum_engine"] for s in steps]
        recalls = [0] + [s["cum_recall"] * 100 for s in steps]
        # Extend to max_engine with final recall
        engines.append(max_engine)
        recalls.append(recalls[-1])
        interp = np.interp(engine_grid, engines, recalls)
        recall_interps.append(interp)

    if not recall_interps:
        plt.close(fig)
        return

    arr = np.array(recall_interps)
    median = np.median(arr, axis=0)
    p25 = np.percentile(arr, 25, axis=0)
    p75 = np.percentile(arr, 75, axis=0)
    p10 = np.percentile(arr, 10, axis=0)
    p90 = np.percentile(arr, 90, axis=0)

    ax.fill_between(engine_grid, p10, p90, alpha=0.08, color='#2c3e50', label='P10–P90')
    ax.fill_between(engine_grid, p25, p75, alpha=0.2, color='#2c3e50', label='IQR')
    ax.plot(engine_grid, median, color='#2c3e50', linewidth=1.8, label='Median')

    # Budget markers
    for budget in [5000, 10000, 20000]:
        idx = np.searchsorted(engine_grid, budget)
        if idx < len(median):
            ax.plot(budget, median[idx], 'o', color='#e74c3c', markersize=4, zorder=5)
            ax.annotate(f'{median[idx]:.0f}%',
                       xy=(budget, median[idx]),
                       xytext=(5, -12), textcoords='offset points',
                       fontsize=7, color='#e74c3c')

    ax.set_xlabel('Cumulative documents retrieved')
    ax.set_ylabel('Recall (%)')
    ax.set_xlim(0, max_engine)
    ax.set_ylim(0, 100)
    ax.legend(loc='lower right', framealpha=0.9, edgecolor='none')

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_efficiency_boxplot(curves, output_path, figsize=(3.5, 3)):
    """
    Efficiency by query type — box plots.
    """
    type_data = defaultdict(list)
    for qid, data in curves.items():
        for step in data["steps"]:
            qtype = _classify_step(step["query"])
            if step["engine_count"] > 0:
                eff = step["new_docs"] / step["engine_count"]
                type_data[qtype].append(eff)

    fig, ax = plt.subplots(figsize=figsize)

    types = ["3way_and", "2way_and", "standalone"]
    plot_data = [type_data.get(t, [0]) for t in types]
    labels = [STEP_LABELS[t] for t in types]
    colors = [STEP_COLORS[t] for t in types]

    bp = ax.boxplot(plot_data, labels=labels, patch_artist=True,
                     showfliers=False, widths=0.5,
                     medianprops=dict(color='white', linewidth=1.5))

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    # Add median annotations
    for i, (t, d) in enumerate(zip(types, plot_data)):
        med = np.median(d)
        ax.text(i + 1, med, f' {med:.4f}', va='center', fontsize=7, color='#2c3e50')

    ax.set_ylabel('Efficiency\n(relevant docs / engine count)')
    ax.set_yscale('log')

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_recall_waterfall(curves, output_path, figsize=(4, 3)):
    """
    Waterfall chart: marginal recall contribution by query type.
    Averaged across all topics.
    """
    type_order = ["3way_and", "2way_and", "standalone"]

    # Per-topic contributions
    contributions = {t: [] for t in type_order}

    for qid, data in curves.items():
        n_docs = data["n_docs"]
        type_docs = defaultdict(int)
        for step in data["steps"]:
            qtype = _classify_step(step["query"])
            type_docs[qtype] += step["new_docs"]
        for t in type_order:
            contributions[t].append(type_docs[t] / n_docs * 100 if n_docs else 0)

    fig, ax = plt.subplots(figsize=figsize)

    means = [np.mean(contributions[t]) for t in type_order]
    stds = [np.std(contributions[t]) for t in type_order]
    labels = [STEP_LABELS[t] for t in type_order]
    colors = [STEP_COLORS[t] for t in type_order]

    # Waterfall: cumulative bottom
    bottoms = [0]
    for m in means[:-1]:
        bottoms.append(bottoms[-1] + m)

    bars = ax.bar(labels, means, bottom=bottoms, color=colors,
                   edgecolor='white', linewidth=0.5, width=0.55)

    # Annotations
    for bar, mean, bottom in zip(bars, means, bottoms):
        if mean > 2:
            ax.text(bar.get_x() + bar.get_width() / 2, bottom + mean / 2,
                    f'{mean:.1f}%', ha='center', va='center',
                    fontsize=8, color='white', fontweight='bold')

    # Total line
    total = sum(means)
    ax.axhline(total, color='#2c3e50', linestyle='--', linewidth=0.7, alpha=0.5)
    ax.text(len(labels) - 0.5, total + 1, f'Total: {total:.1f}%',
            fontsize=7, color='#2c3e50', ha='right')

    ax.set_ylabel('Recall contribution (%)')
    ax.set_ylim(0, min(total * 1.15, 105))

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_cost_waterfall(curves, output_path, figsize=(4, 3)):
    """
    Waterfall chart: engine count contribution by query type.
    """
    type_order = ["3way_and", "2way_and", "standalone"]

    contributions = {t: [] for t in type_order}

    for qid, data in curves.items():
        type_eng = defaultdict(int)
        for step in data["steps"]:
            qtype = _classify_step(step["query"])
            type_eng[qtype] += step["engine_count"]
        for t in type_order:
            contributions[t].append(type_eng[t] / 1000)

    fig, ax = plt.subplots(figsize=figsize)

    means = [np.mean(contributions[t]) for t in type_order]
    labels = [STEP_LABELS[t] for t in type_order]
    colors = [STEP_COLORS[t] for t in type_order]

    bottoms = [0]
    for m in means[:-1]:
        bottoms.append(bottoms[-1] + m)

    bars = ax.bar(labels, means, bottom=bottoms, color=colors,
                   edgecolor='white', linewidth=0.5, width=0.55)

    for bar, mean, bottom in zip(bars, means, bottoms):
        if mean > 3:
            ax.text(bar.get_x() + bar.get_width() / 2, bottom + mean / 2,
                    f'{mean:.0f}k', ha='center', va='center',
                    fontsize=8, color='white', fontweight='bold')

    total = sum(means)
    ax.axhline(total, color='#2c3e50', linestyle='--', linewidth=0.7, alpha=0.5)
    ax.text(len(labels) - 0.5, total + total * 0.03, f'Total: {total:.0f}k',
            fontsize=7, color='#2c3e50', ha='right')

    ax.set_ylabel('Documents retrieved (thousands)')

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_recall_vs_cost_by_type(curves, output_path, figsize=(4.5, 3.2)):
    """
    Scatter: per-topic recall vs engine count, one point per type.
    Shows the tradeoff each query type offers.
    """
    fig, ax = plt.subplots(figsize=figsize)

    type_order = ["3way_and", "2way_and", "standalone"]

    for qtype in type_order:
        recalls = []
        engines = []
        for qid, data in curves.items():
            n_docs = data["n_docs"]
            type_docs = 0
            type_eng = 0
            for step in data["steps"]:
                if _classify_step(step["query"]) == qtype:
                    type_docs += step["new_docs"]
                    type_eng += step["engine_count"]
            if type_eng > 0:
                recalls.append(type_docs / n_docs * 100 if n_docs else 0)
                engines.append(type_eng / 1000)

        if recalls:
            ax.scatter(engines, recalls, s=12, alpha=0.4,
                      color=STEP_COLORS[qtype], label=STEP_LABELS[qtype],
                      edgecolors='none')

    ax.set_xlabel('Documents retrieved (thousands)')
    ax.set_ylabel('Recall contribution (%)')
    ax.legend(loc='upper right', framealpha=0.9, edgecolor='none', markerscale=2)
    ax.set_xscale('log')

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_proximity_curve(prox_data, output_path, figsize=(4, 3)):
    """
    Recall vs proximity threshold from span distribution.
    """
    if not prox_data:
        print("  Skipping proximity plot (no data)")
        return

    spans = [d["span"] for d in prox_data if d.get("span") is not None]
    if not spans:
        return

    prox_range = np.arange(0, min(max(spans), 500) + 5, 2)
    recalls = [np.mean([s <= p for s in spans]) * 100 for p in prox_range]

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(prox_range, recalls, color='#2c3e50', linewidth=1.5)
    ax.fill_between(prox_range, 0, recalls, alpha=0.08, color='#2c3e50')

    for target, color, ls in [(90, '#c0392b', '--'), (95, '#e67e22', ':')]:
        idx = next((i for i, r in enumerate(recalls) if r >= target), None)
        if idx is not None:
            pval = prox_range[idx]
            ax.axhline(target, color=color, linestyle=ls, linewidth=0.6, alpha=0.5)
            ax.plot(pval, target, 'o', color=color, markersize=4, zorder=5)
            ax.annotate(f'{target}% @ {pval}',
                       xy=(pval, target),
                       xytext=(15, -3), textcoords='offset points',
                       fontsize=7, color=color)

    ax.set_xlabel('Proximity threshold (tokens)')
    ax.set_ylabel('Documents reachable (%)')
    ax.set_xlim(0, min(max(spans), 500))
    ax.set_ylim(0, 102)

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_combined_waterfall(curves, output_path, figsize=(6, 3.2)):
    """
    Side-by-side: recall contribution (left) vs cost (right) by query type.
    Legend outside the plot. All segments labeled.
    """
    type_order = ["3way_and", "2way_and", "standalone"]

    recall_contribs = {t: [] for t in type_order}
    engine_contribs = {t: [] for t in type_order}

    for qid, data in curves.items():
        n_docs = data["n_docs"]
        type_docs = defaultdict(int)
        type_eng = defaultdict(int)
        for step in data["steps"]:
            qtype = _classify_step(step["query"])
            type_docs[qtype] += step["new_docs"]
            type_eng[qtype] += step["engine_count"]
        for t in type_order:
            recall_contribs[t].append(type_docs[t] / n_docs * 100 if n_docs else 0)
            engine_contribs[t].append(type_eng[t] / 1000)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    labels = [STEP_LABELS[t] for t in type_order]
    colors = [STEP_COLORS[t] for t in type_order]

    # Left: recall
    r_means = [np.mean(recall_contribs[t]) for t in type_order]
    r_bottoms = [0]
    for m in r_means[:-1]:
        r_bottoms.append(r_bottoms[-1] + m)

    for i, (mean, bottom, color, label) in enumerate(zip(r_means, r_bottoms, colors, labels)):
        ax1.bar(['Recall'], [mean], bottom=bottom, color=color,
                edgecolor='white', linewidth=0.5, width=0.45, label=label)
        # Always label, even small segments
        if mean > 1:
            ax1.text(0, bottom + mean / 2, f'{mean:.1f}%',
                    ha='center', va='center', fontsize=8, color='white', fontweight='bold')

    ax1.set_ylabel('Recall (%)')
    ax1.set_ylim(0, sum(r_means) * 1.15)

    # Right: engine count — always show labels
    e_means = [np.mean(engine_contribs[t]) for t in type_order]
    e_bottoms = [0]
    for m in e_means[:-1]:
        e_bottoms.append(e_bottoms[-1] + m)

    for i, (mean, bottom, color) in enumerate(zip(e_means, e_bottoms, colors)):
        ax2.bar(['Cost'], [mean], bottom=bottom, color=color,
                edgecolor='white', linewidth=0.5, width=0.45)
        # Label all segments — use offset for tiny ones
        if mean > sum(e_means) * 0.05:
            ax2.text(0, bottom + mean / 2, f'{mean:.0f}k',
                    ha='center', va='center', fontsize=8, color='white', fontweight='bold')
        elif mean > 0:
            ax2.annotate(f'{mean:.0f}k', xy=(0, bottom + mean / 2),
                        xytext=(0.35, bottom + mean / 2),
                        fontsize=7, color=color, fontweight='bold',
                        arrowprops=dict(arrowstyle='-', color=color, lw=0.5))

    ax2.set_ylabel('Documents retrieved (thousands)')
    ax2.set_ylim(0, sum(e_means) * 1.15)

    # Legend outside the plot area
    fig.legend(labels, loc='upper center', bbox_to_anchor=(0.5, 1.08),
               ncol=3, frameon=False, fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_engine_count_distribution(curves, output_path, figsize=(5.5, 3.5)):
    """
    Per-topic engine count distribution by query type.
    Shows KDE density curves on log scale.
    """
    type_order = ["3way_and", "2way_and", "standalone"]
    type_engines = {t: [] for t in type_order}

    for qid, data in curves.items():
        per_type = defaultdict(int)
        for step in data["steps"]:
            qtype = _classify_step(step["query"])
            per_type[qtype] += step["engine_count"]
        for t in type_order:
            if per_type.get(t, 0) > 0:
                type_engines[t].append(per_type[t])

    fig, ax = plt.subplots(figsize=figsize)

    for t in type_order:
        vals = np.array(type_engines[t])
        if len(vals) < 5:
            continue

        # KDE on log-transformed values for better density estimation on skewed data
        log_vals = np.log10(vals)
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(log_vals, bw_method=0.3)

        # Plot range: from smallest to largest across all types
        x_range = np.linspace(log_vals.min() - 0.5, log_vals.max() + 0.5, 300)
        density = kde(x_range)

        # Normalize density so each curve peaks at 1 (makes comparison fair across types)
        density = density / density.max()

        ax.fill_between(10**x_range, density, alpha=0.2, color=STEP_COLORS[t])
        ax.plot(10**x_range, density, color=STEP_COLORS[t], linewidth=1.8,
                label=f"{STEP_LABELS[t]} (n={len(vals)}, median={np.median(vals):,.0f})")

        # Mark median
        median_val = np.median(vals)
        ax.axvline(median_val, color=STEP_COLORS[t], linestyle='--',
                   linewidth=0.8, alpha=0.6)

    ax.set_xlabel('Documents retrieved per topic')
    ax.set_ylabel('Density (normalized)')
    ax.set_xscale('log')
    ax.set_yticks([])

    # Add reference lines for budget thresholds
    for budget, label in [(10000, '10k'), (20000, '20k'), (50000, '50k')]:
        ax.axvline(budget, color='#bdc3c7', linestyle=':', linewidth=0.6, alpha=0.7)
        ax.text(budget, ax.get_ylim()[1] * 0.95, label, fontsize=6,
                color='#7f8c8d', ha='center', va='top')

    # Legend outside above plot
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.35),
              ncol=1, frameon=False, fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.75])
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_all(curves, output_dir="./plots/", prox_data=None):
    """Generate all paper-ready plots."""
    os.makedirs(output_dir, exist_ok=True)

    print("Generating plots...")
    plot_recall_vs_engine(curves, os.path.join(output_dir, "recall_vs_engine.pdf"))
    plot_efficiency_boxplot(curves, os.path.join(output_dir, "efficiency_boxplot.pdf"))
    plot_recall_waterfall(curves, os.path.join(output_dir, "recall_waterfall.pdf"))
    plot_cost_waterfall(curves, os.path.join(output_dir, "cost_waterfall.pdf"))
    plot_combined_waterfall(curves, os.path.join(output_dir, "combined_waterfall.pdf"))
    plot_recall_vs_cost_by_type(curves, os.path.join(output_dir, "recall_vs_cost_scatter.pdf"))
    plot_engine_count_distribution(curves, os.path.join(output_dir, "engine_count_distribution.pdf"))

    if prox_data:
        plot_proximity_curve(prox_data, os.path.join(output_dir, "proximity_curve.pdf"))

    print(f"All plots saved to {output_dir}")