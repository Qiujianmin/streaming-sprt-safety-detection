#!/usr/bin/env python3
"""
Generate Pareto Frontier Figure for Camera-Ready Paper.

F1 vs. Token Savings across all four datasets.
Uses Wong (2011) color-blind safe palette.

Author: Qiu Jianmin, Han Jinguang
Date: April 2026
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os

# ============================================================
# Style: color-blind friendly (Wong 2011 palette)
# ============================================================
CB_BLUE    = '#0072B2'  # Blue
CB_ORANGE  = '#E69F00'  # Orange
CB_GREEN   = '#009E73'  # Bluish Green
CB_RED     = '#D55E00'  # Vermillion
CB_PURPLE  = '#CC79A7'  # Reddish Purple
CB_BLACK   = '#000000'  # Black

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.unicode_minus': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'text.usetex': False,
})

# ============================================================
# Data from Table 2 (Main Results)
# Format: (token_savings_pct, F1)
# ============================================================

# -- Streaming methods only (no full-text baselines) --

# Cascade-Guard (stars) - zero-shot
cascade_data = {
    'CivilComments':  (92.5, 0.052),
    'BeaverTails':    (71.1, 0.530),
    'PKU-SafeRLHF':   (93.8, 0.416),
    'Qwen3GuardTest': (96.1, 0.238),
}

# Wald streaming (triangles) - best prior per dataset
wald_data = {
    'CivilComments':  (92.5, 0.356),
    'BeaverTails':    (71.1, 0.780),
    'PKU-SafeRLHF':   (93.8, 0.513),
    'Qwen3GuardTest': (96.1, 1.000),
}

# Confidence threshold (circles) - best tau per dataset
conf_data = {
    'CivilComments':  (61.0, 0.522),
    'BeaverTails':    (82.1, 0.769),
    'PKU-SafeRLHF':   (62.7, 0.866),
    'Qwen3GuardTest': (96.1, 1.000),
}

# Fixed-Window (squares) - only available for PKU and Qwen3
fixed_data = {
    'PKU-SafeRLHF':   (58.3, 0.884),
    'Qwen3GuardTest': (96.1, 1.000),
}

# Full-text baselines (shown as horizontal dashed lines)
ft_roberta_f1 = {
    'CivilComments':  0.637,
    'BeaverTails':    0.816,
    'PKU-SafeRLHF':   0.933,
    'Qwen3GuardTest': 1.000,
}

# Dataset colors
dataset_colors = {
    'CivilComments':  CB_BLUE,
    'BeaverTails':    CB_ORANGE,
    'PKU-SafeRLHF':   CB_GREEN,
    'Qwen3GuardTest': CB_RED,
}

dataset_labels = {
    'CivilComments':  'CivilComments (8%)',
    'BeaverTails':    'BeaverTails (57%)',
    'PKU-SafeRLHF':   'PKU-SafeRLHF (52%)',
    'Qwen3GuardTest': 'Qwen3Guard (100%)',
}

# ============================================================
# Plot
# ============================================================
fig, ax = plt.subplots(figsize=(7, 5))

# Method markers
markers = {
    'cascade': '*',
    'wald': '^',
    'conf': 'o',
    'fixed': 's',
}

marker_size = {
    'cascade': 14,
    'wald': 9,
    'conf': 8,
    'fixed': 8,
}

datasets = ['CivilComments', 'BeaverTails', 'PKU-SafeRLHF', 'Qwen3GuardTest']

# Plot each dataset as a different color, each method as different marker
for ds in datasets:
    color = dataset_colors[ds]

    # Wald
    if ds in wald_data:
        x, y = wald_data[ds]
        ax.scatter(x, y, marker=markers['wald'], s=marker_size['wald']**2,
                   c=color, edgecolors='black', linewidths=0.5, zorder=5)

    # Cascade
    if ds in cascade_data:
        x, y = cascade_data[ds]
        ax.scatter(x, y, marker=markers['cascade'], s=marker_size['cascade']**2,
                   c=color, edgecolors='black', linewidths=0.5, zorder=6)

    # Confidence
    if ds in conf_data:
        x, y = conf_data[ds]
        ax.scatter(x, y, marker=markers['conf'], s=marker_size['conf']**2,
                   c=color, edgecolors='black', linewidths=0.5, zorder=5)

    # Fixed-window
    if ds in fixed_data:
        x, y = fixed_data[ds]
        ax.scatter(x, y, marker=markers['fixed'], s=marker_size['fixed']**2,
                   c=color, edgecolors='black', linewidths=0.5, zorder=5)

# Connect cascade → wald for same dataset (showing F1 gap)
for ds in datasets:
    if ds in cascade_data and ds in wald_data:
        cx, cy = cascade_data[ds]
        wx, wy = wald_data[ds]
        ax.plot([cx, wx], [cy, wy], '--', color=dataset_colors[ds],
                alpha=0.4, linewidth=1.0, zorder=3)

# Annotate key points
annotations = [
    (wald_data['BeaverTails'], 'Wald(π=0.1)\nBeaverTails', (10, 8)),
    (wald_data['PKU-SafeRLHF'], 'Wald(π=0.01)\nPKU', (8, -18)),
    (wald_data['CivilComments'], 'Wald(π=0.01)\nCivil', (-10, 10)),
    (cascade_data['Qwen3GuardTest'], 'Cascade\nQwen3', (-30, -15)),
    (wald_data['Qwen3GuardTest'], 'Wald/Conf\nQwen3', (8, 5)),
    (conf_data['PKU-SafeRLHF'], 'Conf(0.5)\nPKU', (8, 8)),
    (conf_data['BeaverTails'], 'Conf(0.5)\nBeaverTails', (8, -12)),
]
for (x, y), label, offset in annotations:
    ax.annotate(label, (x, y), xytext=offset, textcoords='offset points',
                fontsize=7, ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor='gray', alpha=0.8))

# Legend: method types
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='*', color='w', markerfacecolor='gray',
           markersize=12, markeredgecolor='black', markeredgewidth=0.5,
           label='Cascade-Guard'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='gray',
           markersize=9, markeredgecolor='black', markeredgewidth=0.5,
           label='Wald streaming'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
           markersize=8, markeredgecolor='black', markeredgewidth=0.5,
           label='Confidence(τ)'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
           markersize=8, markeredgecolor='black', markeredgewidth=0.5,
           label='Fixed-Window'),
]
# Dataset colors in legend
for ds in datasets:
    legend_elements.append(
        Line2D([0], [0], marker='s', color='w', markerfacecolor=dataset_colors[ds],
               markersize=8, markeredgecolor='black', markeredgewidth=0.5,
               label=dataset_labels[ds])
    )

ax.legend(handles=legend_elements, loc='lower left', fontsize=7.5,
          ncol=2, framealpha=0.9, edgecolor='gray')

ax.set_xlabel('Token Savings (%)', fontsize=11)
ax.set_ylabel('F1 Score', fontsize=11)
ax.set_xlim(50, 100)
ax.set_ylim(0, 1.08)
ax.grid(True, alpha=0.25)
ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)

plt.tight_layout()

# Save
out_dir = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(out_dir, '..', 'paper', 'figures', 'pareto_frontier.pdf')
png_path = os.path.join(out_dir, 'results', 'supplemental', 'pareto_frontier.png')
os.makedirs(os.path.dirname(pdf_path), exist_ok=True)

fig.savefig(pdf_path, bbox_inches='tight', dpi=300)
fig.savefig(png_path, bbox_inches='tight', dpi=300)
print(f"Saved: {pdf_path}")
print(f"Saved: {png_path}")
plt.close()
print("Done.")
