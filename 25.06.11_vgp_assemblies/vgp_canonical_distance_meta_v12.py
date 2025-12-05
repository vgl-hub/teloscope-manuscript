#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Meta stacked horizontal bar: TTAGGG-TTAGGG distance composition
- Terminal (top) and Interstitial (bottom)
- Categories: d0, d1to100, d101to1000, d1000+
- Legend uses Nature-style gradient (viridis, light end) so black labels are readable
- Outputs: PNG (600 dpi) and PDF
"""

import os, matplotlib
import matplotlib.pyplot as plt

# -------------------- Paths --------------------
WD = "./25.10.28_plots/canonical_distance"
os.makedirs(WD, exist_ok=True)
META_TXT = f"{WD}/canonical_gap_meta.txt"

# -------------------- Helpers --------------------
def read_meta(meta_path):
    meta = {}
    with open(meta_path, "r", encoding="utf-8") as f:
        for raw in f:
            if not raw.strip() or raw.lstrip().startswith("#"):
                continue
            line = raw.lstrip("\ufeff").rstrip("\n\r")
            # Prefer tab; fallback to any whitespace if no tab present
            if "\t" in line:
                k, v = line.split("\t", 1)
            else:
                parts = line.split()
                if len(parts) < 2:
                    continue
                k, v = parts[0], parts[1]
            k = k.strip()
            v = v.strip()
            # Cast to int/float when possible
            try:
                meta[k] = int(v)
            except ValueError:
                try:
                    meta[k] = float(v)
                except ValueError:
                    meta[k] = v
    return meta

def fmt_millions(n):
    return f"{n/1e6:.1f}M"

# -------------------- Load --------------------
meta = read_meta(META_TXT)

terminal_counts = [
    meta["terminal_d0"],
    meta["terminal_d1to100"],
    meta["terminal_d101to1000"],
    meta["terminal_d1000+"],
]
interstitial_counts = [
    meta["interstitial_d0"],
    meta["interstitial_d1to100"],
    meta["interstitial_d101to1000"],
    meta["interstitial_d1000+"],
]

t_total = sum(terminal_counts)
i_total = sum(interstitial_counts)

t_pct = [c / t_total * 100 for c in terminal_counts]
i_pct = [c / i_total * 100 for c in interstitial_counts]

# -------------------- Category labels & palette --------------------
cat_labels = [
    "0 bp",
    "1-100 bp",
    "0.1-1 kbp",
    ">1kbp",
]

# Nature-recommended, colorblind-safe gradient: use the lighter end of 'viridis'
cmap = matplotlib.cm.get_cmap("viridis")
stops = [0.65, 0.75, 0.85, 0.95]
cat_colors = [(*cmap(s)[:3], 0.85) for s in stops]  # rgba with alpha

# -------------------- Plot --------------------
fig, ax = plt.subplots(figsize=(8, 2.4), dpi=150)

y_positions = [0.6, 0.2]  # Terminal top, Interstitial bottom
bar_h = 0.24

def stacked_barh(y, pcts, colors):
    left = 0.0
    for pct, color in zip(pcts, colors):
        ax.barh(y, pct, left=left, height=bar_h, color=color, edgecolor="white", linewidth=0.6)
        if pct >= 5:  # your threshold
            ax.text(left + pct/2, y, f"{pct:.0f}%", va="center", ha="center",
                    fontsize=9, color="#222")
        left += pct

stacked_barh(y_positions[0], t_pct, cat_colors)
stacked_barh(y_positions[1], i_pct, cat_colors)

# Y labels with line break and compact totals
ax.set_yticks(y_positions)
ax.set_yticklabels([
    f"Terminal\n(n={fmt_millions(t_total)})",
    f"Interstitial\n(n={fmt_millions(i_total)})"
])

ax.set_xlim(0, 100)
ax.set_xlabel("Composition (%)")
ax.set_xticks([0, 25, 50, 75, 100])

legend_handles = [plt.Rectangle((0,0),1,1, color=c) for c in cat_colors]
ax.legend(
    legend_handles, cat_labels,
    title="Canonical spacing",
    loc="center left", bbox_to_anchor=(1.02, 0.5),
    frameon=False
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(False)

fig.tight_layout()

fig.savefig(f"{WD}/meta_stacked_barh_v12.png", dpi=600)
fig.savefig(f"{WD}/meta_stacked_barh_v12.pdf")
fig.savefig(f"{WD}/meta_stacked_barh_v12.svg")
