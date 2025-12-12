#!/usr/bin/env python3
"""
Terminal telomere analysis playground for VGP assemblies.

Goal
--------------
- Explore telomere length vs distance to chromosome end.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ───────────────────────────────────────────────────────────────────────────────
# Directory structure
# ───────────────────────────────────────────────────────────────────────────────
base_dir = "/mnt/d/research/teloscope_article"
data_dir = os.path.join(base_dir, "data/2_processed")
figures_dir = os.path.join(base_dir, "figures/1_draft")
run_label = "v2"  # Label for output filenames

vgp_metadata_file = os.path.join(data_dir, "25.09.25_metadata_telomere_assembly_simplified.csv")
vgp_terminal_telomeres = os.path.join(data_dir, "25.12.05_compiled_TTAGGG_terminal_telomeres.bed")

df = pd.read_csv(
    vgp_terminal_telomeres, sep="\t", header=None,
    names=[
        "chr", "start", "end", "length", "label",
        "fwdCounts", "revCounts", "canCounts", "nonCanCounts",
        "chr_length", "assembly", "clade", "extended_lineage"
    ]
)

# ───────────────────────────────────────────────────────────────────────────────
# Lineage order & palette
# ───────────────────────────────────────────────────────────────────────────────
lineage_order = [
    "Other Deuterostomes",
    "Cyclostomes",
    "Cartilaginous fishes",
    "Ray-finned fishes",
    "Lobe-finned fishes",
    "Amphibians",
    "Lepidosauria",
    "Turtles",
    "Crocodilians",
    "Birds",
    "Mammals",
]
lineage_colors = {
    "Mammals": "#E69F00",
    "Birds": "#00796B",
    "Crocodilians": "#009688",
    "Turtles": "#4DB6AC",
    "Lepidosauria": "#80CBC4",
    "Amphibians": "#984EA3",
    "Lobe-finned fishes": "#A6761D",
    "Ray-finned fishes": "#56B4E9",
    "Cartilaginous fishes": "#0072B2",
    "Cyclostomes": "#CC79A7",
    "Other Deuterostomes": "#999999",
}

# ───────────────────────────────────────────────────────────────────────────────
# Load technology metadata
# ───────────────────────────────────────────────────────────────────────────────
tech_df = pd.read_csv(vgp_metadata_file, dtype=str).fillna("")

# ───────────────────────────────────────────────────────────────────────────────
# Compute distance to end and chromosome positioning
# ───────────────────────────────────────────────────────────────────────────────
# distance2end: minimum of "start" vs ("chr_length" - "end")
df["distance2end"] = np.minimum(df["start"], df["chr_length"] - df["end"])

# distance2end_pct: normalized by chromosome length (as percentage)
df["distance2end_pct"] = (df["distance2end"] / df["chr_length"]) * 100

# chr_position_pct: position along chromosome (0% = start, 100% = end)
# Use midpoint of telomere region
df["chr_position_pct"] = ((df["start"] + df["end"]) / 2 / df["chr_length"]) * 100

# canonical_pct: canonical proportion as percentage (canCounts * 6 / length)
df["canonical_pct"] = (df["canCounts"] * 6 / df["length"]) * 100

# ───────────────────────────────────────────────────────────────────────────────
# Plot 1a: End distance (log) vs Telomere length - Scatter by lineage
# ───────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5.5))
for lin in lineage_order:
    subset = df[df["extended_lineage"] == lin]
    if subset.empty:
        continue
    # Filter out zero/negative distances for log scale
    subset = subset[subset["distance2end"] > 0]
    ax.scatter(
        subset["distance2end"],
        subset["length"] / 1000.0,
        c=lineage_colors[lin],
        label=lin,
        alpha=0.6,
        s=10,
        edgecolors="none"
    )
ax.set_xscale("log")
ax.set_xlabel("Distance to end (bp)")
ax.set_ylabel("Telomere length (Kbp)")
ax.legend(title="Lineage", fontsize=7, title_fontsize=8, markerscale=2)
sns.despine(ax=ax, top=True, right=True)
fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(
        os.path.join(figures_dir, f"Telomere_vs_distance2end_log_{run_label}.{ext}"),
        dpi=600 if ext == "png" else None
    )
plt.close(fig)

# ───────────────────────────────────────────────────────────────────────────────
# Plot 1b: End distance (log) vs Telomere length - Heatmap (all data)
# ───────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5.5))
# Filter out zero/negative distances for log scale
df_pos = df[df["distance2end"] > 0].copy()
h, xedges, yedges, im = ax.hist2d(
    np.log10(df_pos["distance2end"]),
    df_pos["length"] / 1000.0,
    bins=100,
    cmap="viridis",
    cmin=1
)
cb = plt.colorbar(im, ax=ax)
cb.set_label("Count")
ax.set_xlabel("Distance to end (log₁₀ bp)")
ax.set_ylabel("Telomere length (Kbp)")
sns.despine(ax=ax, top=True, right=True)
fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(
        os.path.join(figures_dir, f"Telomere_vs_distance2end_log_heatmap_{run_label}.{ext}"),
        dpi=600 if ext == "png" else None
    )
plt.close(fig)

# ───────────────────────────────────────────────────────────────────────────────
# Plot 2a: End distance (% chr) vs Telomere length - Scatter by lineage
# ───────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5.5))
for lin in lineage_order:
    subset = df[df["extended_lineage"] == lin]
    if subset.empty:
        continue
    ax.scatter(
        subset["distance2end_pct"],
        subset["length"] / 1000.0,
        c=lineage_colors[lin],
        label=lin,
        alpha=0.6,
        s=10,
        edgecolors="none"
    )
ax.set_xlabel("Distance to end (%chr)")
ax.set_ylabel("Telomere length (Kbp)")
ax.legend(title="Lineage", fontsize=7, title_fontsize=8, markerscale=2)
sns.despine(ax=ax, top=True, right=True)
fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(
        os.path.join(figures_dir, f"Telomere_vs_distance2end_pct_{run_label}.{ext}"),
        dpi=600 if ext == "png" else None
    )
plt.close(fig)

# ───────────────────────────────────────────────────────────────────────────────
# Plot 2b: End distance (% chr) vs Telomere length - Heatmap (all data)
# ───────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5.5))
h, xedges, yedges, im = ax.hist2d(
    df["distance2end_pct"],
    df["length"] / 1000.0,
    bins=100,
    cmap="viridis",
    cmin=1
)
cb = plt.colorbar(im, ax=ax)
cb.set_label("Count")
ax.set_xlabel("Distance to end (%chr)")
ax.set_ylabel("Telomere length (Kbp)")
sns.despine(ax=ax, top=True, right=True)
fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(
        os.path.join(figures_dir, f"Telomere_vs_distance2end_pct_heatmap_{run_label}.{ext}"),
        dpi=600 if ext == "png" else None
    )
plt.close(fig)

# ───────────────────────────────────────────────────────────────────────────────
# Plot 3a: Chromosome positioning vs Telomere length - Scatter by lineage
# ───────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5.5))
for lin in lineage_order:
    subset = df[df["extended_lineage"] == lin]
    if subset.empty:
        continue
    ax.scatter(
        subset["chr_position_pct"],
        subset["length"] / 1000.0,
        c=lineage_colors[lin],
        label=lin,
        alpha=0.6,
        s=10,
        edgecolors="none"
    )
ax.set_xlabel("Chromosome positioning (%chr)")
ax.set_ylabel("Telomere length (Kbp)")
ax.legend(title="Lineage", fontsize=7, title_fontsize=8, markerscale=2)
sns.despine(ax=ax, top=True, right=True)
fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(
        os.path.join(figures_dir, f"Telomere_vs_chr_position_{run_label}.{ext}"),
        dpi=600 if ext == "png" else None
    )
plt.close(fig)

# ───────────────────────────────────────────────────────────────────────────────
# Plot 3b: Chromosome positioning vs Telomere length - Heatmap (all data)
# ───────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5.5))
h, xedges, yedges, im = ax.hist2d(
    df["chr_position_pct"],
    df["length"] / 1000.0,
    bins=100,
    cmap="viridis",
    cmin=1
)
cb = plt.colorbar(im, ax=ax)
cb.set_label("Count")
ax.set_xlabel("Chromosome positioning (%chr)")
ax.set_ylabel("Telomere length (Kbp)")
sns.despine(ax=ax, top=True, right=True)
fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(
        os.path.join(figures_dir, f"Telomere_vs_chr_position_heatmap_{run_label}.{ext}"),
        dpi=600 if ext == "png" else None
    )
plt.close(fig)

# ───────────────────────────────────────────────────────────────────────────────
# Plot 4a: End distance (log) vs Canonical proportion - Scatter by lineage
# ───────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5.5))
for lin in lineage_order:
    subset = df[df["extended_lineage"] == lin]
    if subset.empty:
        continue
    # Filter out zero/negative distances for log scale
    subset = subset[subset["distance2end"] > 0]
    ax.scatter(
        subset["distance2end"],
        subset["canonical_pct"],
        c=lineage_colors[lin],
        label=lin,
        alpha=0.6,
        s=10,
        edgecolors="none"
    )
ax.set_xscale("log")
ax.set_xlabel("Distance to end (bp)")
ax.set_ylabel("Canonical proportion (%)")
ax.legend(title="Lineage", fontsize=7, title_fontsize=8, markerscale=2)
sns.despine(ax=ax, top=True, right=True)
fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(
        os.path.join(figures_dir, f"Canonical_vs_distance2end_log_{run_label}.{ext}"),
        dpi=600 if ext == "png" else None
    )
plt.close(fig)

# ───────────────────────────────────────────────────────────────────────────────
# Plot 4b: End distance (log) vs Canonical proportion - Heatmap (all data)
# ───────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5.5))
df_pos = df[df["distance2end"] > 0].copy()
h, xedges, yedges, im = ax.hist2d(
    np.log10(df_pos["distance2end"]),
    df_pos["canonical_pct"],
    bins=100,
    cmap="viridis",
    cmin=1
)
cb = plt.colorbar(im, ax=ax)
cb.set_label("Count")
ax.set_xlabel("Distance to end (log₁₀ bp)")
ax.set_ylabel("Canonical proportion (%)")
sns.despine(ax=ax, top=True, right=True)
fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(
        os.path.join(figures_dir, f"Canonical_vs_distance2end_log_heatmap_{run_label}.{ext}"),
        dpi=600 if ext == "png" else None
    )
plt.close(fig)

# ───────────────────────────────────────────────────────────────────────────────
# Plot 5a: End distance (% chr) vs Canonical proportion - Scatter by lineage
# ───────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5.5))
for lin in lineage_order:
    subset = df[df["extended_lineage"] == lin]
    if subset.empty:
        continue
    ax.scatter(
        subset["distance2end_pct"],
        subset["canonical_pct"],
        c=lineage_colors[lin],
        label=lin,
        alpha=0.6,
        s=10,
        edgecolors="none"
    )
ax.set_xlabel("Distance to end (%chr)")
ax.set_ylabel("Canonical proportion (%)")
ax.legend(title="Lineage", fontsize=7, title_fontsize=8, markerscale=2)
sns.despine(ax=ax, top=True, right=True)
fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(
        os.path.join(figures_dir, f"Canonical_vs_distance2end_pct_{run_label}.{ext}"),
        dpi=600 if ext == "png" else None
    )
plt.close(fig)

# ───────────────────────────────────────────────────────────────────────────────
# Plot 5b: End distance (% chr) vs Canonical proportion - Heatmap (all data)
# ───────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5.5))
h, xedges, yedges, im = ax.hist2d(
    df["distance2end_pct"],
    df["canonical_pct"],
    bins=100,
    cmap="viridis",
    cmin=1
)
cb = plt.colorbar(im, ax=ax)
cb.set_label("Count")
ax.set_xlabel("Distance to end (%chr)")
ax.set_ylabel("Canonical proportion (%)")
sns.despine(ax=ax, top=True, right=True)
fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(
        os.path.join(figures_dir, f"Canonical_vs_distance2end_pct_heatmap_{run_label}.{ext}"),
        dpi=600 if ext == "png" else None
    )
plt.close(fig)

# ───────────────────────────────────────────────────────────────────────────────
# Plot 6a: Chromosome positioning vs Canonical proportion - Scatter by lineage
# ───────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5.5))
for lin in lineage_order:
    subset = df[df["extended_lineage"] == lin]
    if subset.empty:
        continue
    ax.scatter(
        subset["chr_position_pct"],
        subset["canonical_pct"],
        c=lineage_colors[lin],
        label=lin,
        alpha=0.6,
        s=10,
        edgecolors="none"
    )
ax.set_xlabel("Chromosome positioning (%chr)")
ax.set_ylabel("Canonical proportion (%)")
ax.legend(title="Lineage", fontsize=7, title_fontsize=8, markerscale=2)
sns.despine(ax=ax, top=True, right=True)
fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(
        os.path.join(figures_dir, f"Canonical_vs_chr_position_{run_label}.{ext}"),
        dpi=600 if ext == "png" else None
    )
plt.close(fig)

# ───────────────────────────────────────────────────────────────────────────────
# Plot 6b: Chromosome positioning vs Canonical proportion - Heatmap (all data)
# ───────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5.5))
h, xedges, yedges, im = ax.hist2d(
    df["chr_position_pct"],
    df["canonical_pct"],
    bins=100,
    cmap="viridis",
    cmin=1
)
cb = plt.colorbar(im, ax=ax)
cb.set_label("Count")
ax.set_xlabel("Chromosome positioning (%chr)")
ax.set_ylabel("Canonical proportion (%)")
sns.despine(ax=ax, top=True, right=True)
fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(
        os.path.join(figures_dir, f"Canonical_vs_chr_position_heatmap_{run_label}.{ext}"),
        dpi=600 if ext == "png" else None
    )
plt.close(fig)

print(f"[OK] Plots saved to {figures_dir}")
