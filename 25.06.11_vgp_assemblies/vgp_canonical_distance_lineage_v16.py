#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vgp_canonical_distance_lineage_v16.py

Creates the following (Terminal & Interstitial separately, plus BOTH):
  • lineage_v16_counts_{terminal/interstitial}.{png,pdf}     (4x3 multipanel histogram of counts; bins 1–100)
  • lineage_v16_ecdf_{terminal/interstitial}.{png,pdf}       (4x3 multipanel eCDF of counts; bins 0–100; xticks 0,25,50,75,100)
  • lineage_v16_frac_ecdf_{terminal/interstitial}.{png,pdf}  (single-panel eCDF of cumulative fraction; bins 0–100)
  • lineage_v16_counts_both.{png,pdf}                        (4x3 multipanel histogram of FRACTIONS, T vs I side-by-side; bins 1–100; single legend in empty panel)
  • lineage_v16_ecdf_both.{png,pdf}                          (4x3 multipanel eCDF, T solid / I dotted; bins 0–100; single legend in empty panel; xticks 0,25,50,75,100)
  • lineage_v16_frac_ecdf_both.{png,pdf}                     (single-panel eCDF, same color: T solid / I densely dotted; bins 0–100; lineage legend inside bottom-right)
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless: avoid Qt/Wayland
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ───────────────────────────────────────────────────────────────────────────────
# Paths
# ───────────────────────────────────────────────────────────────────────────────
WD = "./25.10.28_plots/canonical_distance/"
LINEAGE_TSV = f"{WD}/canonical_gap_by_lineage.tsv"

# ───────────────────────────────────────────────────────────────────────────────
# Palette
# ───────────────────────────────────────────────────────────────────────────────
LINEAGE_PALETTE = {
  "Mammals"             : "#E69F00",
  "Birds"               : "#00796B",
  "Ray-finned Fishes"   : "#56B4E9",
  "Lepidosauria"        : "#80CBC4",
  "Amphibians"          : "#984EA3",
  "Turtles"             : "#4DB6AC",
  "Crocodiles"          : "#009688",
  "Cartilaginous Fishes": "#0072B2",
  "Lobe-finned Fishes"  : "#A6761D",
  "Cyclostomes"         : "#CC79A7",
}

# Map dataset names → palette keys
NAME_MAP = {
    "Crocodilians": "Crocodiles",
    "Cartilaginous fishes": "Cartilaginous Fishes",
    "Ray-finned fishes": "Ray-finned Fishes",
    "Lobe-finned fishes": "Lobe-finned Fishes",
}

FALLBACK_COLOR = "#888888"

# Order panels by palette order + known extra category
ORDERED_DATASET_LINEAGES = [
    "Mammals",
    "Birds",
    "Ray-finned fishes",
    "Lepidosauria",
    "Amphibians",
    "Turtles",
    "Crocodilians",
    "Cartilaginous fishes",
    "Lobe-finned fishes",
    "Cyclostomes",
    "Other Deuterostomes",
]

# ───────────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────────
def _canon_name(ds_name: str) -> str:
    return NAME_MAP.get(ds_name, ds_name)

def _color_for(ds_name: str) -> str:
    key = _canon_name(ds_name)
    return LINEAGE_PALETTE.get(key, FALLBACK_COLOR)

def _lighten(hex_color: str, factor: float = 0.5) -> tuple:
    """Mix color with white. factor=1.0 → original; factor=0.0 → white."""
    r, g, b = mcolors.to_rgb(hex_color)
    return (1 - (1 - r) * factor, 1 - (1 - g) * factor, 1 - (1 - b) * factor)

def _get_bin_columns(df: pd.DataFrame, start: int, end: int) -> list:
    cols = [str(i) for i in range(start, end + 1)]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing distance bin columns: {missing[:5]}{' ...' if len(missing)>5 else ''}")
    return cols

def _row_for(df: pd.DataFrame, cls: str, lineage: str) -> pd.Series:
    sel = df[(df["Class"] == cls) & (df["Extended_lineage"] == lineage)]
    if sel.empty:
        raise ValueError(f"No row found for Class='{cls}', Extended_lineage='{lineage}'")
    if len(sel) > 1:
        sel = sel.iloc[[0]]
    return sel.squeeze()

def _counts_array(row: pd.Series, bin_cols: list) -> np.ndarray:
    return row[bin_cols].to_numpy(dtype=float)

def _cum(arr: np.ndarray) -> np.ndarray:
    return np.cumsum(arr)

def _frac(arr: np.ndarray) -> np.ndarray:
    tot = arr.sum()
    if tot <= 0:
        return np.zeros_like(arr, dtype=float)
    return arr / tot

def _ecdf(arr: np.ndarray) -> np.ndarray:
    c = _cum(arr)
    tot = c[-1] if len(c) else 0.0
    if tot <= 0:
        return np.zeros_like(arr, dtype=float)
    return c / tot

def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

# ───────────────────────────────────────────────────────────────────────────────
# Plotters – multipanel (4x3)
# ───────────────────────────────────────────────────────────────────────────────
def multipanel_hist_counts(df: pd.DataFrame, cls: str, start: int, end: int, x_scale: str, y_scale: str, out_prefix: str):
    bin_cols = _get_bin_columns(df, start, end)
    x = np.arange(start, end + 1, dtype=float)

    n_panels = 12
    fig, axes = plt.subplots(3, 4, figsize=(12, 9), sharex=True, sharey=False)
    axes = axes.flatten()

    for i, lineage in enumerate(ORDERED_DATASET_LINEAGES):
        if i >= n_panels:
            break
        ax = axes[i]
        try:
            row = _row_for(df, cls, lineage)
        except ValueError:
            ax.axis("off")
            continue
        y = _counts_array(row, bin_cols)
        ax.bar(x, y, width=1.0, color=_color_for(lineage), edgecolor="none")
        ax.set_title(_canon_name(lineage), fontsize=10, pad=3)
        ax.set_yscale(y_scale)
        ax.set_xscale(x_scale)
        ax.set_xlim(start, end)
        ax.grid(False)
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        ax.margins(x=0.0)

    for j in range(len(ORDERED_DATASET_LINEAGES), n_panels):
        axes[j].axis("off")

    fig.supxlabel("Canonical spacing (bp)", y=0.04)
    fig.supylabel("Frequency", x=0.06)
    fig.tight_layout(rect=[0.05, 0.05, 0.98, 0.96])

    _ensure_dir(out_prefix)
    fig.savefig(f"{out_prefix}_counts_{cls.lower()}.png", dpi=600)
    fig.savefig(f"{out_prefix}_counts_{cls.lower()}.pdf")
    plt.close(fig)

def multipanel_ecdf(df: pd.DataFrame, cls: str, start: int, end: int, x_scale: str, out_prefix: str):
    bin_cols = _get_bin_columns(df, start, end)
    x = np.arange(start, end + 1, dtype=float)

    n_panels = 12
    fig, axes = plt.subplots(3, 4, figsize=(12, 9), sharex=True, sharey=False)
    axes = axes.flatten()

    for i, lineage in enumerate(ORDERED_DATASET_LINEAGES):
        if i >= n_panels:
            break
        ax = axes[i]
        try:
            row = _row_for(df, cls, lineage)
        except ValueError:
            ax.axis("off")
            continue
        y = _counts_array(row, bin_cols)
        yecdf = _ecdf(y)
        ax.step(x, yecdf, where="post", linewidth=1.3, color=_color_for(lineage))
        ax.set_title(_canon_name(lineage), fontsize=10, pad=3)
        ax.set_xscale(x_scale)
        ax.set_xlim(start, end)
        ax.set_xticks([0, 25, 50, 75, 100])  # requested ticks
        ax.grid(False)
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        ax.set_ylim(0, 1.0)
        ax.margins(x=0.0)

    for j in range(len(ORDERED_DATASET_LINEAGES), n_panels):
        axes[j].axis("off")

    fig.supxlabel("Canonical spacing (bp)", y=0.04)
    fig.supylabel("Cumulative fraction", x=0.06)
    fig.tight_layout(rect=[0.05, 0.05, 0.98, 0.96])

    _ensure_dir(out_prefix)
    fig.savefig(f"{out_prefix}_ecdf_{cls.lower()}.png", dpi=600)
    fig.savefig(f"{out_prefix}_ecdf_{cls.lower()}.pdf")
    plt.close(fig)

# BOTH: multipanel histogram with FRACTIONS (T vs I side-by-side; I lighter) + single legend in empty bottom-right panel
def multipanel_hist_fractions_both(df: pd.DataFrame, start: int, end: int, x_scale: str, out_prefix: str):
    from matplotlib.patches import Patch

    bin_cols = _get_bin_columns(df, start, end)
    x = np.arange(start, end + 1, dtype=float)

    n_panels = 12
    fig, axes = plt.subplots(3, 4, figsize=(12, 9), sharex=True, sharey=False)
    axes = axes.flatten()

    width = 0.45
    for i, lineage in enumerate(ORDERED_DATASET_LINEAGES):
        if i >= n_panels:
            break
        ax = axes[i]
        try:
            row_t = _row_for(df, "Terminal", lineage)
            row_i = _row_for(df, "Interstitial", lineage)
        except ValueError:
            ax.axis("off")
            continue
        yt = _counts_array(row_t, bin_cols)
        yi = _counts_array(row_i, bin_cols)
        ft = _frac(yt)
        fi = _frac(yi)

        base = _color_for(lineage)
        light = _lighten(base, factor=0.45)  # a bit lighter than before

        ax.bar(x - 0.225, ft, width=width, color=base,  edgecolor="none")
        ax.bar(x + 0.225, fi, width=width, color=light, edgecolor="none")

        ax.set_title(_canon_name(lineage), fontsize=10, pad=3)
        ax.set_xscale(x_scale)
        ax.set_xlim(start, end)
        ax.grid(False)
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        ax.margins(x=0.0)

    # Use the empty bottom-right panel for a single legend
    if len(axes) >= 12:
        legend_ax = axes[11]
        legend_ax.axis("off")
        handles = [
            Patch(facecolor="#222222", edgecolor="none", label="Terminal"),
            Patch(facecolor="#BDBDBD", edgecolor="none", label="Interstitial"),
        ]
        legend_ax.legend(handles=handles, loc="center", frameon=False, ncol=1, fontsize=9)

    for j in range(len(ORDERED_DATASET_LINEAGES), n_panels):
        if j != 11:  # keep 11 for legend
            axes[j].axis("off")

    fig.supxlabel("Canonical spacing (bp)", y=0.04)
    fig.supylabel("Fraction", x=0.06)
    fig.tight_layout(rect=[0.05, 0.05, 0.98, 0.96])

    _ensure_dir(out_prefix)
    fig.savefig(f"{out_prefix}_counts_both.png", dpi=600)
    fig.savefig(f"{out_prefix}_counts_both.pdf")
    plt.close(fig)

# BOTH: multipanel eCDF (T solid / I dotted) + single legend in empty bottom-right panel
def multipanel_ecdf_both(df: pd.DataFrame, start: int, end: int, x_scale: str, out_prefix: str):
    from matplotlib.lines import Line2D

    bin_cols = _get_bin_columns(df, start, end)
    x = np.arange(start, end + 1, dtype=float)

    n_panels = 12
    fig, axes = plt.subplots(3, 4, figsize=(12, 9), sharex=True, sharey=False)
    axes = axes.flatten()

    for i, lineage in enumerate(ORDERED_DATASET_LINEAGES):
        if i >= n_panels:
            break
        ax = axes[i]
        try:
            row_t = _row_for(df, "Terminal", lineage)
            row_i = _row_for(df, "Interstitial", lineage)
        except ValueError:
            ax.axis("off")
            continue
        yt = _counts_array(row_t, bin_cols)
        yi = _counts_array(row_i, bin_cols)
        ect = _ecdf(yt)
        eci = _ecdf(yi)

        base = _color_for(lineage)
        ax.step(x, ect, where="post", linewidth=1.3, color=base, linestyle="-")
        ax.step(x, eci, where="post", linewidth=1.3, color=base, linestyle=":")

        ax.set_title(_canon_name(lineage), fontsize=10, pad=3)
        ax.set_xscale(x_scale)
        ax.set_xlim(start, end)
        ax.set_xticks([0, 25, 50, 75, 100])  # requested ticks
        ax.grid(False)
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        ax.set_ylim(0, 1.0)
        ax.margins(x=0.0)

    # Single legend in empty bottom-right panel
    if len(axes) >= 12:
        legend_ax = axes[11]
        legend_ax.axis("off")
        handles = [
            Line2D([0], [0], color="#333333", linestyle="-", lw=1.3, label="Terminal"),
            Line2D([0], [0], color="#333333", linestyle=":", lw=1.3, label="Interstitial"),
        ]
        legend_ax.legend(handles=handles, loc="center", frameon=False, ncol=1, fontsize=9)

    for j in range(len(ORDERED_DATASET_LINEAGES), n_panels):
        if j != 11:
            axes[j].axis("off")

    fig.supxlabel("Canonical spacing (bp)", y=0.04)
    fig.supylabel("Cumulative fraction", x=0.06)
    fig.tight_layout(rect=[0.05, 0.05, 0.98, 0.96])

    _ensure_dir(out_prefix)
    fig.savefig(f"{out_prefix}_ecdf_both.png", dpi=600)
    fig.savefig(f"{out_prefix}_ecdf_both.pdf")
    plt.close(fig)

# BOTH: single panel fraction eCDF (same color: T solid / I densely dotted) + lineage legend inside bottom-right
def singlepanel_fraction_ecdf_both(df: pd.DataFrame, start: int, end: int, x_scale: str, out_prefix: str):
    from matplotlib.lines import Line2D

    bin_cols = _get_bin_columns(df, start, end)
    x = np.arange(start, end + 1, dtype=float)

    fig, ax = plt.subplots(figsize=(5, 5))
    for lineage in ORDERED_DATASET_LINEAGES:
        try:
            row_t = _row_for(df, "Terminal", lineage)
            row_i = _row_for(df, "Interstitial", lineage)
        except ValueError:
            continue
        yt = _counts_array(row_t, bin_cols)
        yi = _counts_array(row_i, bin_cols)
        ect = _ecdf(yt)
        eci = _ecdf(yi)

        base = _color_for(lineage)
        # Terminal: solid; Interstitial: densely dotted (custom dash)
        ax.step(x, ect, where="post", linewidth=1.4, color=base, linestyle="-")
        ax.step(x, eci, where="post", linewidth=1.4, color=base, linestyle=(0, (1, 1)))

    ax.set_xscale(x_scale)
    ax.set_xlim(start, end)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Canonical spacing (bp)")
    ax.set_ylabel("Cumulative fraction")
    ax.grid(False)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.margins(x=0.0)

    # Single legend: lineage names (inside bottom-right)
    lineage_handles = [Line2D([0], [0], color=_color_for(l), lw=1.6, label=_canon_name(l))
                       for l in ORDERED_DATASET_LINEAGES]
    ax.legend(handles=lineage_handles, title="Lineage",
              loc="lower right", frameon=False, fontsize=8)

    fig.tight_layout(rect=[0.05, 0.05, 0.98, 0.95])

    _ensure_dir(out_prefix)
    fig.savefig(f"{out_prefix}_frac_ecdf_both.png", dpi=600)
    fig.savefig(f"{out_prefix}_frac_ecdf_both.pdf")
    plt.close(fig)

# Single panel fraction eCDF (per-class) — bins 0–100
def singlepanel_fraction_ecdf(df: pd.DataFrame, cls: str, start: int, end: int, x_scale: str, out_prefix: str):
    bin_cols = _get_bin_columns(df, start, end)
    x = np.arange(start, end + 1, dtype=float)

    fig, ax = plt.subplots(figsize=(5, 5))
    for lineage in ORDERED_DATASET_LINEAGES:
        try:
            row = _row_for(df, cls, lineage)
        except ValueError:
            continue
        y = _counts_array(row, bin_cols)
        yecdf = _ecdf(y)
        ax.step(x, yecdf, where="post", label=_canon_name(lineage), linewidth=1.4, color=_color_for(lineage))

    ax.set_xscale(x_scale)
    ax.set_xlim(start, end)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Canonical spacing (bp)")
    ax.set_ylabel("Cumulative fraction")
    ax.grid(False)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.margins(x=0.0)

    ax.legend(title="Lineage", loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout(rect=[0.05, 0.05, 0.98, 0.95])

    _ensure_dir(out_prefix)
    fig.savefig(f"{out_prefix}_frac_ecdf_{cls.lower()}.png", dpi=600)
    fig.savefig(f"{out_prefix}_frac_ecdf_{cls.lower()}.pdf")
    plt.close(fig)

# ───────────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Lineage-level distance plots for Terminal/Interstitial.")
    ap.add_argument("--wd", default=WD, help="Working directory where outputs are written and input TSV is located.")
    ap.add_argument("--lineage-tsv", default=LINEAGE_TSV, help="Path to canonical_gap_by_lineage.tsv")
    ap.add_argument("--bin-start", type=int, default=0, help="Start distance bin (inclusive).")
    ap.add_argument("--bin-end", type=int, default=100, help="End distance bin (inclusive).")
    ap.add_argument("--x-scale", choices=["linear", "log"], default="linear", help="X axis scale.")
    ap.add_argument("--y-scale", choices=["linear", "log"], default="linear", help="Y axis scale for counts.")
    ap.add_argument("--prefix", default="lineage_v16", help="Output filename prefix inside WD.")
    args = ap.parse_args()

    df = pd.read_csv(args.lineage_tsv, sep="\t")
    required = {"Extended_lineage", "Class"}
    if not required.issubset(df.columns):
        raise ValueError(f"Input is missing required columns: {required - set(df.columns)}")

    out_prefix = os.path.join(args.wd, args.prefix)

    # Enforce required bin windows per figure family
    COUNTS_START, COUNTS_END = 1, 100      # exclude 0 (contiguous)
    ECDF_START,   ECDF_END   = 0, 100      # include 0 (contiguous)

    # Terminal / Interstitial
    for cls in ["Terminal", "Interstitial"]:
        multipanel_hist_counts(df, cls, COUNTS_START, COUNTS_END, args.x_scale, args.y_scale, out_prefix)
        multipanel_ecdf(df, cls, ECDF_START, ECDF_END, args.x_scale, out_prefix)
        singlepanel_fraction_ecdf(df, cls, ECDF_START, ECDF_END, args.x_scale, out_prefix)

    # BOTH
    multipanel_hist_fractions_both(df, COUNTS_START, COUNTS_END, args.x_scale, out_prefix)
    multipanel_ecdf_both(df, ECDF_START, ECDF_END, args.x_scale, out_prefix)
    singlepanel_fraction_ecdf_both(df, ECDF_START, ECDF_END, args.x_scale, out_prefix)

if __name__ == "__main__":
    main()
