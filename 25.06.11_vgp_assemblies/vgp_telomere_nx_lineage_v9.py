#!/usr/bin/env python3
"""
Telomere Nx plots for VGP assemblies – per-run workflow (v9).

Changes vs. v8
--------------
- Add sample sizes to legends (single-panel) and subplot titles (multipanel).
- Move Technology legend to lower-right and increase font size on multipanels.
- Compute lineage-level mean/median telomere N50 (Nx at 50%) and append to IN_CSV.
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# ───────────────────────────────────────────────────────────────────────────────
# Directory structure identical to vgp_postprocess_v2.py
# ───────────────────────────────────────────────────────────────────────────────
base_dir = "/mnt/d/research/teloscope_article/25.06.11_vgp_assemblies"
runs = [
    # ("processed_NNNGGG_rgemi/primary", "NNNGGG"),
    # ("processed_TTAGGG_rgemi/primary", "TTAGGG"),
    ("processed_NNNGGG_ut50k/primary", "NNNGGG"),
    ("processed_TTAGGG_ut50k/primary", "TTAGGG"),
]

plot_dir = os.path.join(base_dir, "25.09.26_plots")
os.makedirs(plot_dir, exist_ok=True)
lineage_dir = os.path.join(plot_dir, "lineage")
os.makedirs(lineage_dir, exist_ok=True)

# Metadata for technology per assembly
IN_CSV = "./25.09.25_metadata_telomere_assembly_simplified.csv"

# ───────────────────────────────────────────────────────────────────────────────
# Helper functions
# ───────────────────────────────────────────────────────────────────────────────
def compute_Nx_curve(lengths: np.ndarray):
    """Return cumulative % (ascending) and sorted fragment lengths (descending)."""
    sorted_lengths = np.sort(lengths)[::-1]
    cumsum = np.cumsum(sorted_lengths)
    percentages = cumsum / cumsum[-1] * 100
    return percentages, sorted_lengths


def Nx_at_percent(percentages: np.ndarray, lengths: np.ndarray, p: int) -> float:
    """Given a single assembly's Nx curve, return the length (bp) at percentile p (1–100)."""
    idx = np.searchsorted(percentages, p, side="left")
    idx = min(idx, len(lengths) - 1)
    return lengths[idx]


def normalize_tech(s: str) -> str:
    """Map free-text technology to 'HiFi' or 'CLR' (default HiFi if unknown)."""
    if not isinstance(s, str):
        return "HiFi"
    t = s.lower()
    if "hifi" in t:
        return "HiFi"
    if "clr" in t:
        return "CLR"
    # fall back to HiFi if unspecified/other
    return "HiFi"


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

# Pre-allocate Nx steps (1 … 100 %)
perc_steps = np.arange(1, 101)
p50_index = np.where(perc_steps == 50)[0][0]  # index of Nx=50 in the Nx vectors

# ───────────────────────────────────────────────────────────────────────────────
# Load technology metadata
# ───────────────────────────────────────────────────────────────────────────────
tech_df = pd.read_csv(IN_CSV, dtype=str).fillna("")
# Build both full- and core- accession maps
_acc2tech = {a: normalize_tech(t) for a, t in zip(tech_df.get("Accession", []), tech_df.get("Technology", []))}
_core2tech = {str(a).split(".")[0]: normalize_tech(t) for a, t in zip(tech_df.get("Accession", []), tech_df.get("Technology", []))}

def get_tech_for_assembly(acc: str) -> str:
    if acc in _acc2tech:
        return _acc2tech[acc]
    core = str(acc).split(".")[0]
    if core in _core2tech:
        return _core2tech[core]
    return "HiFi"

# ───────────────────────────────────────────────────────────────────────────────
# Main loop – one run at a time
# ───────────────────────────────────────────────────────────────────────────────
for run_subpath, run_label in runs:
    bed_file = os.path.join(
        base_dir, run_subpath,
        f"merged_{run_label}_terminal_telomeres_longest_pq.bed"
    )
    if not os.path.isfile(bed_file):
        print(f("[WARN] Missing BED file: {bed_file}"))
        continue

    # ── load BED
    df = pd.read_csv(
        bed_file, sep="\t", header=None,
        names=[
            "chr", "start", "end", "length", "label",
            "fwdCounts", "revCounts", "canCounts", "nonCanCounts",
            "chr_length", "assembly", "clade", "extended_lineage"
        ]
    )

    # ───────────────────────────────────────────────────────────────────────────
    # Compute per-assembly Nx arrays grouped by lineage (and technology)
    # ───────────────────────────────────────────────────────────────────────────
    lineage_Nx_table = {lin: [] for lin in lineage_order}  # all tech combined
    lineage_Nx_by_tech = {lin: {"HiFi": [], "CLR": []} for lin in lineage_order}

    for (assembly, lineage), group in df.groupby(["assembly", "extended_lineage"]):
        if lineage not in lineage_order:
            continue
        percentages, lengths = compute_Nx_curve(group["length"].to_numpy())
        nx_vec = [Nx_at_percent(percentages, lengths, p) for p in perc_steps]
        lineage_Nx_table[lineage].append(nx_vec)

        tech = get_tech_for_assembly(assembly)
        lineage_Nx_by_tech[lineage][tech].append(nx_vec)

    # Convert to numpy arrays
    for lin in lineage_Nx_table:
        lineage_Nx_table[lin] = np.array(lineage_Nx_table[lin])  # (nAsm, 100) or (0,)
        for tech in ("HiFi", "CLR"):
            lineage_Nx_by_tech[lin][tech] = np.array(lineage_Nx_by_tech[lin][tech])

    # ───────────────────────────────────────────────────────────────────────────
    # Compute lineage-level N50 metrics and write back to IN_CSV
    # ───────────────────────────────────────────────────────────────────────────
    mean_telo_N50_by_lineage = {}
    median_telo_N50_by_lineage = {}
    for lin in lineage_order:
        arr = lineage_Nx_table[lin]
        if arr.size == 0:
            mean_telo_N50_by_lineage[lin] = np.nan
            median_telo_N50_by_lineage[lin] = np.nan
        else:
            mean_curve = arr.mean(axis=0)
            median_curve = np.median(arr, axis=0)
            mean_telo_N50_by_lineage[lin] = float(mean_curve[p50_index])
            median_telo_N50_by_lineage[lin] = float(median_curve[p50_index])

    # Map each Accession in IN_CSV to lineage from current BED (if present)
    asm_lineage_df = df[["assembly", "extended_lineage"]].drop_duplicates()
    asm2lin_full = {a: l for a, l in zip(asm_lineage_df["assembly"], asm_lineage_df["extended_lineage"])}
    asm2lin_core = {str(a).split(".")[0]: l for a, l in zip(asm_lineage_df["assembly"], asm_lineage_df["extended_lineage"])}

    def lineage_for_acc(a: str):
        if a in asm2lin_full:
            return asm2lin_full[a]
        core = str(a).split(".")[0]
        return asm2lin_core.get(core, None)

    # Create/overwrite columns in metadata with lineage-level N50s (bp)
    mean_col = f"mean_telo_N50"
    median_col = f"median_telo_N50"

    tech_df[mean_col] = tech_df["Accession"].map(lambda a: mean_telo_N50_by_lineage.get(lineage_for_acc(a), np.nan))
    tech_df[median_col] = tech_df["Accession"].map(lambda a: median_telo_N50_by_lineage.get(lineage_for_acc(a), np.nan))

    # Persist metadata with new columns
    tech_df.to_csv(IN_CSV, index=False)

    # ───────────────────────────────────────────────────────────────────────────
    # Plot 1 – Mean Nx curves (single panel – all lineages) with n in legend
    # ───────────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for lin in lineage_order:
        arr = lineage_Nx_table[lin]
        if arr.size == 0:
            continue
        n = arr.shape[0]
        mean_curve = arr.mean(axis=0) / 1000.0
        ax.step(perc_steps, mean_curve, where="mid", color=lineage_colors[lin], linewidth=2, label=f"{lin} (n={n})")

    ax.set_xlabel("Nx (%)")
    ax.set_ylabel("Telomere length (Kbp)")
    ax.set_title("Mean Telomere Nx by Lineage")
    ax.set_xlim(0, 101)
    ax.legend(title="Lineage", fontsize=8, title_fontsize=9)
    sns.despine(ax=ax, top=True, right=True)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(
            os.path.join(lineage_dir, f"Telomere_Nx_mean_by_lineage_{run_label}.{ext}"),
            dpi=600 if ext == "png" else None
        )
    plt.close(fig)

    # ───────────────────────────────────────────────────────────────────────────
    # Plot 2 – Median Nx curves (single panel – all lineages) with n in legend
    # ───────────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for lin in lineage_order:
        arr = lineage_Nx_table[lin]
        if arr.size == 0:
            continue
        n = arr.shape[0]
        median_curve = np.median(arr, axis=0) / 1000.0
        ax.step(perc_steps, median_curve, where="mid", color=lineage_colors[lin], linewidth=2, label=f"{lin} (n={n})")

    ax.set_xlabel("Nx (%)")
    ax.set_ylabel("Telomere length (Kbp)")
    ax.set_title("Median Telomere Nx by Lineage")
    ax.set_xlim(0, 101)
    ax.legend(title="Lineage", fontsize=8, title_fontsize=9)
    sns.despine(ax=ax, top=True, right=True)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(
            os.path.join(lineage_dir, f"Telomere_Nx_median_by_lineage_{run_label}.{ext}"),
            dpi=600 if ext == "png" else None
        )
    plt.close(fig)

    # ───────────────────────────────────────────────────────────────────────────
    # Plot 3 – Mean Nx curves (multipanel – by lineage; tech-styled)
    # ───────────────────────────────────────────────────────────────────────────
    n_cols_l, n_rows_l = 6, 2
    fig, axes = plt.subplots(n_rows_l, n_cols_l, figsize=(16, 6), sharex=True, sharey=True)
    axes = axes.flatten()

    # legend handles for tech styles (bigger, lower-right)
    tech_handles = [
        Line2D([0], [0], color="black", linestyle="-", linewidth=2, label="HiFi"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=2, label="CLR"),
    ]

    for i, lin in enumerate(lineage_order):
        ax = axes[i]
        # HiFi mean
        arr_h = lineage_Nx_by_tech[lin]["HiFi"]
        if arr_h.size > 0:
            ax.step(perc_steps, arr_h.mean(axis=0) / 1000.0, where="mid",
                    color=lineage_colors[lin], linestyle="-", linewidth=2)
        # CLR mean
        arr_c = lineage_Nx_by_tech[lin]["CLR"]
        if arr_c.size > 0:
            ax.step(perc_steps, arr_c.mean(axis=0) / 1000.0, where="mid",
                    color=lineage_colors[lin], linestyle="--", linewidth=2)

        n_total = lineage_Nx_table[lin].shape[0]
        ax.set_title(f"{lin}\n(n={n_total})")
        ax.set_xlim(0, 101)
        ax.set_xlabel("Nx (%)")
        ax.set_ylabel("Telomere length (Kbp)")
        sns.despine(ax=ax, top=True, right=True)

    # remove unused subplots (if any)
    for j in range(len(lineage_order), len(axes)):
        fig.delaxes(axes[j])

    fig.legend(handles=tech_handles, title="Technology", loc="lower right", fontsize=14, title_fontsize=16)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(
            os.path.join(lineage_dir, f"Telomere_Nx_multipanel_mean_by_lineage_{run_label}.{ext}"),
            dpi=600 if ext == "png" else None
        )
    plt.close(fig)

    # ───────────────────────────────────────────────────────────────────────────
    # Plot 4 – Median Nx curves (multipanel – by lineage; tech-styled)
    # ───────────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(n_rows_l, n_cols_l, figsize=(16, 6), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, lin in enumerate(lineage_order):
        ax = axes[i]
        # HiFi median
        arr_h = lineage_Nx_by_tech[lin]["HiFi"]
        if arr_h.size > 0:
            ax.step(perc_steps, np.median(arr_h, axis=0) / 1000.0, where="mid",
                    color=lineage_colors[lin], linestyle="-", linewidth=2)
        # CLR median
        arr_c = lineage_Nx_by_tech[lin]["CLR"]
        if arr_c.size > 0:
            ax.step(perc_steps, np.median(arr_c, axis=0) / 1000.0, where="mid",
                    color=lineage_colors[lin], linestyle="--", linewidth=2)

        n_total = lineage_Nx_table[lin].shape[0]
        ax.set_title(f"{lin}\n(n={n_total})")
        ax.set_xlim(0, 101)
        ax.set_xlabel("Nx (%)")
        ax.set_ylabel("Telomere length (Kbp)")
        sns.despine(ax=ax, top=True, right=True)

    # remove unused subplots (if any)
    for j in range(len(lineage_order), len(axes)):
        fig.delaxes(axes[j])

    fig.legend(handles=tech_handles, title="Technology", loc="lower right", fontsize=14, title_fontsize=16)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(
            os.path.join(lineage_dir, f"Telomere_Nx_multipanel_median_by_lineage_{run_label}.{ext}"),
            dpi=600 if ext == "png" else None
        )
    plt.close(fig)

    print(f"[OK] Plots + metadata update for run {run_label} saved to {lineage_dir} and {IN_CSV}")
