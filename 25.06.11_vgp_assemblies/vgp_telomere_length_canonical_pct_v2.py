#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vgp_telomere_length_canonical_pct_v2.py

Goal
-----
Create a versatile plotting script to visualize terminal telomere length (kbp)
vs. canonical proportion (%) from Teloscope BED-like annotations.

Modes (set below; no CLI)
-------------------------
Mode 1 (BASIC):    Read one Teloscope BED file (10 columns).
Mode 2 (EXTENDED): Read one extended BED/TSV with extra columns (accession, ucscTaxa, vgpTaxa);
                   color/legend by a chosen group column (e.g., vgpTaxa).
Mode 3 (MULTI):    Read ALL files matching "*_terminal_telomeres.bed" inside FOLDERPATH,
                   concatenate, add 'source_file' and inferred 'accession' from filename, and color/legend by accession.

Inputs
------
Teloscope columns (first 10), in order:
  header, start, end, length, label, fwdCounts, revCounts, canCounts, nonCanCounts, chrSize

Outputs
-------
Creates folder: "<WD>/25.11.07_plots"
Saves:          scatter PNG and PDF with minimalistic styling.

Changes in this version
------
1. Keep the column name 'header' (no renaming to 'chr'/'chrName') throughout.
2. MODE 2: add fixed lineage palette + name mapping + ordered legend for taxa columns.
3. Robust parsing for extended files without header row: if ≥13 cols, assign
   the last three as accession/ucscTaxa/vgpTaxa so MODE2_GROUP_COL works.
"""

import os
import re
import glob
import math
from typing import List, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# CONFIG (EDIT HERE; no CLI)
# ---------------------------------------------------------------------

WD = "/mnt/d/research/teloscope_article/25.06.11_vgp_assemblies/"

# Choose one mode: 1 (basic single file), 2 (extended single file), 3 (multi-files folder)
MODE = 2

# Mode 1 (basic single file; only the first 10 columns are required)
MODE1_FILEPATH = os.path.join(
    WD,
    "processed_TTAGGG_ut100k/primary/merged_TTAGGG_terminal_telomeres_longest_pq.bed"
)

# Mode 2 (extended single file; has columns 'accession', 'ucscTaxa', 'vgpTaxa')
MODE2_FILEPATH = os.path.join(
    WD,
    "./processed_TTAGGG_ut100k/primary/merged_TTAGGG_terminal_telomeres_longest_pq_sharks.bed"
)
# Which column to group/color by in MODE=2 ('vgpTaxa', 'accession', or 'ucscTaxa')
MODE2_GROUP_COL = "vgpTaxa"

# Mode 3 (folder with many Teloscope outputs; merge all "*_terminal_telomeres.bed")
MODE3_FOLDERPATH = os.path.join(
    WD,
    "processed_TTAGGG_ut100k/primary"
)
MODE3_GLOB_PATTERN = "**/*_terminal_telomeres.bed"  # recursive

# Plot options (shared across modes)
OUT_DIR = os.path.join(WD, "25.11.07_plots/ut100k")
FIG_BASENAME = "telomere_length_vs_canonical_pct_sharks"

# X axis: 'linear' or 'log' (log10). The requested ticks are at 1, 10, 100 kbp.
X_SCALE = "logw"   # change to "log" for log10
# Y axis min: "zero" -> 0 to 100, "data" -> min(data) to 100 (as requested)
Y_MIN_MODE = "data"  # "zero" or "data"

# Marker aesthetics
MARKER_SIZE = 18
ALPHA = 0.8
EDGEWIDTH = 0.0

# ---------------------------------------------------------------------
# MODE 2 PALETTE & ORDER
# ---------------------------------------------------------------------

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
    # identities pass through: Mammals, Birds, Lepidosauria, Amphibians, Turtles, Cyclostomes
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

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

TEL_COLS_10 = [
    "header", "start", "end", "length", "label",
    "fwdCounts", "revCounts", "canCounts", "nonCanCounts", "chrSize"
]


def try_read_file(path: str) -> pd.DataFrame:
    """
    Read a Teloscope BED-like file robustly.
    - Try header=0 first; if required columns exist, use them.
    - Else read with header=None and assign names for at least the first 10 columns.
    - If there are ≥13 columns, assign the next three as accession/ucscTaxa/vgpTaxa.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    # Attempt 1: header row present
    try:
        df = pd.read_csv(path, sep="\t", header=0, comment="#", dtype=str)
        if "length" in df.columns and "canCounts" in df.columns:
            df = coerce_tel_cols(df)
            return df
    except Exception:
        pass

    # Attempt 2: no header row
    raw = pd.read_csv(path, sep="\t", header=None, comment="#", dtype=str)
    # Ensure at least 10 columns
    if raw.shape[1] < 10:
        raise ValueError(f"File has fewer than 10 columns: {path}")

    # Assign the first 10 canonical column names
    ncols = raw.shape[1]
    if ncols >= 13:
        # 10 teloscope + 3 extended, preserve any further extras generically
        extra_n = ncols - 13
        extra_cols = [f"extra{i+1}" for i in range(extra_n)]
        raw.columns = TEL_COLS_10 + ["accession", "ucscTaxa", "vgpTaxa"] + extra_cols
    else:
        extra_n = ncols - 10
        extra_cols = [f"extra{i+1}" for i in range(extra_n)]
        raw.columns = TEL_COLS_10 + extra_cols

    # If the first line was actually a header mistakenly read as data, attempt fix.
    try:
        if (str(raw.iloc[0]["header"]).lower() in {"header", "chr", "chrname"}) or (raw.iloc[0]["length"] == "length"):
            df = pd.read_csv(path, sep="\t", header=0, comment="#", dtype=str)
            df = coerce_tel_cols(df)
            return df
    except Exception:
        pass

    df = coerce_tel_cols(raw)
    return df


def coerce_tel_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the first 10 Teloscope columns exist exactly as specified and coerce types.
    No renaming of 'header' to any alternative.
    """
    df = df.copy()

    # Make sure first 10 columns exist (case-insensitive check/soft rename on exact case only)
    for c in TEL_COLS_10:
        if c not in df.columns:
            # Attempt soft recovery if a differently cased version exists (but do NOT map to 'chr')
            alt = [x for x in df.columns if x.lower() == c.lower()]
            if alt:
                df.rename(columns={alt[0]: c}, inplace=True)

    missing = [c for c in TEL_COLS_10 if c not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing: {missing}\nColumns present: {list(df.columns)}")

    # Coerce dtypes
    int_cols = ["start", "end", "length", "fwdCounts", "revCounts", "canCounts", "nonCanCounts", "chrSize"]
    for c in int_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "label" in df.columns:
        df["label"] = df["label"].astype(str)

    return df


def infer_accession_from_filename(fname: str) -> str:
    """
    Infer an accession-like ID from a filename. Prefer GCF/GCA patterns; otherwise use base without suffixes.
    """
    base = os.path.basename(fname)
    m = re.search(r"(GC[AF]_\d+\.\d+)", base)
    if m:
        return m.group(1)
    if "." in base:
        return base.split(".")[0]
    return re.sub(r"_terminal_telomeres$", "", os.path.splitext(base)[0])


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute:
      - tel_length_kbp = length / 1000.0
      - canonical_pct  = (canCounts * 600) / length
    Drop length<=0 or NaN rows.
    Cap canonical_pct to [0, 100] for visualization sanity.
    """
    df = df.copy()
    df = df[pd.notnull(df["length"]) & (df["length"] > 0)]
    df["tel_length_kbp"] = df["length"] / 1000.0
    df["canonical_pct"] = (df["canCounts"] * 600.0) / df["length"]
    df["canonical_pct"] = df["canonical_pct"].clip(lower=0.0, upper=100.0)
    return df


def prepare_mode1(filepath: str) -> Tuple[pd.DataFrame, Optional[str]]:
    df = try_read_file(filepath)
    df = compute_metrics(df)
    return df, None  # no grouping


def prepare_mode2(filepath: str, group_col: str) -> Tuple[pd.DataFrame, Optional[str]]:
    df = try_read_file(filepath)

    # Ensure the explicit extended columns exist when requested
    if group_col not in df.columns:
        raise KeyError(f"Group column '{group_col}' not in columns: {list(df.columns)}")

    df = compute_metrics(df)
    return df, group_col


def prepare_mode3(folderpath: str, pattern: str) -> Tuple[pd.DataFrame, Optional[str]]:
    if not os.path.isdir(folderpath):
        raise NotADirectoryError(f"Not a directory: {folderpath}")

    files = glob.glob(os.path.join(folderpath, pattern), recursive=True)
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' found in: {folderpath}")

    dfs: List[pd.DataFrame] = []
    for fp in files:
        try:
            dfi = try_read_file(fp)
            dfi["source_file"] = os.path.basename(fp)
            dfi["accession"] = infer_accession_from_filename(fp)
            dfs.append(dfi)
        except Exception as e:
            print(f"[WARN] Skipping {fp}: {e}")

    if not dfs:
        raise RuntimeError("All files failed to parse; nothing to plot.")

    df = pd.concat(dfs, ignore_index=True)
    df = compute_metrics(df)

    if "accession" not in df.columns:
        raise KeyError("Expected 'accession' column could not be created in Mode 3.")
    return df, "accession"


def filtered_ticks(values: List[float], vmin: float, vmax: float) -> List[float]:
    return [v for v in values if (vmin is None or v >= vmin) and (vmax is None or v <= vmax)]


def minimalistic_box(ax: plt.Axes) -> None:
    for s in ["top", "right", "left", "bottom"]:
        ax.spines[s].set_linewidth(1.0)
    ax.tick_params(axis="both", which="both", length=4, width=1.0, direction="out")


def _norm_lineage_name(name: str) -> str:
    """Normalize dataset name to palette key."""
    n = str(name)
    return NAME_MAP.get(n, n)


def _ordered_groups_for_mode2(groups: List[str]) -> List[str]:
    """Order groups by ORDERED_DATASET_LINEAGES (after normalization), unknowns last alphabetically."""
    order_index = {k: i for i, k in enumerate(ORDERED_DATASET_LINEAGES)}
    def sort_key(g: str):
        norm = _norm_lineage_name(g)
        return (order_index.get(norm, 10**6), str(g))
    return sorted(groups, key=sort_key)


def make_scatter(df: pd.DataFrame, group_col: Optional[str], out_dir: str, fig_basename: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=600)

    # Plot
    if group_col is None:
        ax.scatter(
            df["tel_length_kbp"], df["canonical_pct"],
            s=MARKER_SIZE, alpha=ALPHA, linewidths=EDGEWIDTH
        )
        legend_title = None
        legend_items = None
    else:
        legend_title = group_col
        legend_items = []
        groups = df[group_col].astype(str).fillna("NA").unique().tolist()

        # MODE 2: use lineage palette and ordered legend when grouping by taxa/lineage columns
        if MODE == 2 and group_col in {"vgpTaxa", "ucscTaxa"}:
            groups = _ordered_groups_for_mode2(groups)
            for g in groups:
                sub = df[df[group_col].astype(str).fillna("NA") == g]
                norm = _norm_lineage_name(g)
                color = LINEAGE_PALETTE.get(norm, FALLBACK_COLOR)
                sc = ax.scatter(
                    sub["tel_length_kbp"], sub["canonical_pct"],
                    s=MARKER_SIZE, alpha=ALPHA, linewidths=EDGEWIDTH, label=str(g), c=[color]
                )
                legend_items.append(sc)
        else:
            # Generic grouping (e.g., 'accession')
            for g in sorted(groups):
                sub = df[df[group_col].astype(str).fillna("NA") == g]
                sc = ax.scatter(
                    sub["tel_length_kbp"], sub["canonical_pct"],
                    s=MARKER_SIZE, alpha=ALPHA, linewidths=EDGEWIDTH, label=str(g)
                )
                legend_items.append(sc)

    # Axes scales
    if X_SCALE.lower() == "log":
        ax.set_xscale("log")
    elif X_SCALE.lower() != "linear":
        raise ValueError("X_SCALE must be 'linear' or 'log'.")

    # Labels
    ax.set_xlabel("Telomere length (Kbp)")
    ax.set_ylabel("Canonical proportion (%)")

    # Y-limits (0..100 or data_min..100)
    if Y_MIN_MODE.lower() == "zero":
        ymin = 0.0
    elif Y_MIN_MODE.lower() == "data":
        ymin = max(0.0, float(df["canonical_pct"].min(skipna=True)))
        ymin = math.floor(ymin / 5.0) * 5.0
    else:
        raise ValueError("Y_MIN_MODE must be 'zero' or 'data'.")

    ymax = 100.0
    ax.set_ylim(ymin, ymax)

    # Ticks & grid (only at the specified values)
    y_ticks_all = [0, 20, 40, 60, 80, 100]
    y_ticks = filtered_ticks(y_ticks_all, ymin, ymax)
    ax.set_yticks(y_ticks)

    ax.relim()
    ax.autoscale_view()

    cur_xlim = list(ax.get_xlim())
    if X_SCALE.lower() == "log":
        min_pos = float(df["tel_length_kbp"][df["tel_length_kbp"] > 0].min())
        if not math.isfinite(min_pos):
            min_pos = 0.1
        cur_xlim[0] = max(cur_xlim[0], min_pos * 0.8)
        ax.set_xlim(cur_xlim)

    x_ticks_all = [1, 10, 100]
    cur_xlim = ax.get_xlim()
    x_ticks = filtered_ticks(x_ticks_all, cur_xlim[0], cur_xlim[1])
    ax.set_xticks(x_ticks)

    minimalistic_box(ax)
    ax.grid(True, which="major", axis="both", linewidth=0.7, alpha=0.35, color="lightgrey")

    # Legend (show all groups; no cap)
    if group_col is not None and legend_items:
        ax.legend(
            title=legend_title,
            frameon=False,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
            fontsize=8,
            title_fontsize=9,
            markerscale=1.0
        )
        plt.tight_layout(rect=[0, 0, 0.9, 1])  # leave room for legend
    else:
        plt.tight_layout()

    # Save
    mode_tag = f"mode{MODE}_{X_SCALE}_{Y_MIN_MODE}"
    png_path = os.path.join(out_dir, f"{fig_basename}_{mode_tag}.png")
    pdf_path = os.path.join(out_dir, f"{fig_basename}_{mode_tag}.pdf")
    fig.savefig(png_path, dpi=600)
    fig.savefig(pdf_path)
    print(f"[SAVE] {png_path}")
    print(f"[SAVE] {pdf_path}")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main() -> None:
    if MODE == 1:
        df, group_col = prepare_mode1(MODE1_FILEPATH)
    elif MODE == 2:
        df, group_col = prepare_mode2(MODE2_FILEPATH, MODE2_GROUP_COL)
    elif MODE == 3:
        df, group_col = prepare_mode3(MODE3_FOLDERPATH, MODE3_GLOB_PATTERN)
    else:
        raise ValueError("MODE must be one of: 1 (basic), 2 (extended), 3 (multi)")

    if df.empty:
        raise RuntimeError("No data to plot after parsing.")

    make_scatter(df, group_col, OUT_DIR, FIG_BASENAME)


if __name__ == "__main__":
    main()
