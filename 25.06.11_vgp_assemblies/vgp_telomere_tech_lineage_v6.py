#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vgp_telomere_tech_v6.py

Purpose
-------
Create raincloud plots (right half-violin + thin box + left-offset jitter) comparing
HiFi vs CLR assemblies for telomere metrics, following the exact STYLE used in
plot_distributions_v9.py.

Inputs
------
./25.09.25_metadata_telomere_assembly_simplified.csv  (columns include:)
Accession,Clade,total_paths,total_gaps,total_telomeres,mean_length,median_length,
min_length,max_length,two_telomeres,one_telomere,no_telomeres,t2t,gapped_t2t,
missassembled,gapped_missassembled,incomplete,gapped_incomplete,no_telomeres_comp,
gapped_no_telomeres,discordant,gapped_discordant,core_id,version_num,with_telomeres,
n_chromosomes,two_telomeres_pct,one_telomere_pct,with_telomeres_pct,
total_telomeres_pct_of_ends,Technology,Extended lineage,Assembly tech

Outputs
-------
./25.09.26_plots/tech/
  <metric>_raincloud_all.{png,pdf}
  <metric>_raincloud_lineages.{png,pdf}

Stats & Tests
-------------
- Groups: Technology ∈ {HiFi, CLR} (ONT ignored).
- Tests: Mann–Whitney U (two-sided), non-parametric and robust to non-normality.
  Rationale: we compare *per-assembly continuous percentages/lengths* (0–100 for %),
  not Bernoulli events; a Two-Proportion Z-Test applies to binary outcomes only.
- P-adjustment: Bonferroni across the four metrics for the global plots; for lineage
  panels, Bonferroni across the number of panels with data for that metric.
- For all p-values, OUTLIERS ARE EXCLUDED to match the violin/box (outliers still shown in scatter).
- Plots are annotated only with the Bonferroni-adjusted p-value.

Style
-----
- Matplotlib default style; minimalist axes (top/right spines off).
- Right half-violin (KDE bw=0.3), thin box, jitter points offset left by 0.30.
- Colors: HiFi=#CC79A7, CLR=#009E73.

Changes from v5
-------
- P-values now computed on NON-OUTLIERS ONLY (per group).
- Removed clade-based panels; only lineage-based panels remain.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from scipy.stats import gaussian_kde, mannwhitneyu

# ───────────────────────────────────────────── Config / paths
IN_CSV = "./25.09.25_metadata_telomere_assembly_simplified.csv"
OUT_DIR = "./25.09.26_plots/tech"
os.makedirs(OUT_DIR, exist_ok=True)

# Metrics to analyze and plot
METRICS = [
    "mean_length",
    "median_length",
    "total_telomeres_pct_of_ends",
    "with_telomeres_pct",
]

# Technology palette (deep colors, consistent with previous style)
PALETTE = {"HiFi": "#CC79A7", "CLR": "#009E73"}

# ── Extended lineage order & palette (subset and order per spec)
lineage_order = [
    "Other Deuterostomes",
    "Cartilaginous fishes",
    "Ray-finned fishes",
    "Amphibians",
    "Lepidosauria",
    "Turtles",
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

# Random state for reproducible jitter
RNG = np.random.default_rng(7)

# ───────────────────────────────────────────── Helpers (style-matched)

def lighten_color(hex_color, amount=0.55):
    rgb = np.array(mcolors.to_rgb(hex_color))
    white = np.array([1.0, 1.0, 1.0])
    mixed = rgb * (1 - amount) + white * amount
    return mcolors.to_hex(mixed, keep_alpha=False)

def savefig(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=600, bbox_inches="tight")

def bonferroni_adjust(pvals):
    """Bonferroni adjustment; returns list aligned to input order (NaNs preserved)."""
    m = sum(pd.notna(p) for p in pvals)
    if m == 0:
        return [np.nan] * len(pvals)
    out = []
    for p in pvals:
        if pd.isna(p):
            out.append(np.nan)
        else:
            out.append(float(min(1.0, p * m)))
    return out

def _draw_half_violin(ax, center_x, vals, color, width_violin=0.28):
    """Right half-violin with KDE (slightly thinner)."""
    vals = np.asarray(vals, dtype=float)
    if len(vals) > 1 and np.nanmax(vals) > np.nanmin(vals):
        kde = gaussian_kde(vals, bw_method=0.3)
        y = np.linspace(np.nanmin(vals), np.nanmax(vals), 300)
        dens = kde(y)
        if dens.max() > 0:
            scale = width_violin / dens.max()
            x_right = center_x + dens * scale
            ax.fill_betweenx(y, center_x, x_right, alpha=0.6, linewidth=0, color=color, zorder=1)
            ax.plot(x_right, y, linewidth=1.0, color=color, zorder=2)

def _draw_box(ax, center_x, vals, color, box_width=0.09):
    """Thin boxplot."""
    sns.boxplot(
        x=np.full(shape=len(vals), fill_value=center_x, dtype=float),
        y=vals,
        width=box_width,
        showcaps=True,
        boxprops={'facecolor': color, 'edgecolor': 'black', 'linewidth': 1.0},
        whiskerprops={'color': 'black'},
        capprops={'color': 'black'},
        medianprops={'color': 'black', 'linewidth': 1.0},
        showfliers=False,
        orient='v',
        ax=ax,
        zorder=3
    )

def _pretty_ylabel(name: str) -> str:
    """Human-friendly y-labels per spec."""
    mapping = {
        "with_telomeres_pct": "chrs. with telomeres %",
        "total_telomeres_pct_of_ends": "total telomeres %",
        "mean_length": "mean telomere length (Kbp)",
        "median_length": "median telomere length (Kbp)",
    }
    if name in mapping:
        return mapping[name]
    return name.replace("_", " ").replace("pct", "%")

def _apply_headroom(ax, series: pd.Series, metric: str, frac=0.12):
    """Add vertical headroom to avoid label overlap with top datapoints."""
    y = series.dropna().values
    if y.size == 0:
        return
    ymin = np.nanmin(y)
    ymax = np.nanmax(y)
    if np.isfinite(ymin) and np.isfinite(ymax):
        yrng = max(1e-9, ymax - ymin)
        top = ymax + frac * yrng
        # For percentage-like metrics, ensure at least up to 110
        if "pct" in metric:
            top = max(top, 110.0)
            ymin = min(ymin, 0.0)
        ax.set_ylim(ymin, top)

def _split_outliers(vals):
    """Return (non_outliers, outliers) using 1.5*IQR rule."""
    vals = np.asarray(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return vals, np.array([])
    q1, q3 = np.percentile(vals, [25, 75])
    iqr = q3 - q1
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    mask = (vals >= lo) & (vals <= hi)
    return vals[mask], vals[~mask]

def raincloud_plot_tech(ax, df, value_col, *,
                        jitter_sd=0.035, box_width=0.09, violin_width=0.28,
                        outlier_mode=False, min_nonout=1, title=None, title_color="#000000",
                        tick_n_from="all"):
    """
    Two-group raincloud (Technology on X): CLR then HiFi.
    - outlier_mode=False  -> standard: all points included in violin/box, hollow scatter
    - outlier_mode=True   -> violin/box use NON-OUTLIERS ONLY; scatter shows ALL points:
                             non-outliers hollow with tech color edges; outliers hollow
                             with light-grey edges; groups with <min_nonout non-outliers
                             are omitted entirely.
    tick_n_from: "all" (count all non-NaN points) or "nonout" (count only non-outliers)
    """
    tech_order_full = ["CLR", "HiFi"]  # CLR first, then HiFi

    # Prepare per-tech values (and outliers if requested)
    per = {}
    present = []
    for tech in tech_order_full:
        vals_all = df.loc[df["Technology"] == tech, value_col].dropna().values
        if outlier_mode:
            vals_nonout, vals_out = _split_outliers(vals_all)
            if len(vals_nonout) >= min_nonout:
                per[tech] = (vals_nonout, vals_out, vals_all)
                present.append(tech)
        else:
            if len(vals_all) > 0:
                per[tech] = (vals_all, np.array([]), vals_all)  # no outlier distinction
                present.append(tech)

    if len(present) == 0:
        ax.axis("off")
        return

    positions = np.arange(len(present), dtype=float)
    sizes = 18

    # Compose tick labels with (n=...) line
    tick_labels = []
    for tech in present:
        _, _, vals_all = per[tech]
        if outlier_mode and tick_n_from == "nonout":
            n = per[tech][0].size
        else:
            n = vals_all.size
        tick_labels.append(f"{tech}\n(n={n})")

    for i, tech in enumerate(present):
        deep = PALETTE[tech]
        vals_nonout, vals_out, vals_all = per[tech]

        # layers: violin/box from NON-OUTLIERS ONLY
        _draw_half_violin(ax, positions[i], vals_nonout, deep, width_violin=violin_width)
        _draw_box(ax, positions[i], vals_nonout, deep, box_width=box_width)

        # scatter: ALL points as hollow circles; outliers get light-grey edges
        x_jit_all = RNG.normal(loc=positions[i] - 0.30, scale=jitter_sd, size=len(vals_all))
        if vals_out.size > 0:
            lightgrey = "#B3B3B3"
            outs_set = set(np.round(vals_out, 12))
            edgecols = [lightgrey if (np.round(v, 12) in outs_set) else deep for v in vals_all]
        else:
            edgecols = [deep] * len(vals_all)

        ax.scatter(x_jit_all, vals_all, s=sizes, linewidths=0.6, facecolors='none',
                   edgecolors=edgecols, zorder=4)

    ax.set_xticks(positions, tick_labels, fontsize=10)
    ax.set_xlim(-0.6, len(present) - 0.4)
    ax.set_ylabel(_pretty_ylabel(value_col))
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # add headroom
    _apply_headroom(ax, df[value_col], value_col)

    if title is not None:
        ax.set_title(title, fontsize=11, fontweight="bold", color=title_color, pad=6)

# ───────────────────────────────────────────── Stats

def mannwhitney_p(a, b):
    """Two-sided Mann–Whitney U p-value, or NaN if not computable."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    p = mannwhitneyu(a, b, alternative="two-sided").pvalue
    return float(p)

# ───────────────────────────────────────────── Main workflow

def main():
    # Load
    df = pd.read_csv(IN_CSV)
    if "Technology" not in df.columns:
        raise RuntimeError("Column 'Technology' not found in input CSV.")
    if "Extended lineage" not in df.columns:
        raise RuntimeError("Column 'Extended lineage' not found in input CSV.")

    # Normalize technology labels, filter to HiFi/CLR only
    df["Technology"] = df["Technology"].astype(str).str.strip()
    df = df[df["Technology"].isin(["HiFi", "CLR"])].copy()

    # Ensure metric columns are numeric
    for m in METRICS:
        df[m] = pd.to_numeric(df[m], errors="coerce")

    # Convert lengths to Kbp
    if "mean_length" in df.columns:
        df["mean_length"] = df["mean_length"] / 1000.0
    if "median_length" in df.columns:
        df["median_length"] = df["median_length"] / 1000.0

    # ── GLOBAL (all datapoints): compute raw p-values across the four metrics USING NON-OUTLIERS ONLY
    pvals_global = []
    for m in METRICS:
        a_all = df.loc[df["Technology"] == "HiFi", m].dropna().values
        b_all = df.loc[df["Technology"] == "CLR",  m].dropna().values
        a_non, _ = _split_outliers(a_all)
        b_non, _ = _split_outliers(b_all)
        pvals_global.append(mannwhitney_p(a_non, b_non))
    padj_global = bonferroni_adjust(pvals_global)
    padj_map_global = {m: padj_global[i] for i, m in enumerate(METRICS)}

    # ── Plotter: single global plot per metric (compact)
    def plot_global(metric):
        sub = df.copy()
        padj = padj_map_global[metric]

        fig, ax = plt.subplots(figsize=(4, 5))
        fig.subplots_adjust(bottom=0.20)  # room for multi-line x labels

        raincloud_plot_tech(
            ax, sub, metric,
            jitter_sd=0.038,            # smaller jitter
            box_width=0.09,             # thinner box
            violin_width=0.26,          # thinner violin
            outlier_mode=True,          # scatter all; violin/box without outliers
            min_nonout=1,
            tick_n_from="all"
        )

        padj_str = "NA" if pd.isna(padj) else f"{padj:.3g}"
        ax.text(0.02, 0.98, f"p-adj={padj_str}", transform=ax.transAxes,
                va="top", ha="left", fontsize=9)

        stem = os.path.join(OUT_DIR, f"{metric}_raincloud_all")
        savefig(fig, stem)
        plt.close(fig)

    # ── Plotter: multi-panel by lineage per metric (with outlier logic)
    def plot_by_lineage(metric):
        # compute lineage-wise raw p-values (NON-OUTLIERS ONLY) in requested order
        raw = []
        for lin in lineage_order:
            sub = df[df["Extended lineage"] == lin]
            if sub.empty:
                raw.append(np.nan)
                continue
            a_all = sub.loc[sub["Technology"] == "HiFi", metric].dropna().values
            b_all = sub.loc[sub["Technology"] == "CLR",  metric].dropna().values
            a_non, _ = _split_outliers(a_all)
            b_non, _ = _split_outliers(b_all)
            raw.append(mannwhitney_p(a_non, b_non))
        padj_lineages = bonferroni_adjust(raw)

        # 4x2 layout, 12x8 figure
        ncols, nrows = 4, 2
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 6), sharey=False)
        axes = np.atleast_2d(axes).flatten()

        for i, lin in enumerate(lineage_order):
            ax = axes[i]
            sub = df[df["Extended lineage"] == lin]
            if sub.empty:
                ax.axis("off")
                continue

            raincloud_plot_tech(
                ax, sub, metric,
                jitter_sd=0.038,
                box_width=0.09,
                violin_width=0.26,
                outlier_mode=True,
                min_nonout=3,
                title=lin,
                title_color=lineage_colors.get(lin, "#000000"),
                tick_n_from="nonout"
            )

            p_adj = padj_lineages[i] if i < len(padj_lineages) else np.nan
            padj_str = "NA" if pd.isna(p_adj) else f"{p_adj:.3g}"
            ax.text(0.02, 0.98, f"p-adj={padj_str}", transform=ax.transAxes,
                    va="top", ha="left", fontsize=8)

        fig.tight_layout()
        stem = os.path.join(OUT_DIR, f"{metric}_raincloud_lineages")
        savefig(fig, stem)
        plt.close(fig)

    # ── Generate plots
    for m in METRICS:
        plot_global(m)
        plot_by_lineage(m)

    print(f"Plots written to: {OUT_DIR}")

if __name__ == "__main__":
    main()
