#!/usr/bin/env python3
"""
Multipanel scatter-plot of chromosome length vs. telomere length per lineage (v4)

• 6 × 2 grid (2 rows × 6 columns) on a compact 10 × 5-inch canvas
• Two figures per run:    linear-X   &   log10-X
• Points + regression line share the same deep colour per lineage
• 95 % CI band in grey
• Panel title = lineage; in-panel annotation = R² & BH-adjusted p-value
• No legend, no grid; only left/bottom spines
• Each panel keeps its own X/Y scale (no shared axes, all tick labels visible)
• Outlier handling: X-axis outliers are removed using IQR on log10(chr length),
  and excluded from both plotting and correlations.
• Additionally, an alternative set also removes Y-axis outliers (IQR on tel_len_kb).

Outputs
    <base_dir>/25.09.26_plots/chr_length/vgp_telomere_scatter_v4_<scale>_<run>.{png,pdf}
    <base_dir>/25.09.26_plots/chr_length/vgp_telomere_scatter_v4_xy_<scale>_<run>.{png,pdf}
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from scipy.stats import linregress
from matplotlib.ticker import LogLocator, NullFormatter, FuncFormatter

# ───────────────────────────────────────────────────────────────────────────────
# PATH CONFIG
# ───────────────────────────────────────────────────────────────────────────────
BASE_DIR = "/mnt/d/research/teloscope_article/25.06.11_vgp_assemblies"
RUNS = [
    ("processed_NNNGGG_ut50k/primary", "NNNGGG"),
    ("processed_TTAGGG_ut50k/primary", "TTAGGG"),
]
PLOT_DIR = os.path.join(BASE_DIR, "25.09.26_plots/chr_length")
os.makedirs(PLOT_DIR, exist_ok=True)

# ───────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLE & LINEAGES
# ───────────────────────────────────────────────────────────────────────────────
LINEAGE_ORDER = [
    "Other Deuterostomes",
    "Cartilaginous fishes",
    "Ray-finned fishes",
    "Amphibians",
    "Lepidosauria",
    "Turtles",
    "Birds",
    "Mammals",
]
LINEAGE_COLORS = {
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

sns.set_style("white")
sns.set_context("paper", font_scale=1.1)

# ───────────────────────────────────────────────────────────────────────────────
# HELPERS
# ───────────────────────────────────────────────────────────────────────────────
def filter_x_outliers_log_iqr(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Remove X-axis outliers per lineage using IQR on log10(chr_len_mb).
    This avoids clouding due to extreme chromosome lengths and applies
    consistently for both linear and log-scale plots.
    """
    out = []
    for lin in df_in["lineage"].dropna().unique():
        sub = df_in[df_in["lineage"] == lin].copy()
        if sub.shape[0] < 4:
            out.append(sub)
            continue
        logx = np.log10(sub["chr_len_mb"].values)
        q1, q3 = np.percentile(logx, [25, 75])
        iqr = q3 - q1
        if iqr == 0:
            out.append(sub); continue
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        keep = (logx >= lo) & (logx <= hi)
        out.append(sub.loc[keep])
    return pd.concat(out, axis=0, ignore_index=True)

def filter_y_outliers_iqr(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Remove Y-axis outliers per lineage using IQR on tel_len_kb (linear scale).
    """
    out = []
    for lin in df_in["lineage"].dropna().unique():
        sub = df_in[df_in["lineage"] == lin].copy()
        if sub.shape[0] < 4:
            out.append(sub)
            continue
        y = sub["tel_len_kb"].values
        q1, q3 = np.percentile(y, [25, 75])
        iqr = q3 - q1
        if iqr == 0:
            out.append(sub); continue
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        keep = (y >= lo) & (y <= hi)
        out.append(sub.loc[keep])
    return pd.concat(out, axis=0, ignore_index=True)

def _log_major_formatter(val, pos=None):
    """Label only powers of 10 as plain numbers (10, 100, 1000)."""
    if val <= 0:
        return ""
    exp = np.round(np.log10(val))
    if np.isclose(val, 10 ** exp):
        return f"{int(10 ** exp)}"
    return ""

# ───────────────────────────────────────────────────────────────────────────────
# PLOTTER
# ───────────────────────────────────────────────────────────────────────────────
def multipanel_scatter(df_raw, run_label: str, log_x: bool = False, filter_y: bool = False) -> None:
    """Build 6 × 2 multipanel scatter + regression figure."""
    present = [lin for lin in df_raw["lineage"].unique() if lin in LINEAGE_COLORS]
    extra = [lin for lin in present if lin not in LINEAGE_ORDER]
    lineage_list = LINEAGE_ORDER + [lin for lin in extra if lin not in LINEAGE_ORDER]

    # Apply outlier filters
    df = filter_x_outliers_log_iqr(df_raw)
    if filter_y:
        df = filter_y_outliers_iqr(df)

    n_rows, n_cols = 3, 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6), sharex=False, sharey=False)

    # p-values for slope (after filtering)
    pvals = []
    for lin in lineage_list:
        sub = df[df["lineage"] == lin]
        if len(sub) < 2:
            pvals.append(1.0); continue
        x_vals = np.log10(sub["chr_len_mb"]) if log_x else sub["chr_len_mb"]
        _, _, _, p, _ = linregress(x_vals, sub["tel_len_kb"])
        pvals.append(p)
    _, p_adj, _, _ = multipletests(pvals, method="fdr_bh") if len(pvals) else (None, [], None, None)
    p_adj_map = dict(zip(lineage_list, p_adj))

    # plot each lineage
    for idx, lin in enumerate(lineage_list):
        r, c = divmod(idx, n_cols)
        if r >= n_rows:
            break
        ax = axes[r, c]

        sub = df[df["lineage"] == lin]
        if sub.empty:
            ax.set_visible(False)
            continue

        color = LINEAGE_COLORS.get(lin, "#444444")
        ax.scatter(
            sub["chr_len_mb"], sub["tel_len_kb"],
            s=18, alpha=0.8, color=color, edgecolor="none"
        )

        # Regression + 95% CI
        x_raw = sub["chr_len_mb"].values
        if x_raw.min() == x_raw.max():
            x_fit = np.array([x_raw.min(), x_raw.max() + 1e-9])
        else:
            x_fit = np.linspace(x_raw.min(), x_raw.max(), 100)

        X     = sm.add_constant(np.log10(x_raw) if log_x else x_raw)
        X_fit = sm.add_constant(np.log10(x_fit)  if log_x else x_fit)
        model = sm.OLS(sub["tel_len_kb"], X).fit()
        preds = model.get_prediction(X_fit)
        y_fit = preds.predicted_mean
        ci_lo, ci_hi = preds.conf_int().T

        ax.fill_between(x_fit, ci_lo, ci_hi, color="grey", alpha=0.4, zorder=0)
        ax.plot(x_fit, y_fit, color=color, lw=1.4)

        if log_x:
            ax.set_xscale("log")
            # Force clean ticks at 10, 100, 1000… as plain numbers
            ax.xaxis.set_major_locator(LogLocator(base=10.0))
            ax.xaxis.set_major_formatter(FuncFormatter(_log_major_formatter))
            ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
            ax.xaxis.set_minor_formatter(NullFormatter())

        sns.despine(ax=ax, top=True, right=True)
        ax.set_title(lin, fontsize=9, pad=2)

        if r == n_rows - 1:
            ax.set_xlabel("Chr length (Mb)", fontsize=8)
        else:
            ax.set_xlabel("")
        if c == 0:
            ax.set_ylabel("Telomere length (Kb)", fontsize=8)
        else:
            ax.set_ylabel("")

        padj = p_adj_map.get(lin, np.nan)
        ax.text(
            0.04, 0.96,
            f"R² = {model.rsquared:.2f}\np-adj: {padj:.3g}" if not np.isnan(padj) else f"R² = {model.rsquared:.2f}\np-adj: NA",
            transform=ax.transAxes, va="top", ha="left", fontsize=7
        )

    total_panels = n_rows * n_cols
    for j in range(len(lineage_list), total_panels):
        r, c = divmod(j, n_cols)
        axes[r, c].set_visible(False)

    plt.tight_layout(w_pad=0.7, h_pad=1.0)
    tag = "log10" if log_x else "linear"
    suffix = "xy_" if filter_y else ""
    for ext, dpi in [("pdf", None), ("png", 600)]:
        fig.savefig(
            os.path.join(PLOT_DIR, f"vgp_telomere_scatter_v4_{suffix}{tag}_{run_label}.{ext}"),
            dpi=dpi, bbox_inches="tight"
        )
    plt.close(fig)
    print(f"[OK] Saved {tag} figure ({'X+Y outliers removed' if filter_y else 'X outliers removed'}) for {run_label}")

# ───────────────────────────────────────────────────────────────────────────────
# MAIN
# ───────────────────────────────────────────────────────────────────────────────
for subpath, label in RUNS:
    bed_fp = os.path.join(
        BASE_DIR, subpath,
        f"merged_{label}_terminal_telomeres_longest_pq.bed"
    )
    if not os.path.isfile(bed_fp):
        print(f"[WARN] Missing BED: {bed_fp}")
        continue

    cols = [
        "chr", "start", "end", "length", "label",
        "fwdCounts", "revCounts", "canCounts", "nonCanCounts",
        "chr_length", "assembly", "clade", "extended_lineage"
    ]
    df = pd.read_csv(bed_fp, sep="\t", header=None, names=cols)
    df["lineage"] = df["extended_lineage"]
    df = df[df["lineage"].isin(LINEAGE_COLORS.keys())].copy()

    df["chr_len_mb"] = df["chr_length"] / 1e6
    df["tel_len_kb"] = df["length"]     / 1e3

    # X-outlier filtered
    multipanel_scatter(df, label, log_x=False, filter_y=False)
    multipanel_scatter(df, label, log_x=True,  filter_y=False)
    # X+Y-outlier filtered
    multipanel_scatter(df, label, log_x=False, filter_y=True)
    multipanel_scatter(df, label, log_x=True,  filter_y=True)
