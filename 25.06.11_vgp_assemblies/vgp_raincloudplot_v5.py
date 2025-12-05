#!/usr/bin/env python3
"""
Raincloud-style violin–box–scatter plot (v4) with bifurcated heatmap
for VGP telomere lengths per clade.

• Raincloud plot: violin-box-scatter (pastel colors, no edge scatter)
• Heatmap subplot: 
    – upper triangle: –log10(Bonferroni p-adj) (sequential Blues)
    – lower triangle: Δ mean telomere length (Kb) (diverging RdBu_r)
• No significance bars on the raincloud
• One panel per run (NNNGGG, TTAGGG) – no title, 15×6" layout
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde, mannwhitneyu
from itertools import combinations

# ────────────────────────────────────────────────────────────────────────────────
# PATH CONFIG
# ────────────────────────────────────────────────────────────────────────────────
base_dir = "/mnt/d/research/teloscope_article/25.06.11_vgp_assemblies"
runs     = [
    ("processed_NNNGGG_ut50k/primary", "NNNGGG"),
    ("processed_TTAGGG_ut50k/primary", "TTAGGG"),
]
plot_dir = os.path.join(base_dir, "plots_ut50k")
os.makedirs(plot_dir, exist_ok=True)

# ────────────────────────────────────────────────────────────────────────────────
# STYLE CONFIG
# ────────────────────────────────────────────────────────────────────────────────
display_names = {"otherChordates": "other chord."}
clade_order   = [
    "invertebrate", "echinoderm", "other chord.", "fish", "sharks",
    "amphibians",   "reptiles",   "birds",        "mammals", "primates"
]
pastel_palette = sns.color_palette("pastel", n_colors=len(clade_order))
clade_colors   = dict(zip(clade_order, pastel_palette))
sns.set_style("whitegrid")

# ────────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ────────────────────────────────────────────────────────────────────────────────
for run_subpath, run_label in runs:
    bed_file = os.path.join(
        base_dir, run_subpath,
        f"merged_{run_label}_terminal_telomeres_longest_pq.bed"
    )
    if not os.path.isfile(bed_file):
        print(f"[WARN] Missing BED: {bed_file}")
        continue

    # -- Load and preprocess
    df = pd.read_csv(bed_file, sep="\t", header=None, names=[
        "chr", "start", "end", "length", "label",
        "fwdCounts", "revCounts", "canCounts", "nonCanCounts", "chr_length",
        "assembly", "clade"
    ])
    df["clade"]   = df["clade"].replace(display_names)
    df            = df[df["clade"].isin(clade_order)].copy()
    df["lengthKb"]= df["length"] / 1e3

    q1, q3 = df["lengthKb"].quantile([0.25, 0.75])
    iqr     = q3 - q1
    lb, ub  = q1 - 1.5*iqr, q3 + 1.5*iqr

    # -- Set up figure: raincloud + heatmap
    fig, (ax_rain, ax_heatmap) = plt.subplots(
        ncols=2, figsize=(15, 5),
        gridspec_kw={'width_ratios': [3, 2]}
    )
    positions = np.arange(len(clade_order))

    # -- Raincloud plot
    for i, clade in enumerate(clade_order):
        sub   = df[df["clade"] == clade]
        clean = sub[(sub["lengthKb"]>=lb)&(sub["lengthKb"]<=ub)]["lengthKb"]

        if len(clean) > 1:
            kde    = gaussian_kde(clean, bw_method=0.5)
            y_vals = np.linspace(clean.min(), clean.max(), 200)
            dens   = 0.4 * kde(y_vals) / kde(y_vals).max()
            ax_rain.fill_betweenx(y_vals,
                                  positions[i],
                                  positions[i] + dens,
                                  color=clade_colors[clade],
                                  alpha=0.6, linewidth=0, zorder=1)
            ax_rain.plot(positions[i] + dens,
                         y_vals,
                         color=clade_colors[clade],
                         linewidth=1, zorder=2)

        sns.boxplot(y=clean, orient="v", width=0.25,
                    showcaps=True, boxprops={"facecolor":clade_colors[clade]},
                    whiskerprops={"color":"black"},
                    capprops={"color":"black"},
                    medianprops={"color":"black"},
                    showfliers=False,
                    positions=[positions[i]],
                    ax=ax_rain, zorder=3)

        jitter = np.random.normal(positions[i]-0.28, 0.035, size=len(sub))
        ax_rain.scatter(jitter, sub["lengthKb"],
                        color=clade_colors[clade],
                        s=20, alpha=0.7, edgecolor="none", zorder=4)

    ax_rain.set_xticks(positions)
    ax_rain.set_xticklabels(clade_order, rotation=45,
                            ha="right", fontsize=9)
    ax_rain.set_ylabel("Telomere length (Kb)")
    ax_rain.set_xlabel("")
    sns.despine(ax=ax_rain, left=True, top=True, right=True)
    ax_rain.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax_rain.xaxis.grid(False)

    # -- Compute matrices for heatmap
    n       = len(clade_order)
    pvals   = np.ones((n,n))
    means   = np.zeros(n)
    for idx, clade in enumerate(clade_order):
        means[idx] = df[df["clade"]==clade]["lengthKb"].mean()

    pairs  = list(combinations(range(n),2))
    n_tests= len(pairs)

    for i1,i2 in pairs:
        v1 = df[df["clade"]==clade_order[i1]]["lengthKb"]
        v2 = df[df["clade"]==clade_order[i2]]["lengthKb"]
        _, p = mannwhitneyu(v1, v2, alternative="two-sided")
        p_adj = min(p * n_tests, 1.0)
        pvals[i1,i2] = pvals[i2,i1] = p_adj

    diff_means = means[:,None] - means[None,:]

    # -- Bifurcated heatmap on same axis
    mask_lower = np.tril(np.ones_like(pvals, dtype=bool))
    sns.heatmap(
        -np.log10(pvals),
        mask=mask_lower,
        cmap="Greens",
        vmin=0, vmax=25,
        xticklabels=clade_order,
        yticklabels=clade_order,
        cbar_kws={'label':'-log10(p-adj)'},
        square=True,
        ax=ax_heatmap
    )

    mask_upper = np.triu(np.ones_like(diff_means, dtype=bool))
    sns.heatmap(
        diff_means,
        mask=mask_upper,
        cmap="coolwarm",
        center=0,
        xticklabels=clade_order,
        yticklabels=clade_order,
        cbar_kws={'label': 'Δ mean (Kb)'},
        square=True,
        ax=ax_heatmap,
        alpha=0.8
    )

    ax_heatmap.set_xticklabels(clade_order, rotation=45,
                               ha="right", fontsize=8)
    ax_heatmap.set_yticklabels(clade_order, rotation=0,
                               fontsize=8)
    ax_heatmap.grid(False)

    plt.tight_layout()


    # -- Save
    for ext, dpi in [("pdf",None), ("png",600), ("svg",None)]:
        out_fp = os.path.join(
            plot_dir, f"vgp_telomere_raincloud_v5_{run_label}.{ext}"
        )
        fig.savefig(out_fp, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {out_fp}")
