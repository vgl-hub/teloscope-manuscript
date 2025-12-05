#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
telomere_busco_compleasm_gfastats_correlations.py (v8)
"""
from __future__ import annotations
import os
import re
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import to_rgb


# =========================
# ====== FILE PATHS =======
# =========================
TELOMERE_REPORT_CANDIDATES = [
    "./merged_TTAGGG_telomeres_report_ncbi_revised.xlsx",
]
BUSCO_XLSX      = "./busco.xlsx"
COMPLEASM_XLSX  = "./compleasm.xlsx"
CHR_COUNTS_XLSX = "./chr_counts.xlsx"
GFASTATS_TSV    = "./gfastats.tsv"
TECH_TXT        = "./tech.txt"                    # NEW
VGP_CSV         = "./vgp_v1.2_simplified.csv"     # NEW

PLOTS_DIR = "./plots_v8"
PLOTS_DIR_BUSCO = os.path.join(PLOTS_DIR, "busco")
PLOTS_DIR_COMPLEASM = os.path.join(PLOTS_DIR, "compleasm")
PLOTS_DIR_GFASTATS = os.path.join(PLOTS_DIR, "gfastats")
os.makedirs(PLOTS_DIR_BUSCO, exist_ok=True)
os.makedirs(PLOTS_DIR_COMPLEASM, exist_ok=True)
os.makedirs(PLOTS_DIR_GFASTATS, exist_ok=True)

BUSCO_COLS = [
    "Complete_BUSCO", "Single_BUSCO", "Duplicated_BUSCO",
    "Fragmented_BUSCO", "Missing_BUSCO", "STOP_BUSCO", "Frameshift_BUSCO",
]
COMPLEASM_COLS = [
    "Complete_Compleasm", "Single_Compleasm", "Duplicated_Compleasm",
    "Retrocopy_Compleasm", "Fragmented_Compleasm", "Missing_Compleasm",
    "Frameshift_Compleasm",
]


# =========================
# ====== UTILITIES  =======
# =========================
def load_telomere_report() -> pd.DataFrame:
    tel_path = next((p for p in TELOMERE_REPORT_CANDIDATES if os.path.exists(p)), None)
    if tel_path is None:
        raise FileNotFoundError(f"Telomere report not found. Tried: {', '.join(TELOMERE_REPORT_CANDIDATES)}")
    tel = pd.read_excel(tel_path)
    if "Accession" not in tel.columns and "Assembly" in tel.columns:
        tel = tel.rename(columns={"Assembly": "Accession"})
    if "Accession" not in tel.columns:
        raise KeyError("Telomere report must contain 'Assembly' or 'Accession'.")
    for c in ["total_telomeres", "two_telomeres", "one_telomere"]:
        if c not in tel.columns:
            raise KeyError(f"Telomere report missing column: {c}")
    for c in ["total_telomeres", "two_telomeres", "one_telomere"]:
        tel[c] = pd.to_numeric(tel[c], errors="coerce")
    return tel


def load_chr_counts() -> pd.DataFrame:
    if not os.path.exists(CHR_COUNTS_XLSX):
        raise FileNotFoundError(f"Missing chromosome counts file: {CHR_COUNTS_XLSX}")
    df = pd.read_excel(CHR_COUNTS_XLSX)
    if "Accession" not in df.columns and "Assembly" in df.columns:
        df = df.rename(columns={"Assembly": "Accession"})
    if "Accession" not in df.columns:
        raise KeyError("chr_counts.xlsx must contain 'Accession' or 'Assembly'.")
    chr_col = None
    for c in df.columns:
        lc = str(c).lower()
        if lc.startswith("chromosome") and "count" in lc:
            chr_col = c
            break
    if chr_col is None:
        raise KeyError("chr_counts.xlsx must contain a 'Chromosome count' column.")
    out = df[["Accession", chr_col]].rename(columns={chr_col: "n_chromosomes"})
    out["n_chromosomes"] = pd.to_numeric(out["n_chromosomes"], errors="coerce")
    return out


def load_busco() -> pd.DataFrame:
    if not os.path.exists(BUSCO_XLSX):
        raise FileNotFoundError(f"Missing BUSCO file: {BUSCO_XLSX}")
    df = pd.read_excel(BUSCO_XLSX)
    if "Accession" not in df.columns and "Assembly" in df.columns:
        df = df.rename(columns={"Assembly": "Accession"})
    if "Accession" not in df.columns:
        raise KeyError("busco.xlsx must contain 'Accession' or 'Assembly'.")
    for c in BUSCO_COLS:
        if c not in df.columns:
            raise KeyError(f"busco.xlsx missing expected column: {c}")
    cols = ["Accession"] + BUSCO_COLS
    df = df[cols].copy()
    for c in BUSCO_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_compleasm() -> pd.DataFrame:
    if not os.path.exists(COMPLEASM_XLSX):
        raise FileNotFoundError(f"Missing Compleasm file: {COMPLEASM_XLSX}")
    df = pd.read_excel(COMPLEASM_XLSX)
    if "Accession" not in df.columns and "Assembly" in df.columns:
        df = df.rename(columns={"Assembly": "Accession"})
    if "Accession" not in df.columns:
        raise KeyError("compleasm.xlsx must contain 'Accession' or 'Assembly'.")
    for c in COMPLEASM_COLS:
        if c not in df.columns:
            raise KeyError(f"compleasm.xlsx missing expected column: {c}")
    cols = ["Accession"] + COMPLEASM_COLS
    df = df[cols].copy()
    for c in COMPLEASM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_gfastats() -> pd.DataFrame:
    if not os.path.exists(GFASTATS_TSV):
        raise FileNotFoundError(f"Missing gfastats file: {GFASTATS_TSV}")
    df = pd.read_csv(GFASTATS_TSV, sep="\t")
    if "Accession" not in df.columns:
        raise KeyError("gfastats.tsv must contain 'Accession'.")
    cols = df.columns.tolist()
    if "# scaffolds" not in cols:
        raise KeyError("gfastats.tsv must contain '# scaffolds' to anchor metric columns.")
    start_idx = cols.index("# scaffolds")
    metric_cols = cols[start_idx:]
    for c in metric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_tech() -> pd.DataFrame:
    """Load tech.txt (tab-separated) with columns: Accession, Technology."""
    if not os.path.exists(TECH_TXT):
        raise FileNotFoundError(f"Missing tech file: {TECH_TXT}")
    df = pd.read_csv(TECH_TXT, sep="\t")
    if "Accession" not in df.columns or "Technology" not in df.columns:
        raise KeyError("tech.txt must contain 'Accession' and 'Technology'.")
    df["Technology"] = df["Technology"].astype(str).str.strip()
    return df


def load_vgp() -> pd.DataFrame:
    """Load vgp_v1.2_simplified.csv; extract Extended lineage and Assembly tech via UCSC Browser main haplotype."""
    if not os.path.exists(VGP_CSV):
        raise FileNotFoundError(f"Missing VGP file: {VGP_CSV}")
    df = pd.read_csv(VGP_CSV)
    needed = ["UCSC Browser main haplotype", "Extended lineage", "Assembly tech"]
    for c in needed:
        if c not in df.columns:
            raise KeyError(f"vgp_v1.2_simplified.csv missing column: {c}")
    df = df.rename(columns={"UCSC Browser main haplotype": "Accession"})
    # Trim "Assembly tech" before first comma
    df["Assembly tech"] = df["Assembly tech"].astype(str).str.split(",").str[0].str.strip()
    return df[["Accession", "Extended lineage", "Assembly tech"]].copy()


def safe_filename(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s)).strip("_")
    return s[:180]


def spearman(x: pd.Series, y: pd.Series) -> Tuple[float, float, int]:
    s = pd.concat([x, y], axis=1).dropna()
    if len(s) < 3:
        return np.nan, np.nan, len(s)
    try:
        rho, p = spearmanr(s.iloc[:, 0], s.iloc[:, 1])
    except Exception:
        rho, p = np.nan, np.nan
    return float(rho), float(p), int(len(s))


def pearson(x: pd.Series, y: pd.Series) -> Tuple[float, float, int]:
    s = pd.concat([x, y], axis=1).dropna()
    if len(s) < 3:
        return np.nan, np.nan, len(s)
    try:
        r, p = pearsonr(s.iloc[:, 0], s.iloc[:, 1])
    except Exception:
        r, p = np.nan, np.nan
    return float(r), float(p), int(len(s))


# --------- Plot styling helpers (≥12 colors, pastel FILL + normal EDGE) ----------
OKABE_ITO = ["#000000", "#E69F00", "#56B4E9", "#009E73",
             "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]

# An extended, high-contrast, colorblind-friendly-ish set (12+)
EXTENDED_CB = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442",
    "#0072B2", "#D55E00", "#CC79A7", "#999999",
    "#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3",
    "#A6D854", "#FFD92F", "#E5C494", "#B3B3B3"
]

def build_group_palette(groups: List[str]) -> Dict[str, str]:
    colors = (EXTENDED_CB + sns.color_palette("tab20", max(0, len(groups) - len(EXTENDED_CB))).as_hex())[:len(groups)]
    return {g: colors[i] for i, g in enumerate(sorted(groups))}

def pastelize(hex_color: str, factor: float = 0.25) -> tuple:
    """Slightly lighten color toward white (small factor => minimal difference)."""
    r, g, b = to_rgb(hex_color)
    return (r + (1 - r) * factor, g + (1 - g) * factor, b + (1 - b) * factor)

def pretty_tel_label(s: str) -> str:
    # Replace underscores with spaces and 'pct' with '%'
    return s.replace("_", " ").replace("pct", "%")


def scatter_all_by_group(
    df: pd.DataFrame,
    x: str,
    y: str,
    out_png: str,
    x_label: str,
    y_label: str,
    stats: Dict[str, float],
    group_col: str,
    palette: Dict[str, str],
    wide: bool = False,
) -> None:
    """All data in one axes, colored by a grouping column."""
    sns.set_style("white")
    plt.figure(figsize=((8.5 if wide else 6.5), 5.0), dpi=300)  # dpi here is overridden in savefig
    ax = plt.gca()
    ax.set_facecolor("white")
    ax.grid(False)

    if group_col in df.columns and df[group_col].notna().any():
        groups_in = [c for c in df[group_col].dropna().unique() if c in palette]
        for grp in sorted(groups_in):
            sub = df[df[group_col] == grp]
            edge = palette[grp]
            face = pastelize(edge, factor=0.25)   # pastel fill, minimal difference
            ax.scatter(
                sub[x], sub[y],
                s=32, linewidths=0.9,
                edgecolors=edge, facecolors=face,
                alpha=0.95, label=str(grp)
            )
        # compact legend
        ax.legend(loc="lower right", frameon=False, fontsize=8, title=group_col, title_fontsize=9, ncol=1)
    else:
        edge = "#4C78A8"
        face = pastelize(edge, 0.25)
        ax.scatter(df[x], df[y], s=32, linewidths=0.9, edgecolors=edge, facecolors=face, alpha=0.95)

    # Trendline (OLS)
    try:
        s = df[[x, y]].dropna()
        if len(s) >= 3:
            xv = s[x].to_numpy(dtype=float)
            yv = s[y].to_numpy(dtype=float)
            if np.isfinite(xv).all() and np.isfinite(yv).all() and np.std(xv) > 0:
                coef = np.polyfit(xv, yv, 1)
                xx = np.linspace(xv.min(), xv.max(), 200)
                yy = coef[0] * xx + coef[1]
                ax.plot(xx, yy, color="#333333", linewidth=1.2, alpha=0.8)
    except Exception:
        pass

    # Stats box
    txt = (
        f"Spearman ρ={stats['spearman_rho']:.2f}\n"
        f"Spearman p_adj={stats['spearman_p_adj']:.2e}\n"
        f"Pearson r={stats['pearson_r']:.2f}\n"
        f"Pearson p_adj={stats['pearson_p_adj']:.2e}\n"
        f"N={int(stats['n'])}"
    )
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left", fontsize=8)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=600)
    plt.close()


def facet_by_group_4x3(
    df: pd.DataFrame,
    x: str,
    y: str,
    out_png: str,
    x_label: str,
    y_label: str,
    group_col: str,
    palette: Dict[str, str],
) -> None:
    """4×3 small-multiples, one panel per group (up to 12)."""
    sns.set_style("white")
    groups = sorted([g for g in df[group_col].dropna().unique()] if group_col in df.columns else [])
    # Fix order to first 12 (or pad blanks)
    n_panels = 12
    grids = groups[:n_panels] + [""] * max(0, n_panels - len(groups[:n_panels]))

    # Shared limits for consistency
    dff = df[[x, y]].dropna()
    if len(dff) == 0:
        return
    x_min, x_max = np.nanmin(dff[x]), np.nanmax(dff[x])
    y_min, y_max = np.nanmin(dff[y]), np.nanmax(dff[y])

    fig, axes = plt.subplots(3, 4, figsize=(11.5, 8.5), dpi=300)
    axes = axes.flatten()

    for ax, grp in zip(axes, grids):
        ax.set_facecolor("white")
        ax.grid(False)
        if grp == "":
            ax.axis("off")
            continue
        sub = df[df[group_col] == grp]
        if len(sub.dropna(subset=[x, y])) == 0:
            ax.axis("off")
            continue
        edge = palette.get(grp, "#4C78A8")
        face = pastelize(edge, 0.25)
        ax.scatter(sub[x], sub[y], s=16, linewidths=0.8, edgecolors=edge, facecolors=face, alpha=0.95)
        ax.set_title(str(grp), fontsize=8)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.tick_params(labelsize=7)
    # One shared set of labels on outer figure
    fig.supxlabel(x_label, fontsize=10)
    fig.supylabel(y_label, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_png, dpi=600)
    plt.close()


def p_adjust_bh(pvals: pd.Series) -> pd.Series:
    p = pd.Series(pvals, dtype=float)
    mask = p.notna()
    x = p[mask].values
    n = len(x)
    if n == 0:
        return p
    order = np.argsort(x)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    adj = x * n / ranks
    adj_sorted = adj[order]
    adj_sorted = np.minimum.accumulate(adj_sorted[::-1])[::-1]
    adj_final = np.minimum(adj_sorted, 1.0)
    out = p.copy()
    out.loc[mask] = adj_final[np.argsort(order)]
    return out


# ---- accession parsing and core-ID helpers ----
ACCESSION_RE = re.compile(r'^(GC[AF])_(\d+)(?:\.(\d+))?$')

def extract_core_id(acc: str) -> str | None:
    if pd.isna(acc):
        return None
    m = ACCESSION_RE.match(str(acc).strip())
    return m.group(2) if m else None

def extract_version(acc: str) -> int:
    if pd.isna(acc):
        return -1
    m = ACCESSION_RE.match(str(acc).strip())
    return int(m.group(3)) if (m and m.group(3) is not None) else -1

def add_core_version(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["core_id"] = df["Accession"].map(extract_core_id)
    df["version_num"] = df["Accession"].map(extract_version)
    return df

def dedup_keep_latest(df: pd.DataFrame, by_cols: List[str]) -> pd.DataFrame:
    if "core_id" not in df.columns:
        return df
    df = df.sort_values(["core_id", "version_num"], ascending=[True, False])
    return df.drop_duplicates(subset=["core_id"], keep="first")


# =========================
# ========= MAIN ==========
# =========================
def main():
    # 1) Load base tables
    tel = load_telomere_report()
    chrdf = load_chr_counts()
    busco = load_busco()
    comp = load_compleasm()
    gfa = load_gfastats()
    tech = load_tech()     # NEW
    vgp  = load_vgp()      # NEW

    # Add core/version columns
    tel   = add_core_version(tel)
    chrdf = add_core_version(chrdf)
    busco = add_core_version(busco)
    comp  = add_core_version(comp)
    gfa   = add_core_version(gfa)
    tech  = add_core_version(tech)     # use Accession -> core_id
    vgp   = add_core_version(vgp)      # use UCSC haplotype -> core_id

    # Warn unparsable
    for name, df in [("telomere report", tel), ("chr_counts", chrdf), ("busco", busco),
                     ("compleasm", comp), ("gfastats", gfa), ("tech", tech), ("vgp", vgp)]:
        unparsable = df[df["core_id"].isna()]["Accession"].dropna().unique().tolist()
        if len(unparsable) > 0:
            print(f"[WARN] Unparsable accessions in {name}: {len(unparsable)}")
            for a in unparsable:
                print("   -", a)

    # Build percentage telomere variables
    tel["with_telomeres"] = tel["two_telomeres"].fillna(0) + tel["one_telomere"].fillna(0)
    chrdf_latest = dedup_keep_latest(chrdf, ["Accession", "n_chromosomes", "core_id", "version_num"])
    tel = tel.merge(chrdf_latest[["core_id", "n_chromosomes"]], on="core_id", how="left")
    tel["n_chromosomes"] = pd.to_numeric(tel["n_chromosomes"], errors="coerce")
    den = tel["n_chromosomes"]
    tel["two_telomeres_pct"]           = np.where(den > 0, (tel["two_telomeres"] / den) * 100.0, np.nan)
    tel["one_telomere_pct"]            = np.where(den > 0, (tel["one_telomere"] / den) * 100.0, np.nan)
    tel["with_telomeres_pct"]          = np.where(den > 0, (tel["with_telomeres"] / den) * 100.0, np.nan)
    tel["total_telomeres_pct_of_ends"] = np.where(den > 0, (tel["total_telomeres"] / (2.0 * den)) * 100.0, np.nan)

    # Dedup metrics by latest version per core_id
    busco_latest = dedup_keep_latest(busco, ["core_id", "version_num"] + BUSCO_COLS)
    comp_latest  = dedup_keep_latest(comp,  ["core_id", "version_num"] + COMPLEASM_COLS)
    gfa_cols = gfa.columns.tolist()
    start_idx = gfa_cols.index("# scaffolds")
    GFASTATS_METRIC_COLS = [c for c in gfa_cols[start_idx:] if c not in ("core_id", "version_num")]
    gfa_latest = dedup_keep_latest(gfa, ["core_id", "version_num"] + GFASTATS_METRIC_COLS)
    gfa_latest = gfa_latest.loc[:, ~gfa_latest.columns.duplicated()]

    # Merge everything into ONE dataframe (keep Clade, add lineage & tech)
    merged_df = tel.merge(busco_latest[["core_id"] + BUSCO_COLS], on="core_id", how="left") \
                   .merge(comp_latest[["core_id"] + COMPLEASM_COLS], on="core_id", how="left") \
                   .merge(gfa_latest[["core_id"] + GFASTATS_METRIC_COLS], on="core_id", how="left") \
                   .merge(tech[["core_id", "Technology"]], on="core_id", how="left") \
                   .merge(vgp[["core_id", "Extended lineage", "Assembly tech"]], on="core_id", how="left")

    # Log10(x+1) transforms
    for m in BUSCO_COLS:
        merged_df[f"{m}_log10p1"] = np.log10(merged_df[m].clip(lower=0) + 1.0)
    for m in COMPLEASM_COLS:
        merged_df[f"{m}_log10p1"] = np.log10(merged_df[m].clip(lower=0) + 1.0)
    for m in GFASTATS_METRIC_COLS:
        merged_df[f"{m}_log10p1"] = np.log10(merged_df[m].clip(lower=0) + 1.0)

    # Info on missing matches
    for name, ref_latest in [("busco", busco_latest), ("compleasm", comp_latest), ("gfastats", gfa_latest)]:
        missing = sorted(set(tel["core_id"].dropna()) - set(ref_latest["core_id"].dropna()))
        if len(missing) > 0:
            print(f"[INFO] Assemblies missing in {name} by core_id: {len(missing)}")

    # Save the single merged dataframe
    merged_df.to_csv("./merged_telomere_assembly_df.csv", index=False)

    # Telomere variables — ONLY percentages
    tel_vars = ["two_telomeres_pct", "one_telomere_pct", "with_telomeres_pct", "total_telomeres_pct_of_ends"]

    # ============ correlations_all ============
    corr_rows: List[dict] = []

    def add_corr_rows(df_in: pd.DataFrame, tv: str, metric_col: str, metric_transform: str):
        cols = [tv, metric_col, "Clade"]
        # If some columns are missing (unlikely), skip
        if not all(c in df_in.columns for c in cols):
            return
        df0 = df_in[cols].rename(columns={tv: "telomere_var", metric_col: "metric_val"})
        df0[["telomere_var", "metric_val"]] = df0[["telomere_var", "metric_val"]].replace([np.inf, -np.inf], np.nan)
        df0 = df0.dropna(subset=["telomere_var", "metric_val"])
        sr, sp, n_s = spearman(df0["metric_val"], df0["telomere_var"])
        pr, pp, n_p = pearson(df0["metric_val"], df0["telomere_var"])
        corr_rows.append({
            "clade": "all",
            "telomere_variable": tv,
            "metric": metric_col.replace("_log10p1", ""),
            "metric_transform": metric_transform,
            "spearman_rho": sr, "spearman_p": sp,
            "pearson_r": pr, "pearson_p": pp,
            "n": int(min(n_s, n_p))
        })
        if "Clade" in df0.columns and df0["Clade"].notna().any():
            for cl in sorted(df0["Clade"].dropna().unique()):
                sub = df0[df0["Clade"] == cl]
                sr, sp, n_s = spearman(sub["metric_val"], sub["telomere_var"])
                pr, pp, n_p = pearson(sub["metric_val"], sub["telomere_var"])
                corr_rows.append({
                    "clade": str(cl),
                    "telomere_variable": tv,
                    "metric": metric_col.replace("_log10p1", ""),
                    "metric_transform": metric_transform,
                    "spearman_rho": sr, "spearman_p": sp,
                    "pearson_r": pr, "pearson_p": pp,
                    "n": int(min(n_s, n_p))
                })

    # BUSCO (log + raw)
    for tv in tel_vars:
        for metric in BUSCO_COLS:
            add_corr_rows(merged_df, tv, f"{metric}_log10p1", "log10(x+1)")
    for tv in tel_vars:
        for metric in BUSCO_COLS:
            add_corr_rows(merged_df, tv, metric, "x")

    # Compleasm (log + raw)
    for tv in tel_vars:
        for metric in COMPLEASM_COLS:
            add_corr_rows(merged_df, tv, f"{metric}_log10p1", "log10(x+1)")
    for tv in tel_vars:
        for metric in COMPLEASM_COLS:
            add_corr_rows(merged_df, tv, metric, "x")

    # gfastats (log + raw)
    for tv in tel_vars:
        for metric in GFASTATS_METRIC_COLS:
            add_corr_rows(merged_df, tv, f"{metric}_log10p1", "log10(x+1)")
            add_corr_rows(merged_df, tv, metric, "x")

    # Finalize correlations_all
    all_corr = pd.DataFrame(corr_rows)
    all_corr["spearman_p_adj"] = p_adjust_bh(all_corr["spearman_p"])
    all_corr["pearson_p_adj"]  = p_adjust_bh(all_corr["pearson_p"])
    col_order = [
        "clade", "telomere_variable", "metric", "metric_transform",
        "spearman_rho", "spearman_p", "spearman_p_adj",
        "pearson_r", "pearson_p", "pearson_p_adj",
        "n"
    ]
    all_corr = all_corr[col_order]
    all_corr.to_csv("./correlations_all.csv", index=False)

    # -------- generate plots (ONLY clade == 'all') --------
    # Build palettes for lineage & technology
    lineages_all = sorted(merged_df["Extended lineage"].dropna().unique().tolist()) if "Extended lineage" in merged_df.columns else []
    LINEAGE_PALETTE = build_group_palette(lineages_all)

    tech_all = sorted(merged_df["Technology"].dropna().unique().tolist()) if "Technology" in merged_df.columns else []
    TECH_PALETTE = build_group_palette(tech_all) if len(tech_all) > 0 else {}

    def plot_set(metric: str, transform: str, tv: str, stats_row) -> None:
        # Decide base folder by metric family
        if metric in BUSCO_COLS:
            base_dir = PLOTS_DIR_BUSCO
        elif metric in COMPLEASM_COLS:
            base_dir = PLOTS_DIR_COMPLEASM
        else:
            base_dir = PLOTS_DIR_GFASTATS

        metric_dir = os.path.join(base_dir, safe_filename(metric))
        os.makedirs(metric_dir, exist_ok=True)

        xcol = f"{metric}_log10p1" if transform == "log10(x+1)" else metric
        if xcol not in merged_df.columns or tv not in merged_df.columns:
            return

        dfp = merged_df[[xcol, tv, "Extended lineage", "Technology"]].copy()
        dfp = dfp.dropna(subset=[xcol, tv])
        if len(dfp) <= 3:
            return

        # Labels
        x_label = rf"$\log_{{10}}$ {metric}" if transform == "log10(x+1)" else f"{metric}"
        y_label = pretty_tel_label(tv)

        stats = {
            "spearman_rho": stats_row.spearman_rho,
            "spearman_p_adj": stats_row.spearman_p_adj,
            "pearson_r": stats_row.pearson_r,
            "pearson_p_adj": stats_row.pearson_p_adj,
            "n": int(stats_row.n),
        }

        # 1) All data colored by Extended lineage (wider figure)
        out1 = os.path.join(metric_dir, safe_filename(f"ALL__{metric}__{transform}__vs__{tv}__by_lineage.png"))
        scatter_all_by_group(
            dfp.rename(columns={xcol: "X", tv: "Y"}),
            x="X", y="Y", out_png=out1,
            x_label=x_label, y_label=y_label,
            stats=stats, group_col="Extended lineage",
            palette=LINEAGE_PALETTE, wide=True
        )

        # 2) Multipanel 4×3 by lineage
        out2 = os.path.join(metric_dir, safe_filename(f"FACET__{metric}__{transform}__vs__{tv}__lineage_4x3.png"))
        facet_by_group_4x3(
            dfp.rename(columns={xcol: "X", tv: "Y"}),
            x="X", y="Y", out_png=out2,
            x_label=x_label, y_label=y_label,
            group_col="Extended lineage", palette=LINEAGE_PALETTE
        )

        # 3) All data colored by Technology
        if "Technology" in dfp.columns and dfp["Technology"].notna().any():
            out3 = os.path.join(metric_dir, safe_filename(f"ALL__{metric}__{transform}__vs__{tv}__by_technology.png"))
            scatter_all_by_group(
                dfp.rename(columns={xcol: "X", tv: "Y"}),
                x="X", y="Y", out_png=out3,
                x_label=x_label, y_label=y_label,
                stats=stats, group_col="Technology",
                palette=TECH_PALETTE if len(TECH_PALETTE) > 0 else build_group_palette(sorted(dfp["Technology"].dropna().unique().tolist())),
                wide=True
            )

    # Loop over correlations rows, but ONLY clade == 'all'
    for row in all_corr.itertuples(index=False):
        if (row.n is None) or (int(row.n) <= 3):
            continue
        if row.clade != "all":
            continue
        plot_set(metric=row.metric, transform=row.metric_transform, tv=row.telomere_variable, stats_row=row)

    print("Done.")
    print(f"Merged rows in merged_df: {len(merged_df)}")
    print(f"Plots: {PLOTS_DIR_BUSCO} (BUSCO), {PLOTS_DIR_COMPLEASM} (Compleasm), {PLOTS_DIR_GFASTATS} (gfastats)")
    print("Outputs:")
    print("  - ./merged_telomere_assembly_df.csv")
    print("  - ./correlations_all.csv")


if __name__ == "__main__":
    main()
