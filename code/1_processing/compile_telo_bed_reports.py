"""
vgp_postprocess_v3.py
"""

import os
import pandas as pd
import re

# Base directory containing processed runs
base_dir = "/mnt/d/research/teloscope_article/25.06.11_vgp_assemblies"
# Tuple list of (run subpath, run label)
runs = [
    ("25.12.03_processed_TTAGGG/primary", "TTAGGG"),
]

# --- NEW: load accession -> Extended_lineage map once ---
# Expect a CSV with columns: Accession, Clade, Extended_lineage
_lineage_df = pd.read_csv("./25.09.25_clade_lineage.csv", sep=",", dtype=str).fillna("NA")
# Support lookups by full accession (with version) and by core id (without version)
_acc2lineage = {a: e for a, e in zip(_lineage_df["Accession"], _lineage_df["Extended_lineage"])}
_core2lineage = {a.split(".")[0]: e for a, e in zip(_lineage_df["Accession"], _lineage_df["Extended_lineage"])}
_lineage_total = len(_lineage_df)

_acc_patterns = re.compile(r"(GC[AF]_\d+(?:\.\d+)?)")

def _get_extended_lineage(assembly_name: str) -> str:
    """
    Try to resolve Extended_lineage for an assembly by extracting the GCA/GCF accession
    (with or without version) from its directory name.
    """
    m = _acc_patterns.search(assembly_name)
    if m:
        acc = m.group(1)
        if acc in _acc2lineage:
            return _acc2lineage[acc]
        core = acc.split(".")[0]
        if core in _core2lineage:
            return _core2lineage[core]
    # Fallback: if assembly_name itself is an accession key
    if assembly_name in _acc2lineage:
        return _acc2lineage[assembly_name]
    core = assembly_name.split(".")[0]
    if core in _core2lineage:
        return _core2lineage[core]
    return "NA"


def _write_compiled_bed(assemblies, bed_name, alt_bed_name, output_path, label):
    """
    Merge telomere BEDs while tracking which assemblies were written, missing, or empty.
    """
    written = 0
    missing = []
    empty = []

    with open(output_path, 'w') as bed_out:
        for assembly_name, clade, asm_dir, extended_lineage in assemblies:
            bed_file = os.path.join(asm_dir, bed_name)
            if not os.path.isfile(bed_file) and alt_bed_name:
                bed_file = os.path.join(asm_dir, alt_bed_name.format(asm=assembly_name))
            if not os.path.isfile(bed_file):
                missing.append(assembly_name)
                continue

            had_lines = False
            with open(bed_file) as bf:
                for line in bf:
                    had_lines = True
                    bed_out.write(f"{line.strip()}\t{assembly_name}\t{clade}\t{extended_lineage}\n")

            if had_lines:
                written += 1
            else:
                empty.append(assembly_name)

    print(
        f"  {label}: {written}/{len(assemblies)} assemblies written; "
        f"missing {len(missing)}, empty {len(empty)}"
    )
    if missing:
        print(f"    Missing ({len(missing)}): {', '.join(sorted(missing))}")
    if empty:
        print(f"    Empty ({len(empty)}): {', '.join(sorted(empty))}")

for run_subpath, run_label in runs:
    run_dir = os.path.join(base_dir, run_subpath)
    terminal_bed = os.path.join(run_dir, f"compiled_{run_label}_terminal_telomeres.bed")
    interstitial_bed = os.path.join(run_dir, f"compiled_{run_label}_interstitial_telomeres.bed")
    compiled_report = os.path.join(run_dir, f"compiled_{run_label}_telomeres_report.csv")

    assemblies = []
    for clade in os.listdir(run_dir):
        clade_dir = os.path.join(run_dir, clade)
        if not os.path.isdir(clade_dir):
            continue

        for entry in os.listdir(clade_dir):
            asm_dir = os.path.join(clade_dir, entry)
            if not os.path.isdir(asm_dir) or not entry.endswith('.teloscope'):
                continue

            assembly_name = entry.replace('.teloscope', '')
            assemblies.append((assembly_name, clade, asm_dir, _get_extended_lineage(assembly_name)))

    print(
        f"Run {run_label}: discovered {len(assemblies)} assemblies in {run_dir} "
        f"(lineage entries available: {_lineage_total})"
    )

    _write_compiled_bed(
        assemblies,
        "terminal_telomeres.bed",
        "{asm}.chr.fa_terminal_telomeres.bed",
        terminal_bed,
        "Terminal BED",
    )
    _write_compiled_bed(
        assemblies,
        "interstitial_telomeres.bed",
        "{asm}.chr.fa_interstitial_telomeres.bed",
        interstitial_bed,
        "Interstitial BED",
    )

    # Merge report files for this run
    columns = [
        'Assembly', 'Clade', 'Extended_lineage', 'total_paths', 'total_gaps', 'total_telomeres',
        'total_its_blocks', 'total_canonical_matches', 'total_windows_analyzed',
        'mean_length', 'median_length', 'min_length', 'max_length',
        'two_telomeres', 'one_telomere', 'zero_telomeres',
        't2t', 'gapped_t2t', 'missassembled', 'gapped_missassembled',
        'incomplete', 'gapped_incomplete', 'no_telomeres', 'gapped_no_telomeres',
        'discordant', 'gapped_discordant'
    ]
    rows = []
    missing_reports = []

    for assembly_name, clade, asm_dir, extended_lineage in assemblies:
        report_file = os.path.join(asm_dir, f"{assembly_name}.telo.report")
        if not os.path.isfile(report_file):
            missing_reports.append(assembly_name)
            continue

        data = {col: '0' for col in columns}
        data['Assembly'] = assembly_name
        data['Clade'] = clade
        data['Extended_lineage'] = extended_lineage

        content = open(report_file).read()
        patterns = {
            'total_paths': r'Total paths:\s+(\d+)',
            'total_gaps': r'Total gaps:\s+(\d+)',
            'total_telomeres': r'Total telomeres:\s+(\d+)',
            'total_its_blocks': r'Total ITS blocks:\s+(\d+)',
            'total_canonical_matches': r'Total canonical matches:\s+(\d+)',
            'total_windows_analyzed': r'Total windows analyzed:\s+(\d+)',
            'mean_length': r'Mean length:\s+([\d.]+)',
            'median_length': r'Median length:\s+(\d+)',
            'min_length': r'Min length:\s+(\d+)',
            'max_length': r'Max length:\s+(\d+)',
            'two_telomeres': r'Two telomeres:\s+(\d+)',
            'one_telomere': r'One telomere:\s+(\d+)',
            'zero_telomeres': r'No telomeres:\s+(\d+)',
            't2t': r'T2T:\s+(\d+)',
            'gapped_t2t': r'Gapped T2T:\s+(\d+)',
            'missassembled': r'Missassembled:\s+(\d+)',
            'gapped_missassembled': r'Gapped missassembled:\s+(\d+)',
            'incomplete': r'Incomplete:\s+(\d+)',
            'gapped_incomplete': r'Gapped incomplete:\s+(\d+)',
            'no_telomeres': r'No telomeres:\s+(\d+)',
            'gapped_no_telomeres': r'Gapped no telomeres:\s+(\d+)',
            'discordant': r'Discordant:\s+(\d+)',
            'gapped_discordant': r'Gapped discordant:\s+(\d+)'
        }

        for key, patt in patterns.items():
            m = re.search(patt, content)
            if m:
                data[key] = m.group(1)
        rows.append(data)

    # Write consolidated report
    df = pd.DataFrame(rows, columns=columns)
    if df.empty:
        print(f"[Warning] No report data found for run {run_label}")
    else:
        df.to_csv(compiled_report, index=False)

    print(
        f"  Reports: {len(rows)}/{len(assemblies)} assemblies written; "
        f"missing reports {len(missing_reports)}"
    )
    if missing_reports:
        print(f"    Missing reports ({len(missing_reports)}): {', '.join(sorted(missing_reports))}")

    print(f"Run {run_label} merging complete:")
    print(f"  Terminal BED -> {terminal_bed}")
    print(f"  Interstitial BED -> {interstitial_bed}")
    print(f"  CSV -> {compiled_report}")
