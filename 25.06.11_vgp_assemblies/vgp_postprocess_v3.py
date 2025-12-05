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
    # ("processed_NNNGGG_rgemi/primary", "NNNGGG"),
    # ("processed_TTAGGG_rgemi/primary", "TTAGGG"),
    ("processed_NNNGGG_ut100k/primary", "NNNGGG"),
    ("processed_TTAGGG_ut100k/primary", "TTAGGG"),
    # ("processed_NNNGGG_ut50k/primary", "NNNGGG"),
    # ("processed_TTAGGG_ut50k/primary", "TTAGGG"),
]

# --- NEW: load accession -> Extended_lineage map once ---
# Expect a CSV with columns: Accession, Clade, Extended_lineage
_lineage_df = pd.read_csv("./25.09.25_clade_lineage.csv", sep=",", dtype=str).fillna("NA")
# Support lookups by full accession (with version) and by core id (without version)
_acc2lineage = {a: e for a, e in zip(_lineage_df["Accession"], _lineage_df["Extended_lineage"])}
_core2lineage = {a.split(".")[0]: e for a, e in zip(_lineage_df["Accession"], _lineage_df["Extended_lineage"])}

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

for run_subpath, run_label in runs:
    run_dir = os.path.join(base_dir, run_subpath)
    merged_bed = os.path.join(run_dir, f"merged_{run_label}_terminal_telomeres.bed")
    merged_report = os.path.join(run_dir, f"merged_{run_label}_telomeres_report.csv")

    # Merge BED files for this run
    with open(merged_bed, 'w') as bed_out:
        for clade in os.listdir(run_dir):
            clade_dir = os.path.join(run_dir, clade)
            if not os.path.isdir(clade_dir):
                continue

            for entry in os.listdir(clade_dir):
                asm_dir = os.path.join(clade_dir, entry)
                # only process .teloscope directories
                if not os.path.isdir(asm_dir) or not entry.endswith('.teloscope'):
                    continue

                assembly_name = entry.replace('.teloscope', '')
                bed_file = os.path.join(asm_dir, "terminal_telomeres.bed")
                # NEW: resolve Extended_lineage for this assembly
                extended_lineage = _get_extended_lineage(assembly_name)

                if os.path.isfile(bed_file):
                    had_lines = False
                    with open(bed_file) as bf:
                        for line in bf:
                            had_lines = True
                            # NEW: append Extended_lineage as the last column
                            bed_out.write(f"{line.strip()}\t{assembly_name}\t{clade}\t{extended_lineage}\n")
                    if not had_lines:
                        print(f"[Warning] Empty BED for assembly {assembly_name} in run {run_label}")
                else:
                    print(f"[Warning] Missing BED: {bed_file}")

    # Merge report files for this run
    columns = [
        'Assembly', 'Clade', 'Extended_lineage', 'total_paths', 'total_gaps', 'total_telomeres',
        'mean_length', 'median_length', 'min_length', 'max_length',
        'two_telomeres', 'one_telomere', 'no_telomeres',
        't2t', 'gapped_t2t', 'missassembled', 'gapped_missassembled',
        'incomplete', 'gapped_incomplete', 'no_telomeres_comp', 'gapped_no_telomeres',
        'discordant', 'gapped_discordant'
    ]
    rows = []

    for clade in os.listdir(run_dir):
        clade_dir = os.path.join(run_dir, clade)
        if not os.path.isdir(clade_dir):
            continue

        for entry in os.listdir(clade_dir):
            asm_dir = os.path.join(clade_dir, entry)
            # only process .teloscope directories
            if not os.path.isdir(asm_dir) or not entry.endswith('.teloscope'):
                continue

            assembly_name = entry.replace('.teloscope', '')
            report_file = os.path.join(asm_dir, f"{assembly_name}.telo.report")
            if not os.path.isfile(report_file):
                print(f"[Warning] Missing report: {report_file}")
                continue

            # initialize data dictionary
            data = {col: '0' for col in columns}
            data['Assembly'] = assembly_name
            data['Clade'] = clade
            data['Extended_lineage'] = _get_extended_lineage(assembly_name)

            content = open(report_file).read()
            patterns = {
                'total_paths': r'Total paths:\s+(\d+)',
                'total_gaps': r'Total gaps:\s+(\d+)',
                'total_telomeres': r'Total telomeres:\s+(\d+)',
                'mean_length': r'Mean length:\s+([\d.]+)',
                'median_length': r'Median length:\s+(\d+)',
                'min_length': r'Min length:\s+(\d+)',
                'max_length': r'Max length:\s+(\d+)',
                'two_telomeres': r'Two telomeres:\s+(\d+)',
                'one_telomere': r'One telomere:\s+(\d+)',
                'no_telomeres': r'No telomeres:\s+(\d+)',
                't2t': r'T2T:\s+(\d+)',
                'gapped_t2t': r'Gapped T2T:\s+(\d+)',
                'missassembled': r'Missassembled:\s+(\d+)',
                'gapped_missassembled': r'Gapped missassembled:\s+(\d+)',
                'incomplete': r'Incomplete:\s+(\d+)',
                'gapped_incomplete': r'Gapped incomplete:\s+(\d+)',
                'no_telomeres_comp': r'No telomeres:\s+(\d+)(?!.*No telomeres:)',
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
        df.to_csv(merged_report, index=False)

    print(f"Run {run_label} merging complete:")
    print(f"  BED -> {merged_bed}")
    print(f"  CSV -> {merged_report}")
