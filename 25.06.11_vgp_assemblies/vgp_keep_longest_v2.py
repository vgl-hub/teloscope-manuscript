"""
vgp_keep_longest_v2.py
"""

import os
import pandas as pd

# Base directory containing processed runs
base_dir = "/mnt/d/research/teloscope_article/25.06.11_vgp_assemblies"

# Runs definition
runs = [
    # ("processed_NNNGGG_rgemi/primary", "NNNGGG"),
    # ("processed_TTAGGG_rgemi/primary", "TTAGGG"),
    ("processed_NNNGGG_ut100k/primary", "NNNGGG"),
    ("processed_TTAGGG_ut100k/primary", "TTAGGG"),
    # ("processed_NNNGGG_ut50k/primary", "NNNGGG"),
    # ("processed_TTAGGG_ut50k/primary", "TTAGGG"),
]

for run_subpath, run_label in runs:
    run_dir = os.path.join(base_dir, run_subpath)
    merged_bed = os.path.join(run_dir, f"merged_{run_label}_terminal_telomeres.bed")
    output_bed = os.path.join(run_dir, f"merged_{run_label}_terminal_telomeres_longest_pq.bed")

    # Read merged BED
    df = pd.read_csv(merged_bed, sep='\t', header=None)

    # Define columns explicitly for clarity
    df.columns = [
        'chr_id', 'start', 'end', 'length', 'label', 'fwdCounts', 'revCounts', 'canCounts',
        'nonCanCounts', 'chr_len', 'assembly', 'clade', 'extended_lineage'
    ]

    # Keep the longest telomeres per chromosome for 'p' and 'q'
    longest_rows = []
    for chr_id, group_df in df.groupby('chr_id'):
        for arm in ['p', 'q']:
            arm_df = group_df[group_df['label'] == arm]
            if not arm_df.empty:
                longest_row = arm_df.loc[arm_df['length'].idxmax()]
                longest_rows.append(longest_row)

    # Create DataFrame for the longest telomeres
    longest_df = pd.DataFrame(longest_rows)

    # Write the result to BED file
    longest_df.to_csv(output_bed, sep='\t', header=False, index=False)

    print(f"Longest telomeres extracted for run {run_label} at: {output_bed}")
