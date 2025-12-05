#!/usr/bin/env python3

import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Base paths
data_dir = "."
plot_dir = os.path.join(data_dir, "plots_rgemi")
os.makedirs(plot_dir, exist_ok=True)

# Global plot style
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 12,
    'axes.linewidth': 1.0,
    'figure.dpi': 600,
    'savefig.dpi': 600
})

# Run directories (input_path, run_label)
runs = [
    ("processed_NNNGGG_rgemi/primary", "NNNGGG"),
    ("processed_TTAGGG_rgemi/primary",  "TTAGGG"),
]

# Composition columns (rename errors → discordant)
cols = [
    't2t', 'gapped_t2t',
    'incomplete', 'gapped_incomplete',
    'missassembled', 'gapped_missassembled',
    'discordant', 'gapped_discordant',
    'no_telomeres_comp', 'gapped_no_telomeres'
]
colors = {
    't2t':                 '#1a9641',
    'gapped_t2t':          '#9CCF60',
    'incomplete':          '#FFC754',
    'gapped_incomplete':   '#FFE885',
    'missassembled':       '#D6594C',
    'gapped_missassembled':'#F58B6D',
    'discordant':          '#8278F4',
    'gapped_discordant':   '#B395EB',
    'no_telomeres_comp':   '#C8C8C8',
    'gapped_no_telomeres': '#F0F0F0'
}

# Clade ordering and display-name overrides
clade_order = [
    'invertebrate', 'echinoderm', 'otherChordates', 'fish', 'sharks',
    'amphibians', 'reptiles', 'birds', 'mammals', 'primates'
]
display_names = {
    'otherChordates': 'other chordates'
}

for run_path, run_label in runs:
    input_dir   = os.path.join(data_dir, run_path)
    report_file = f"merged_{run_label}_telomeres_report.csv"
    report_path = os.path.join(input_dir, report_file)

    if not os.path.isfile(report_path):
        print(f"[Warning] Missing report: {report_path}")
        continue

    # Load & filter
    df = pd.read_csv(report_path)
    df = df.dropna(subset=['Clade'])
    df = df[df['Clade'].isin(clade_order)]
    df['Clade'] = pd.Categorical(df['Clade'], categories=clade_order, ordered=True)

    # Percentage conversion
    for col in cols:
        df[col] = df[col] / df['total_paths'] * 100

    present_clades = [c for c in clade_order if c in df['Clade'].unique()]
    num_clades     = len(present_clades)

    # 2×5 panel layout
    fig, axes = plt.subplots(2, 5, figsize=(25, 10), sharey=True,
                             gridspec_kw={'wspace': 0.2, 'hspace': 0.3})
    fig.subplots_adjust(left=0.10, right=0.75)
    axes_flat = axes.flatten()

    # Plot each clade
    for idx, ax in enumerate(axes_flat):
        if idx < num_clades:
            clade = present_clades[idx]
            sub   = df[df['Clade']==clade].sort_values(
                        ['t2t','gapped_t2t','gapped_incomplete','incomplete','gapped_no_telomeres'],
                        ascending=False
                    )
            sub[cols].plot(kind='bar', stacked=True, ax=ax,
                           color=[colors[k] for k in cols],
                           width=1.0, edgecolor='none', legend=False)
            ax.set_title(display_names.get(clade, clade), fontsize=14, pad=14)
            ax.set_xticks([])
            ax.set_xlabel(f"N={len(sub)}", fontsize=12, labelpad=14)
        else:
            ax.axis('off')

    # Shared Y-axis label
    fig.text(0.06, 0.5, 'Chromosomes (%)',
             va='center', rotation='vertical', fontsize=14)

    # Single legend
    handles = [
        Patch(facecolor=colors[k],
              label=('No Telomeres' if k=='no_telomeres_comp'
                     else k.replace('_',' ').title()))
        for k in cols
    ]
    fig.legend(handles=handles, ncol=1,
               bbox_to_anchor=(0.78, 0.5), loc='center left',
               frameon=False, fontsize=12)

    plt.tight_layout(rect=[0.10, 0, 0.75, 1])

    # Save all formats
    for ext in ('png','pdf','svg'):
        out = os.path.join(plot_dir, f'vgp_telomere_completeness_{run_label}.{ext}')
        fig.savefig(out)
    plt.close(fig)
    print(f"Saved plots for {run_label} → {plot_dir}")
