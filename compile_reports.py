#!/usr/bin/env python3
"""
Collect summary fields from every '<assembly>.telo.report' under the Teloscope output
directory (with telomere statistics included) and write a single CSV with one row per assembly.

Usage:
    1. Edit ROOT_DIR to point to your local '25.06.04_ncbi_teloscope' directory.
    2. Run:
         python3 compile_reports.py
"""

import os
import glob
import re
import csv

# ------------------------------------------------------------------------------
# 1. Configuration: set this to your local Teloscope output folder (with leading slash)
# ------------------------------------------------------------------------------
ROOT_DIR = "/mnt/d/research/teloscope_article/25.06.04_ncbi_teloscope"

# ------------------------------------------------------------------------------
# 2. Regexes for the various summary sections in each .telo.report
# ------------------------------------------------------------------------------
SECTION_ASSEMBLY = re.compile(r'^\+\+\+\s*Assembly Summary Report\s*\+\+\+')
SECTION_STATS    = re.compile(r'^\+\+\+\s*Telomere Statistics\s*\+\+\+')
SECTION_COUNTS   = re.compile(r'^\+\+\+\s*Chromosome Telomere Counts\s*\+\+\+')
SECTION_COMP     = re.compile(r'^\+\+\+\s*Chromosome Telomere/Gap Completeness\s*\+\+\+')

# In “Assembly Summary Report” we capture:
sum_re = {
    'total_paths':     re.compile(r'^Total paths:\s*(\d+)', re.IGNORECASE),
    'total_gaps':      re.compile(r'^Total gaps:\s*(\d+)', re.IGNORECASE),
    'total_telomeres': re.compile(r'^Total telomeres:\s*(\d+)', re.IGNORECASE),
}

# In “Telomere Statistics” we capture:
stats_re = {
    'mean_length':   re.compile(r'^Mean length:\s*([\d\.]+)', re.IGNORECASE),
    'median_length': re.compile(r'^Median length:\s*([\d\.]+)', re.IGNORECASE),
    'min_length':    re.compile(r'^Min length:\s*(\d+)', re.IGNORECASE),
    'max_length':    re.compile(r'^Max length:\s*(\d+)', re.IGNORECASE),
}

# In “Chromosome Telomere Counts” we capture:
cnt_re = {
    'two_telomeres': re.compile(r'^Two telomeres:\s*(\d+)', re.IGNORECASE),
    'one_telomere':  re.compile(r'^One telomere:\s*(\d+)', re.IGNORECASE),
    'no_telomeres':  re.compile(r'^No telomeres:\s*(\d+)', re.IGNORECASE),
}

# In “Chromosome Telomere/Gap Completeness” we capture:
comp_re = {
    't2t':                   re.compile(r'^T2T:\s*(\d+)', re.IGNORECASE),
    'gapped_t2t':            re.compile(r'^Gapped T2T:\s*(\d+)', re.IGNORECASE),
    'missassembled':         re.compile(r'^Missassembled:\s*(\d+)', re.IGNORECASE),
    'gapped_missassembled':  re.compile(r'^Gapped missassembled:\s*(\d+)', re.IGNORECASE),
    'incomplete':            re.compile(r'^Incomplete:\s*(\d+)', re.IGNORECASE),
    'gapped_incomplete':     re.compile(r'^Gapped incomplete:\s*(\d+)', re.IGNORECASE),
    'no_telomeres_comp':     re.compile(r'^No telomeres:\s*(\d+)', re.IGNORECASE),
    'gapped_no_telomeres':   re.compile(r'^Gapped no telomeres:\s*(\d+)', re.IGNORECASE),
    'errors':                re.compile(r'^Errors:\s*(\d+)', re.IGNORECASE),
    'gapped_errors':         re.compile(r'^Gapped errors:\s*(\d+)', re.IGNORECASE),
}

# ------------------------------------------------------------------------------
# 3. Find all '<assembly>.telo.report' files under ROOT_DIR/*/*.telo.report
# ------------------------------------------------------------------------------
pattern = os.path.join(ROOT_DIR, "*", "*.telo.report")
report_paths = glob.glob(pattern)
if not report_paths:
    raise FileNotFoundError(f"No '.telo.report' files found under {ROOT_DIR}")

# ------------------------------------------------------------------------------
# 4. Parse each report into a record dictionary
# ------------------------------------------------------------------------------
records = []

for rpt_path in report_paths:
    base = os.path.basename(rpt_path)
    if not base.endswith(".telo.report"):
        continue
    assembly = base.replace(".telo.report", "")

    # Initialize with empty strings or zeros
    current = {
        'Assembly': assembly,
        'total_paths': "",
        'total_gaps': "",
        'total_telomeres': "",
        'mean_length': "",
        'median_length': "",
        'min_length': "",
        'max_length': "",
        'two_telomeres': "",
        'one_telomere': "",
        'no_telomeres': "",
        't2t': "",
        'gapped_t2t': "",
        'missassembled': "",
        'gapped_missassembled': "",
        'incomplete': "",
        'gapped_incomplete': "",
        'no_telomeres_comp': "",
        'gapped_no_telomeres': "",
        'errors': "",
        'gapped_errors': ""
    }

    state = None

    with open(rpt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Detect section headers
            if SECTION_ASSEMBLY.match(line):
                state = 'assembly'
                continue
            if SECTION_STATS.match(line):
                state = 'stats'
                continue
            if SECTION_COUNTS.match(line):
                state = 'counts'
                continue
            if SECTION_COMP.match(line):
                state = 'comp'
                continue

            # Depending on current state, match against the appropriate regex dictionary
            if state == 'assembly':
                for key, rx in sum_re.items():
                    m = rx.match(line)
                    if m:
                        current[key] = int(m.group(1))
                        break

            elif state == 'stats':
                for key, rx in stats_re.items():
                    m = rx.match(line)
                    if m:
                        # mean/median may be float
                        if key in ('mean_length', 'median_length'):
                            current[key] = float(m.group(1))
                        else:
                            current[key] = int(m.group(1))
                        break

            elif state == 'counts':
                for key, rx in cnt_re.items():
                    m = rx.match(line)
                    if m:
                        current[key] = int(m.group(1))
                        break

            elif state == 'comp':
                for key, rx in comp_re.items():
                    m = rx.match(line)
                    if m:
                        current[key] = int(m.group(1))
                        break

    records.append(current)

# ------------------------------------------------------------------------------
# 5. Write out a CSV with one row per assembly
# ------------------------------------------------------------------------------
fieldnames = [
    'Assembly',
    'total_paths', 'total_gaps', 'total_telomeres',
    'mean_length', 'median_length', 'min_length', 'max_length',
    'two_telomeres', 'one_telomere', 'no_telomeres',
    't2t', 'gapped_t2t', 'missassembled', 'gapped_missassembled',
    'incomplete', 'gapped_incomplete', 'no_telomeres_comp',
    'gapped_no_telomeres', 'errors', 'gapped_errors'
]
output_csv = os.path.join(ROOT_DIR, "teloscope_summary_all.csv")

with open(output_csv, 'w', newline='') as out:
    writer = csv.DictWriter(out, fieldnames=fieldnames)
    writer.writeheader()
    for rec in records:
        writer.writerow(rec)

print(f"Parsed {len(records)} reports. Summary CSV written to:\n  {output_csv}")
