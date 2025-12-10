#!/bin/bash
#===============================================================================
# 25.12.08_teloscope_TTAGGG_array.sh
# 
# Process NCBI shark assemblies with Teloscope (TTAGGG motif)
# - Downloads assemblies from NCBI in parallel
# - Runs gfastats + teloscope (parallel runs)
#===============================================================================

set -euo pipefail

#----------------------------
# 1. Configuration
#----------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="/mnt/d/research/teloscope_article/telomere_analysis"
ASM_LIST="${WORK_DIR}/25.10.23_ncbi_sharks.rerun.asms.txt"
OUTPUT_DIR="${WORK_DIR}/25.12.08_shark_annotations"
FAIL_LOG="${OUTPUT_DIR}/teloscope_failures.log"
DOWNLOAD_DIR="${OUTPUT_DIR}/downloads"

# Parallelism settings
MAX_DOWNLOAD_JOBS=8       # Parallel NCBI downloads
MAX_TELOSCOPE_JOBS=2      # Max runs
THREADS_PER_JOB=8         # Max threads per run

#----------------------------
# 2. Setup
#----------------------------
mkdir -p "$OUTPUT_DIR" "$DOWNLOAD_DIR"
echo "===== Teloscope NCBI Sharks Pipeline =====" | tee "$FAIL_LOG"
echo "Start time:    $(date)" | tee -a "$FAIL_LOG"
echo "Host:          $(hostname)" | tee -a "$FAIL_LOG"
echo "Assembly list: $ASM_LIST" | tee -a "$FAIL_LOG"
echo "Output dir:    $OUTPUT_DIR" | tee -a "$FAIL_LOG"
echo "===========================================" | tee -a "$FAIL_LOG"

# Check dependencies
for cmd in datasets gfastats teloscope; do
  if ! command -v "$cmd" &>/dev/null; then
    echo "[ERROR] Required command not found: $cmd" | tee -a "$FAIL_LOG"
    exit 1
  fi
done

# Read assembly list into array
mapfile -t ASM_ARRAY < <(grep -v '^#' "$ASM_LIST" | grep -v '^$' | sort -u)
TOTAL=${#ASM_ARRAY[@]}
echo "Total assemblies to process: $TOTAL" | tee -a "$FAIL_LOG"

#----------------------------
# 3. Download assemblies from NCBI (parallel)
#----------------------------
# PHASE 1 DISABLED - Using previously downloaded genomes
# echo ""
# echo "=== Phase 1: Downloading assemblies from NCBI ===" | tee -a "$FAIL_LOG"
#
# download_assembly() {
#   local asm="$1"
#   local outdir="$2"
#   local logfile="$3"
#   local fasta_gz="${outdir}/${asm}.fa.gz"
#
#   # Skip if already downloaded
#   if [[ -f "$fasta_gz" ]]; then
#     echo "[SKIP] $asm already downloaded"
#     return 0
#   fi
#
#   echo "[DL] Downloading $asm..."
#
#   # Create temp directory for this download
#   local tmpdir="${outdir}/tmp_${asm}"
#   mkdir -p "$tmpdir"
#
#   # Download using NCBI datasets
#   if datasets download genome accession "$asm" \
#       --include genome \
#       --filename "${tmpdir}/${asm}.zip" 2>/dev/null; then
#
#     # Extract and rename
#     if unzip -q -o "${tmpdir}/${asm}.zip" -d "$tmpdir" 2>/dev/null; then
#       # Find the fasta file (handles both .fna and .fna.gz)
#       local fna_file
#       fna_file=$(find "$tmpdir" -name "*.fna" -o -name "*.fna.gz" 2>/dev/null | head -1)
#
#       if [[ -n "$fna_file" && -f "$fna_file" ]]; then
#         if [[ "$fna_file" == *.gz ]]; then
#           mv "$fna_file" "$fasta_gz"
#         else
#           gzip -c "$fna_file" > "$fasta_gz"
#         fi
#         echo "[OK] $asm downloaded"
#       else
#         echo "[FAIL] $asm: no FASTA found in archive" | tee -a "$logfile"
#       fi
#     else
#       echo "[FAIL] $asm: unzip failed" | tee -a "$logfile"
#     fi
#   else
#     echo "[FAIL] $asm: datasets download failed" | tee -a "$logfile"
#   fi
#
#   # Cleanup temp
#   rm -rf "$tmpdir"
# }
#
# export -f download_assembly
# export DOWNLOAD_DIR FAIL_LOG
#
# # Run downloads in parallel
# printf '%s\n' "${ASM_ARRAY[@]}" | \
#   xargs -P "$MAX_DOWNLOAD_JOBS" -I {} bash -c 'download_assembly "$@"' _ {} "$DOWNLOAD_DIR" "$FAIL_LOG"
#
# echo "=== Download phase complete ===" | tee -a "$FAIL_LOG"

echo "=== Skipping Phase 1: Using previously downloaded genomes ===" | tee -a "$FAIL_LOG"

#----------------------------
# 4. Process with gfastats + Teloscope (parallel)
#----------------------------
echo ""
echo "=== Phase 2: Running gfastats + Teloscope ===" | tee -a "$FAIL_LOG"

process_assembly() {
  local asm="$1"
  local download_dir="$2"
  local output_dir="$3"
  local logfile="$4"
  local threads="$5"
  
  local fasta_gz="${download_dir}/${asm}.fa.gz"
  local ls_file="${output_dir}/${asm}.chr.ls"
  local chr_fa="${output_dir}/${asm}.chr.fa"
  local out_path="${output_dir}/${asm}.teloscope"
  local report_file="${out_path}/${asm}.telo.report"
  
  # Check if input exists
  if [[ ! -f "$fasta_gz" ]]; then
    echo "[SKIP] $asm: no downloaded FASTA found"
    return 0
  fi
  
  # Skip if already processed
  if [[ -d "$out_path" && -f "$report_file" ]]; then
    echo "[SKIP] $asm: already has teloscope report"
    return 0
  fi
  
  echo "[PROC] $asm: starting gfastats + teloscope"
  
  # 1. List & size-sort scaffolds → .chr.ls
  if ! gfastats -j "$threads" -s s "$fasta_gz" 2>/dev/null | cut -f1 > "$ls_file"; then
    echo "[FAIL] $asm: gfastats failed" | tee -a "$logfile"
    return 1
  fi
  
  # Check if file has content
  if [[ ! -s "$ls_file" ]]; then
    echo "[FAIL] $asm: empty scaffold list" | tee -a "$logfile"
    return 1
  fi
  
  # 2. Check for mito-only assemblies
  local prefix
  prefix=$(cut -c1-2 "$ls_file" | uniq | head -1)
  local count
  count=$(grep -c "^${prefix}" "$ls_file" || echo "0")
  
  if (( count <= 1 )); then
    mkdir -p "$out_path"
    echo "# Teloscope report for $asm: only mitochondrial contig—no nuclear molecules" \
      > "$report_file"
    touch "${out_path}/terminal_telomeres.bed"
    echo "[SKIP] $asm: only prefix ${prefix} (mitochondrial?), placeholder created" | tee -a "$logfile"
    return 0
  fi
  
  # 3. Pick the 2-letter prefix from the largest scaffold, grab all matching headers → extract chromosomes
  #    Since ls_file is size-sorted (largest first), head -1 gives the dominant/chromosomal prefix.
  #    Using grep without ^ anchor to match the prefix anywhere (safer for edge cases).
  if ! gfastats -f "$fasta_gz" -o "$chr_fa" -j "$threads" -i <(
    grep -E "$(cut -c1-2 "$ls_file" | uniq | head -1)|CM|NC" "$ls_file"
  ) 2>/dev/null; then
    echo "[FAIL] $asm: chromosome extraction failed" | tee -a "$logfile"
    rm -f "$chr_fa"
    return 1
  fi
  
  # 4. Run Teloscope
  mkdir -p "$out_path"
  
  if teloscope \
      -f "$chr_fa" \
      -o "$out_path" \
      -j "$threads" \
      -t 100000 \
      -x 1 \
      -g -r -e -m -i \
      > "$report_file" 2>&1; then
    echo "[OK] $asm: teloscope complete"
  else
    local rc=$?
    echo "[FAIL] $asm: teloscope exit code $rc" | tee -a "$logfile"
  fi
  
  # Cleanup intermediate files
  rm -f "$chr_fa"
}

export -f process_assembly
export OUTPUT_DIR DOWNLOAD_DIR FAIL_LOG THREADS_PER_JOB

# Run teloscope processing in parallel
# Note: We use || true to prevent xargs from stopping on individual job failures
printf '%s\n' "${ASM_ARRAY[@]}" | \
  xargs -P "$MAX_TELOSCOPE_JOBS" -I {} bash -c \
    'process_assembly "$@" || exit 0' _ {} "$DOWNLOAD_DIR" "$OUTPUT_DIR" "$FAIL_LOG" "$THREADS_PER_JOB" \
  || true

#----------------------------
# 5. Summary
#----------------------------
echo ""
echo "===== Pipeline Complete =====" | tee -a "$FAIL_LOG"
echo "End time: $(date)" | tee -a "$FAIL_LOG"

# Count results
total_reports=$(find "$OUTPUT_DIR" -name "*.telo.report" 2>/dev/null | wc -l)
total_beds=$(find "$OUTPUT_DIR" -name "terminal_telomeres.bed" 2>/dev/null | wc -l)
total_failures=$(grep -c "\[FAIL\]" "$FAIL_LOG" 2>/dev/null || echo "0")

echo "Total assemblies:  $TOTAL" | tee -a "$FAIL_LOG"
echo "Reports generated: $total_reports" | tee -a "$FAIL_LOG"
echo "BED files:         $total_beds" | tee -a "$FAIL_LOG"
echo "Failures:          $total_failures" | tee -a "$FAIL_LOG"
echo "==============================" | tee -a "$FAIL_LOG"

# Optional: cleanup downloads to save space
# echo "Cleaning up downloaded assemblies..."
# rm -rf "$DOWNLOAD_DIR"

echo "Done!"