#!/bin/bash
SP="/private/tmp/claude-501/-Users-manuellerchner-git-ProliferatingCellCollectives-IDP/ef2ec6dc-72e3-4174-bd1f-ea3ce83aebfa/scratchpad"
BIN=/Users/manuellerchner/git/ProliferatingCellCollectives-IDP/code/cpp/build/src/cellcollectives
export OMP_NUM_THREADS=1
R=50; LAM=0.0001
CONFIGS=(
 "soft_R50_cfl1.0_k2e4 soft 1.0 20000"
 "soft_R50_cfl0.5_k2e4 soft 0.5 20000"
 "soft_R50_cfl0.25_k2e4 soft 0.25 20000"
 "soft_R50_cfl0.1_k2e4 soft 0.1 20000"
 "hard_R50_cfl0.5 hard 0.5 20000"
)
run_one(){ read tag mode cfl kcc <<< "$1"; d="$SP/runs/$tag"; rm -rf "$d"; mkdir -p "$d"; cd "$d"
  perl -e 'alarm shift; exec @ARGV' 9000 mpirun -np 1 "$BIN" -mode $mode -end_radius $R -lambda $LAM -cfl_factor $cfl -kcc $kcc -log_every_colony_radius_delta 5 > run.log 2>&1
  echo "DONE $tag exit=$? $(date +%H:%M:%S)" >> "$SP/sweep_r50_status.txt"; }
export -f run_one; export SP BIN R LAM
: > "$SP/sweep_r50_status.txt"
for c in "${CONFIGS[@]}"; do run_one "$c" & done
wait; echo "ALL DONE $(date +%H:%M:%S)" >> "$SP/sweep_r50_status.txt"
