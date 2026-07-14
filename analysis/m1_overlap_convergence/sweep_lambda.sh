#!/bin/bash
SP="/private/tmp/claude-501/-Users-manuellerchner-git-ProliferatingCellCollectives-IDP/ef2ec6dc-72e3-4174-bd1f-ea3ce83aebfa/scratchpad"
BIN=/Users/manuellerchner/git/ProliferatingCellCollectives-IDP/code/cpp/build/src/cellcollectives
export OMP_NUM_THREADS=1
CONFIGS=(
 "hard_L2 hard 0.01" "hard_L3 hard 0.001"
 "soft_L2 soft 0.01" "soft_L3 soft 0.001"
)
run_one(){ read tag mode lam <<< "$1"; d="$SP/orient/$tag"; rm -rf "$d"; mkdir -p "$d"; cd "$d"
  perl -e 'alarm shift; exec @ARGV' 6000 mpirun -np 1 "$BIN" -mode $mode -end_radius 50 -lambda $lam \
     -cfl_factor 0.5 -kcc 20000 -log_every_colony_radius_delta 10 > run.log 2>&1
  echo "DONE $tag exit=$? $(date +%H:%M:%S)" >> "$SP/orient_status.txt"; }
export -f run_one; export SP BIN
: > "$SP/orient_status.txt"
for c in "${CONFIGS[@]}"; do run_one "$c" & done
wait; echo "ALL DONE" >> "$SP/orient_status.txt"
