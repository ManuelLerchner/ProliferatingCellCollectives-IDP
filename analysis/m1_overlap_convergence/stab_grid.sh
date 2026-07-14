#!/bin/bash
SP="/private/tmp/claude-501/-Users-manuellerchner-git-ProliferatingCellCollectives-IDP/ef2ec6dc-72e3-4174-bd1f-ea3ce83aebfa/scratchpad"
BIN=/Users/manuellerchner/git/ProliferatingCellCollectives-IDP/code/cpp/build/src/cellcollectives
export OMP_NUM_THREADS=1
R=20; LAM=0.0001
# (kcc, cfl) grid bracketing the stability boundary for each stiffness
GRID=(
 "2e4 1.5" "2e4 2.0" "2e4 3.0"
 "5e4 0.5" "5e4 0.8" "5e4 1.2"
 "1e5 0.2" "1e5 0.3" "1e5 0.45"
 "2e5 0.1" "2e5 0.18" "2e5 0.28"
 "5e5 0.06" "5e5 0.1" "5e5 0.16"
)
MAXJ=3   # keep light so R=50 runs keep progressing
run_one(){ read kcc cfl <<< "$1"
  tag="stab_k${kcc}_c${cfl}"; d="$SP/stab/$tag"; rm -rf "$d"; mkdir -p "$d"; cd "$d"
  perl -e 'alarm shift; exec @ARGV' 1800 mpirun -np 1 "$BIN" -mode soft -end_radius $R -lambda $LAM \
     -cfl_factor $cfl -kcc $kcc -log_every_colony_radius_delta 100 > run.log 2>&1
  warns=$(grep -c 'unusually large distance' run.log)
  over=$(sed 's/\x1b\[[0-9;]*m//g' run.log | grep -oE 'Colony radius = [0-9.]+' | tail -1 | grep -oE '[0-9.]+')
  dt=$(sed 's/\x1b\[[0-9;]*m//g' run.log | grep -oE 'dt: [0-9.eE+-]+' | tail -1 | grep -oE '[0-9.eE+-]+$')
  echo "$kcc $cfl warns=$warns lastR=$over dt=$dt" >> "$SP/stab_results.txt"
}
export -f run_one; export SP BIN R LAM
: > "$SP/stab_results.txt"
for g in "${GRID[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do sleep 3; done
  run_one "$g" &
done
wait; echo "STAB_GRID_DONE" >> "$SP/stab_results.txt"
