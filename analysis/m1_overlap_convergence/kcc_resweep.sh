#!/bin/bash
# Uniform stiffness sweep for Fig. 5 panel (b): phi_center vs k_cc at R~20.
# Each k_cc at a stable CFL (scaled by the measured k_cc^-0.96 stability law).
SC="/private/tmp/claude-501/-Users-manuellerchner-git-ProliferatingCellCollectives-IDP/8f8b9a87-6789-4baa-ace2-f388c1a9ee58/scratchpad"
BIN=/Users/manuellerchner/git/ProliferatingCellCollectives-IDP/code/cpp/build/src/cellcollectives
R=22; LAM=0.0001
export OMP_NUM_THREADS=2
# tag kcc cfl
CONFIGS=(
 "k2e4 20000 0.5"
 "k5e4 50000 0.2"
 "k1e5 100000 0.1"
 "k2e5 200000 0.05"
 "k5e5 500000 0.02"
)
run_one(){ read tag kcc cfl <<< "$1"; d="$SC/kswp/$tag"; rm -rf "$d"; mkdir -p "$d"; cd "$d"
  perl -e 'alarm shift; exec @ARGV' 5400 mpirun -np 1 "$BIN" -mode soft -end_radius $R -lambda $LAM \
     -cfl_factor $cfl -kcc $kcc -log_every_colony_radius_delta 1 > run.log 2>&1
  echo "DONE $tag kcc=$kcc cfl=$cfl exit=$? $(date +%H:%M:%S)" >> "$SC/kswp_status.txt"; }
export -f run_one; export SC BIN R LAM
: > "$SC/kswp_status.txt"
for c in "${CONFIGS[@]}"; do run_one "$c" & done
wait; echo "ALL DONE $(date +%H:%M:%S)" >> "$SC/kswp_status.txt"
