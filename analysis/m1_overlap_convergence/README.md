# Soft-model overlap: convergence & stability study

Answers Reviewer 2's major concern (1): is the soft model's unphysical central packing
(`phi_center > 1`) an intrinsic limitation, or an under-resolved-parameter artifact?

Produces paper **Fig. 5** (`soft_overlap_analysis.png`) and backs the derivation in
§4.4.2 (Eq. 11). All runs use the production C++ simulator in `code/cpp`.

## Result

- **(a) Independent of the step-size.** At the physical stiffness `k_cc = 2e4`, four runs
  spanning a 10x range of CFL factor (hence `dt`) collapse onto one `phi_center(R)` curve,
  well above `phi = 1` and far above the hard model → the overlap is **converged in dt**,
  not a time-resolution artifact.
- **(b) Set by the stiffness.** Increasing `k_cc` (each at a stable step-size) drives
  `phi_center` toward the rigid/hard limit (~0.9).
- **(c) Stiffness cannot be raised for free.** The measured critical CFL factor separating
  stable from unstable runs scales as `k_cc^-0.96`, matching the derived stability limit
  `dt ~ zeta/k_cc` (paper Eq. 11). So an accurate soft model needs a prohibitively small
  `dt` and merely reproduces the hard model at higher cost.

`phi_center` is the mean packing fraction over `r < 6`, computed exactly as in
`plot_combined.ipynb` (spherocylinder area / annular bin area). Colony radius = 99th
percentile of cell distance from center. Runs use `lambda = 1e-4` (largest-overlap regime).

## Files

| File | Purpose |
|------|---------|
| `plot_soft_convergence.ipynb` | **Main notebook** — reads the CSVs below, produces `soft_overlap_analysis.png` in the repo plotting style. |
| `soft_overlap_analysis.png` | Paper Fig. 5 (3-panel). |
| `conv_curves.csv` | `phi_center` vs radius per (mode, cfl) — panel (a). |
| `kcc_phi.csv` | `phi_center` at `R~20` per stiffness — panel (b). |
| `stab_results.txt` | Stability grid: `kcc  cfl  warns=<n>` (warns==0 => stable) — panel (c). |
| `sweep.sh`, `sweep_r50.sh` | Run the R=30 / R=50 convergence + stiffness sweeps. |
| `stab_grid.sh` | Run the `(k_cc, cfl)` stability grid. |
| `phi_vs_r.py` | Extract `phi_center` vs radius from a run's VTK snapshots. |
| `gen_curves.py`, `gen_kcc.py` | Build `conv_curves.csv` / `kcc_phi.csv` from raw runs. |
| `make_combined.py` | Standalone-script equivalent of the notebook. |

Raw VTK output is large and **not committed**; only the derived CSVs are kept. Regenerate
raw data with the `sweep*.sh` / `stab_grid.sh` scripts, then `gen_*.py`, then the notebook.

## Building the simulator on macOS

The production build targets Linux/GCC. On macOS (Apple clang) two **local, uncommitted**
tweaks are needed:

- `code/cpp/CMakeLists.txt`: replace the GCC-only flag line with
  `set(CMAKE_CXX_FLAGS "-O3 -ffast-math -DNDEBUG -march=native -fPIC -funroll-loops")`.
- `code/cpp/cmake/modules/petsc.cmake` (system-PETSc branch): use
  `set(PETSC_LIBRARIES ${PETSC_PKG_LINK_LIBRARIES})` (full paths) to avoid the bare-name
  `petsc` target collision.

Configure with OpenMP hints, then build:

```
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DOpenMP_C_FLAGS="-Xclang -fopenmp -I$(brew --prefix libomp)/include" -DOpenMP_C_LIB_NAMES=omp \
  -DOpenMP_CXX_FLAGS="-Xclang -fopenmp -I$(brew --prefix libomp)/include" -DOpenMP_CXX_LIB_NAMES=omp \
  -DOpenMP_omp_LIBRARY="$(brew --prefix libomp)/lib/libomp.dylib"
cmake --build . --target cellcollectives -j 8
```

Deps: `brew install petsc open-mpi libomp`.
