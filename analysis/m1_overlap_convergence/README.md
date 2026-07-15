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
| `kcc_resweep.sh` | Uniform 5-stiffness sweep for panel (b): soft to R=22 at a stable CFL per `k_cc`. |
| `stab_grid.sh` | Run the `(k_cc, cfl)` stability grid. |
| `phi_vs_r.py` | Extract `phi_center` vs radius from a run's VTK snapshots. |
| `gen_curves.py`, `gen_kcc.py` | Build `conv_curves.csv` / `kcc_phi.csv` from raw runs. |
| `measure_kcc_phi.py` | Build `kcc_phi.csv` from `kcc_resweep.sh` output, `phi_center` averaged over R∈[18,22]. |
| `make_combined.py` | Standalone-script equivalent of the notebook. |

Panel (b) uses one uniform protocol across stiffness: each `k_cc` at a CFL below its
measured stability threshold (panel c), grown to R=22, with `phi_center` averaged over the
R∈[18,22] snapshot window. Single-snapshot sampling of `phi_center` is noisy at the ~1–2%
level; the window average removes it (an earlier single-snapshot version showed a spurious
non-monotonic bump at `k_cc=2e5`). Run `kcc_resweep.sh`, then `measure_kcc_phi.py`.

Raw VTK output is large and **not committed**; only the derived CSVs are kept. Regenerate
raw data with the `sweep*.sh` / `stab_grid.sh` scripts, then `gen_*.py`, then the notebook.

## Orientational correlation length (Reviewer 2, major concern 2)

Also here (same simulator, same folder): the nematic orientational correlation function
`C(r) = <cos 2(theta_i - theta_j)>` and correlation length `xi_theta`, producing paper
**Fig. 14** (`orientation_correlation.png`, §6.5). Runs = hard + soft at
λ ∈ {1e-2, 1e-3, 1e-4}, grown to R≈48.

**Result:** `xi_theta` grows with stress sensitivity (≈1.2 cell lengths at λ=1e-4 → ≈1.9 at
λ=1e-2) and is comparable between the hard and soft models → the collision scheme sets the
large-scale domain size (Fig. 13), not the local orientational order.

| File | Purpose |
|------|---------|
| `sweep_lambda.sh` | Run hard+soft at λ∈{1e-2,1e-3} to R=50 (λ=1e-4 reuses the R=50 runs). |
| `orient_corr.py` | Compute `C(r)` and `xi_theta` from a run's snapshots (core cells, cos-2θ, snapshot-averaged). |
| `orient_fig.py` | Produce `orientation_correlation.png` (repo style: hard=blue/soft=red, λ by line style). |
| `orient_corr_curves.csv` | `C(r)` per (mode, λ). |
| `orient_xi.csv` | `xi_theta` per (mode, λ) at R≈48. |

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
