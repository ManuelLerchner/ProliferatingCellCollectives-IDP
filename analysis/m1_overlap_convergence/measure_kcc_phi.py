# Panel (b) of Fig. 5: phi_center vs k_cc, averaged over snapshots near R~20.
# Reads the uniform stiffness sweep produced by kcc_resweep.sh (soft runs at
# 5 stiffnesses, each at a stable CFL) plus a hard reference, and writes kcc_phi.csv.
#
# Averaging over the R in [18, 22] window (rather than a single snapshot) removes
# the single-snapshot sampling noise that produced a spurious non-monotonic bump.
import os, glob, sys, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phi_vs_r as P

# Point these at the sweep output (kcc_resweep.sh runs/) and a hard run reaching R>=20.
SOFT_BASE = sys.argv[1] if len(sys.argv) > 1 else "kswp"
HARD_RUN  = sys.argv[2] if len(sys.argv) > 2 else "runs/hard_R50_cfl0.5"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kcc_phi.csv")

def datadir(run, mode):
    d = glob.glob(os.path.join(run, f"vtk_output_{mode}", "data"))
    return d[0] if d else None

def phi_window(dd, lo=18, hi=22):
    c = P.run_curve(dd); w = c[(c.R >= lo) & (c.R <= hi)]
    if len(w) == 0:
        w = c.loc[[(c.R - 20).abs().idxmin()]]
    return w.phi_center.mean(), w.R.mean()

soft = {"k2e4": 2e4, "k5e4": 5e4, "k1e5": 1e5, "k2e5": 2e5, "k5e5": 5e5}
rows = []
for tag, k in soft.items():
    phi, R = phi_window(datadir(os.path.join(SOFT_BASE, tag), "soft"))
    rows.append(dict(kcc=k, R=R, phi_center=phi, mode="soft"))
phi, R = phi_window(datadir(HARD_RUN, "hard"))
rows.append(dict(kcc=2e4, R=R, phi_center=phi, mode="hard"))
pd.DataFrame(rows)[["kcc", "R", "phi_center", "mode"]].to_csv(OUT, index=False)
print(open(OUT).read())
