import sys, os, numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, '.')
import orient_corr as OC

SPD = os.path.dirname(os.path.abspath(__file__))
MODE_COLORS = {"hard": "blue", "soft": "red"}
LINE_STYLES = {"-2": "solid", "-3": "dashed", "-4": "dotted"}
LAMLAB = {"-2": r"$\lambda=10^{-2}$", "-3": r"$\lambda=10^{-3}$", "-4": r"$\lambda=10^{-4}$"}
FS_L, FS_T, FS_LG = 19, 15, 13

# (label, run_dir, mode, lam_key)
RUNS = [
    ("hard", SPD+"/orient/hard_L2", "hard", "-2"),
    ("hard", SPD+"/orient/hard_L3", "hard", "-3"),
    ("hard", SPD+"/runs/hard_R50_cfl0.5", "hard", "-4"),
    ("soft", SPD+"/orient/soft_L2", "soft", "-2"),
    ("soft", SPD+"/orient/soft_L3", "soft", "-3"),
    ("soft", SPD+"/runs/soft_R50_cfl0.5_k2e4", "soft", "-4"),
]

fig, ax = plt.subplots(figsize=(10, 6))
tab = []
for _, d, mode, lk in RUNS:
    if not os.path.isdir(d): 
        print("MISSING", d); continue
    rc, C, Rc, N, xi, xie = OC.analyze(d)
    ax.plot(rc, C, color=MODE_COLORS[mode], ls=LINE_STYLES[lk], lw=2.2,
            label=f"{mode}, {LAMLAB[lk]}")
    tab.append((mode, lk, N, round(Rc,1), round(xi,2), round(xie,2)))
ax.axhline(1/np.e, ls=":", color="grey", lw=1.5)
ax.text(ax.get_xlim()[1]*0.7, 1/np.e+0.02, r"$1/e$", fontsize=FS_T, color="grey")
ax.set_xlabel(r"separation $r$ (cell lengths)", fontsize=FS_L)
ax.set_ylabel(r"orientational correlation $C(r)=\langle\cos 2\Delta\theta\rangle$", fontsize=FS_L)
ax.tick_params(labelsize=FS_T); ax.grid(alpha=0.4); ax.set_xlim(0, 12)
ax.legend(fontsize=FS_LG, ncol=2)
plt.tight_layout(); fig.savefig(SPD+"/orientation_correlation.png", dpi=300, bbox_inches="tight")
print("\nmode  lam   N      R    xi_1/e  xi_expfit")
for r in tab: print(f"{r[0]:5s} {r[1]:4s} {r[2]:6d} {r[3]:5.1f}  {r[4]:6.2f}  {r[5]:.2f}")
