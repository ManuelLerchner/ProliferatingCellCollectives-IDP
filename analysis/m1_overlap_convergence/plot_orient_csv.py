import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
SPD = "."
MODE_COLORS = {"hard": "blue", "soft": "red"}
LINE_STYLES = {"-2": "solid", "-3": "dashed", "-4": "dotted"}
LAMLAB = {"-2": r"$\lambda=10^{-2}$", "-3": r"$\lambda=10^{-3}$", "-4": r"$\lambda=10^{-4}$"}
FS_L, FS_T, FS_LG = 19, 15, 13

cur = pd.read_csv(SPD + "/orient_corr_curves.csv", dtype={"lam": str})
fig, ax = plt.subplots(figsize=(10, 6))
for (mode, lam), g in cur.groupby(["mode", "lam"]):
    g = g.sort_values("r")
    ax.plot(g["r"], g["C"], color=MODE_COLORS[mode], ls=LINE_STYLES[lam], lw=2.2,
            label=f"{mode}, {LAMLAB[lam]}")
ax.axhline(1/np.e, ls=":", color="grey", lw=1.5)
ax.text(8.4, 1/np.e + 0.02, r"$1/e$", fontsize=FS_T, color="grey")
ax.set_xlabel(r"separation $r$ (cell lengths)", fontsize=FS_L)
ax.set_ylabel(r"orientational correlation $C(r)=\langle\cos 2\Delta\theta\rangle$", fontsize=FS_L)
ax.tick_params(labelsize=FS_T); ax.grid(alpha=0.4); ax.set_xlim(0, 12)
ax.legend(fontsize=FS_LG, ncol=2)
plt.tight_layout(); fig.savefig(SPD + "/orientation_correlation.png", dpi=300, bbox_inches="tight")
xi = pd.read_csv(SPD + "/orient_xi.csv", dtype={"lam": str})
print(xi.pivot(index="lam", columns="mode", values="xi").to_string())
