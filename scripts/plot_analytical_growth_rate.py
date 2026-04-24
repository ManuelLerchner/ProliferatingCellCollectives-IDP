"""
Generate analytical relative growth rate curves from Weady et al. (2024).

growth_rate(r, lambda) = exp(-lambda * sigma_bar(r))
sigma_bar(r) = max(0, (2/lambda) * log(1/(8c) - c*lambda*r^2))
c = (sqrt(1 + lambda*R^2/2) - 1) / (2*lambda*R^2)

Saves: latex/figures/comparison_plots/analytical_radial_growth_rate.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

OUT = os.path.join(
    os.path.dirname(__file__),
    "../latex/figures/comparison_plots/analytical_radial_growth_rate.png",
)

R = 100.0
LAMBDAS = [1e-2, 1e-3, 1e-4]
LABELS  = [r"$\lambda = 10^{-2}$", r"$\lambda = 10^{-3}$", r"$\lambda = 10^{-4}$"]
STYLES  = ["solid", "dashed", "dotted"]


def pressure_theoretical(r, R, lam):
    c = (np.sqrt(1.0 + lam * R**2 / 2.0) - 1.0) / (2.0 * lam * R**2)
    return np.maximum(2.0 / lam * np.log(1.0 / (8.0 * c) - c * lam * r**2), 0.0)


r = np.linspace(0, R, 500)

fig, ax = plt.subplots(figsize=(10, 5))

for lam, label, ls in zip(LAMBDAS, LABELS, STYLES):
    sigma = pressure_theoretical(r, R, lam)
    growth = np.exp(-lam * sigma)
    ax.plot(r, growth, color="black", linestyle=ls, lw=2, label=label)

ax.set_xlabel(r"Radius $r$", fontsize=20)
ax.set_ylabel(r"$\langle\text{relative growth rate}\rangle$", fontsize=20)
ax.tick_params(axis="both", labelsize=16)
ax.set_ylim(0, 1.05)
ax.set_xlim(0, R)
ax.grid(True)
ax.legend(title="Sensitivity", fontsize=14, title_fontsize=14)

fig.savefig(OUT, dpi=300, bbox_inches="tight")
print(f"Saved → {OUT}")
