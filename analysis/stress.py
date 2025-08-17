import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from scipy.optimize import curve_fit


def stress_distribution(particles: pd.DataFrame):
    """Run all wavelength detection methods and compare results"""
    # Prepare data (same as your original function)
    particles["dist_center"] = np.sqrt(
        particles["x"]**2 + particles["y"]**2 + particles["z"]**2)

    bins = pd.IntervalIndex.from_tuples(
        [(i, i+1) for i in range(0, int(particles["dist_center"].max()), 3)])
    particles["bin"] = pd.cut(particles["dist_center"], bins=bins)

    avg_stress = particles.groupby(
        "bin", observed=True)["stress"].mean()

    bin_centers = avg_stress.index.map(lambda interval: interval.mid)

    return bin_centers, avg_stress
