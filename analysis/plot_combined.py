import os

import matplotlib.pyplot as plt
import numpy as np
from data_loader import load_all_files, load_latest_iteration


def spherocylinder_area(length, radius=0.25):
    """Projected area of a spherocylinder."""
    return 2 * radius * (length - 2*radius) + np.pi * radius**2


def compute_packing_fraction(shell_particles, r_inner, r_outer, length_col):
    """Compute packing fraction inside a radial shell."""
    if len(shell_particles) == 0:
        return 0.0
    total_area = sum(spherocylinder_area(length)
                     for length in shell_particles[length_col])
    shell_area = np.pi * (r_outer**2 - r_inner**2)
    return total_area / shell_area


def plot_radial_distribution(particles_dict, variable, bin_size=2.0, outname=None):
    """
    Plot radial distribution for multiple simulations.

    Args:
        particles_dict: dict {label: DataFrame}
        variable: column name or 'packing_fraction'
        bin_size: radial bin width
        outname: if provided, save figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, df in particles_dict.items():
        df = df.copy()
        df["dist_center"] = np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2)
        max_radius = int(df["dist_center"].max())
        r_edges = np.arange(0, max_radius + bin_size, bin_size)
        r_centers = 0.5 * (r_edges[1:] + r_edges[:-1])

        vals = []
        for i in range(len(r_centers)):
            mask = (df["dist_center"] >= r_edges[i]) & (
                df["dist_center"] < r_edges[i+1])
            shell = df[mask]

            if variable == "packing_fraction":
                length_col = [c for c in df.columns if "length" in c][0]
                val = compute_packing_fraction(
                    shell, r_edges[i], r_edges[i+1], length_col)
            else:
                val = shell[variable].mean() if len(shell) else 0
            vals.append(val)

        ax.plot(r_centers, vals, "o-", label=label)

    ax.set_xlabel("Radius", fontsize=14)
    ax.set_ylabel(variable.replace("_", " ").title(), fontsize=14)
    ax.set_title(f"Radial Distribution of {variable}", fontsize=14)
    ax.grid(True)
    ax.legend()
    if outname:
        fig.savefig(outname, dpi=300, bbox_inches="tight")
    return fig


def plot_parameter_over_time(sim_dict, x, y, outname=None, rolling=None):
    """Plot time-dependent parameter across multiple simulations."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, df in sim_dict.items():
        x_data, y_data = df[x], df[y]
        if rolling:
            y_data = y_data.rolling(window=rolling).mean()
            x_data = x_data.rolling(window=rolling).mean()
        ax.plot(x_data, y_data, label=label)
    ax.set_xlabel(x, fontsize=13)
    ax.set_ylabel(y, fontsize=13)
    ax.set_title(f"{y} vs {x}", fontsize=14)
    ax.legend()
    ax.grid(True)
    if outname:
        fig.savefig(outname, dpi=300, bbox_inches="tight")
    return fig


def clean_label(sim_dir):
    """
    Create a short label from a simulation directory path.
    Example: '../data/20250928_145202/vtk_output_hard_1.000000e-02'
             -> 'vtk_output_hard_1e-2'
    """
    base = os.path.basename(sim_dir.strip("/"))
    base = base.replace("vtk_output_","")
    # Convert scientific notation directory names into compact form
    base = base.replace("1.000000e-01", "1e-1")
    base = base.replace("1.000000e-02", "1e-2")
    base = base.replace("1.000000e-03", "1e-3")
    return base


def load_and_plot(sim_dirs, base_path="", bin_size=2.0):
    """
    Load multiple simulation directories and produce combined plots.

    Args:
        sim_dirs: list of subdirectories (each containing simulation output)
        base_path: root where simulations are stored
        bin_size: bin width for radial distributions
    """
    particles_dict = {}
    sim_dict = {}

    for sim_dir in sim_dirs:
        print("loading " + sim_dir)

        latest = load_latest_iteration(
            os.path.join(base_path, sim_dir, "data"))
        sim_all = load_all_files(os.path.join(
            base_path, sim_dir, "data"), "simulation")

        particles = latest["particles"]

        # normalize particle columns
        if "lengths_x" in particles:
            particles.rename(columns={"lengths_x": "length"}, inplace=True)
        for drop_col in ["lengths_y", "lengths_z"]:
            if drop_col in particles:
                particles.drop(drop_col, axis=1, inplace=True)

        # use cleaned label
        label = clean_label(sim_dir)
        suffix = ""

        particles = particles.rename(
            columns={
                c: c + suffix for c in particles.columns if c not in ["x", "y", "z"]}
        )
        sim_all = sim_all.rename(
            columns={
                c: c + suffix for c in sim_all.columns if c not in ["simulation_time_s"]}
        )

        particles_dict[label] = particles
        sim_dict[label] = sim_all

    print(
        particles_dict["hard_1e-2"].keys())

    # Example combined plots
    plot_radial_distribution(particles_dict, "length",
                             bin_size, outname="combined_length.png")
    plot_radial_distribution(particles_dict, "stress",
                             bin_size, outname="combined_stress.png")
    plot_radial_distribution(
        particles_dict, "packing_fraction", bin_size, outname="combined_packing.png")
    plot_parameter_over_time(sim_dict, "simulation_time_s",
                             "colony_radius", outname="combined_colony_radius.png")
    plot_parameter_over_time(sim_dict, "colony_radius",
                             "num_particles", outname="combined_growth.png")


if __name__ == "__main__":
    sim_dirs = [
        "../data/20250928_145202/vtk_output_hard_1.000000e-02/",
        "../data/20250928_145202/vtk_output_hard_1.000000e-03/",
        "../data/20250928_145202/vtk_output_hard_1.000000e-04/",
        "../data/20250928_145202/vtk_output_soft_1.000000e-02/",
        "../data/20250928_145202/vtk_output_soft_1.000000e-03/",
        "../data/20250928_145202/vtk_output_soft_1.000000e-04/",
    ]
    load_and_plot(sim_dirs)
