import os
import re

from data_loader import load_all_files, load_latest_iteration


def clean_label(sim_dir):
    """
    Create a short label from a simulation directory path.
    Example: '../data/20250928_145202/vtk_output_hard_1.000000e-02'
             -> 'vtk_output_hard_1e-2'
    """

    if "hard" in sim_dir:
        mode = "Hard"
    elif "soft" in sim_dir:
        mode = "Soft"
    else:
        mode = "Unknown"

    try:
        m = int(re.search(r"e([+-]?\d+)", sim_dir).group(1))
        return (mode, m, mode + " ($\lambda = 10^{" + str(m) + "}$)")
    except AttributeError:
        return (mode, -2, sim_dir + "assumed $\lambda = 10^{-2}$")


def load_combined(sim_dirs, base_path=""):
    """
    Load multiple simulation directories and produce combined plots.

    Args:
        sim_dirs: list of subdirectories (each containing simulation output)
        base_path: root where simulations are stored
        bin_size: bin width for radial distributions
    """
    particles_dict = {}
    sim_dict = {}
    params_dict = {}

    for sim_dir in sim_dirs:
        print("loading " + sim_dir)

        latest = load_latest_iteration(
            os.path.join(base_path, sim_dir, "data"))
        sim_all = load_all_files(os.path.join(
            base_path, sim_dir, "data"), "simulation")
        params_all = load_all_files(os.path.join(
            base_path, sim_dir, "data"), "parameters")

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
        params_dict[label] = params_all

    return particles_dict, sim_dict, params_dict
