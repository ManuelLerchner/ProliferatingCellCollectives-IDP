import os
import re

import pandas as pd
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
        return (mode, "$\\lambda = 10^{" + str(m) + "}$")
    except AttributeError:
        return (mode, sim_dir + "assumed $\\lambda = 10^{-2}$")


def load_combined(sim_dirs, base_path="", offset=0):
    """
    Load multiple simulation directories and produce combined plots.

    Args:
        sim_dirs: list of subdirectories (each containing simulation output)
        base_path: root where simulations are stored
        bin_size: bin width for radial distributions
    """
    particles_list = []
    sim_list = []
    params_list = []

    for sim_dir in sim_dirs:
        print("loading " + sim_dir)

        source = sim_dir.split("/")[2]
        label = clean_label(sim_dir)

        for off in range(offset, 1):

            latest = load_latest_iteration(
                os.path.join(base_path, sim_dir, "data"), offset=off)

            particles = latest["particles"]
            # normalize particle columns
            if "lengths_x" in particles:
                particles.rename(columns={"lengths_x": "length"}, inplace=True)
            for drop_col in ["lengths_y", "lengths_z"]:
                if drop_col in particles:
                    particles.drop(drop_col, axis=1, inplace=True)

            particles["mode"] = label[0]
            particles["sensitivity"] = label[1]
            particles["sim_dir"] = source
            particles["offset"] = off
            particles_list.append(particles)

        sim_all = load_all_files(os.path.join(
            base_path, sim_dir, "data"), "simulation")
        params_all = load_all_files(os.path.join(
            base_path, sim_dir, "data"), "parameters")

        sim_all["mode"] = label[0]
        sim_all["sensitivity"] = label[1]
        sim_all["sim_dir"] = source

        params_all["mode"] = label[0]
        params_all["sensitivity"] = label[1]
        params_all["sim_dir"] = source

        sim_list.append(sim_all)
        params_list.append(params_all)

    particles_df = pd.concat(particles_list, ignore_index=True)
    sim_df = pd.concat(sim_list, ignore_index=True)
    params_df = pd.concat(params_list, ignore_index=True)

    return particles_df, sim_df, params_df
