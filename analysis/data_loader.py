import os

import pandas as pd
from parse_vtu import vtu_to_dataframe


def load_files(folder):
    # Dictionary to store ALL files grouped by type, then iteration
    all_files = {}  # Format: {log_type: {iteration: {rank: filename}}}

    for file in os.listdir(folder):
        if file.endswith(".vtu"):
            # Parse filename (e.g., "domain_0000122_rank_0014.vtu")
            parts = file[:-4].split("_")  # Remove ".vtu" before splitting
            log_type = parts[0]           # "domain"
            iteration = int(parts[1])     # 122
            rank = parts[3].lstrip("0")
            rank = int(rank) if rank is not "" else 0

            # Initialize log_type entry if not exists
            if log_type not in all_files:
                all_files[log_type] = {}

            # Initialize iteration entry if not exists
            if iteration not in all_files[log_type]:
                all_files[log_type][iteration] = {}

            # Store the file
            all_files[log_type][iteration][rank] = file

    return all_files


def find_latest_particles(folder, offset=0):
    particles = load_files(folder)["particles"]

    if not particles:
        raise ValueError(f"No particle files found in folder: {folder}")

    latest_iteration = max(particles.keys()) + offset
    if latest_iteration not in particles:
        raise ValueError(
            f"No particle files found for iteration {latest_iteration} in folder: {folder}")

    return {"particles": particles[latest_iteration]}


def load_latest_iteration(folder, offset=0):

    latest_files = find_latest_particles(folder, offset)

    data = dict()
    for log_type in latest_files:

        df = pd.DataFrame()

        for rank in latest_files[log_type]:

            filename = latest_files[log_type][rank]

            with open(folder + "/" + filename, "r") as f:
                vtu_content = f.read()

            df_temp = vtu_to_dataframe(vtu_content)
            df_temp["rank"] = rank

            df = pd.concat([df, df_temp])

        data[log_type] = df

    return data


def load_all_files(folder, log_type):
    all_files = load_files(folder)

    df_all = pd.DataFrame()

    for iteration in all_files[log_type]:
        for rank in all_files[log_type][iteration]:
            filename = all_files[log_type][iteration][rank]

            with open(folder + "/" + filename, "r") as f:
                vtu_content = f.read()

            df_temp = vtu_to_dataframe(vtu_content)
            df_temp["iteration"] = iteration
            df_temp["rank"] = rank

            df_all = pd.concat([df_all, df_temp])

    df_all.reset_index(drop=True, inplace=True)

    # put iteration as first column
    df_all = df_all.reindex(
        columns=["iteration"] + [col for col in df_all.columns if col != "iteration"])
    df_all.sort_values(by="iteration", inplace=True)

    return df_all
