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
            rank = parts[3].strip("0")
            rank = int(rank) if rank else 0

            # Initialize log_type entry if not exists
            if log_type not in all_files:
                all_files[log_type] = {}

            # Initialize iteration entry if not exists
            if iteration not in all_files[log_type]:
                all_files[log_type][iteration] = {}

            # Store the file
            all_files[log_type][iteration][rank] = file

    return all_files


def find_latest_vtu_files(folder):
    all_files = load_files(folder)
    latest_files = {}  # Format: {log_type: {rank: filename}}

    latest_iteration_global = None
    for log_type, iterations in all_files.items():
        latest_iteration = max(iterations.keys())

        if latest_iteration_global is None:
            latest_iteration_global = latest_iteration
        else:
            if latest_iteration != latest_iteration_global:
                # print(
                #     f"Latest iteration {latest_iteration} is not the same as the global latest iteration {latest_iteration_global}"
                # )
                pass

        latest_files[log_type] = iterations[latest_iteration]

    return latest_files


def load_latest_iteration(folder):

    latest_files = find_latest_vtu_files(folder)

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

    # put iteration as first column
    df_all = df_all.reindex(
        columns=["iteration"] + [col for col in df_all.columns if col != "iteration"])
    df_all.sort_values(by="iteration", inplace=True)

    return df_all
