import os

import pandas as pd

from parse_vtu import vtu_to_dataframe


def find_latest_vtu_files(folder):
    # Dictionary to store ALL files grouped by iteration
    all_files = {}  # Format: {iteration: {"domain": {rank: filename}}}

    # Dictionary to track ONLY the latest iteration files
    latest_files = {}  # Format: {"domain": {rank: filename}}

    for file in os.listdir(folder):
        if file.endswith(".vtu"):
            # Parse filename (e.g., "domain_0000122_rank_0014.vtu")
            parts = file[:-4].split("_")  # Remove ".vtu" before splitting
            log_type = parts[0]  # "domain"
            iteration = int(parts[1])  # 122
            rank = parts[3].strip("0") if len(parts[3].strip("0")) > 0 else 0
            rank = int(rank)

            # Initialize iteration entry if not exists
            if iteration not in all_files:
                all_files[iteration] = {}

            # Initialize log_type entry if not exists
            if log_type not in all_files[iteration]:
                all_files[iteration][log_type] = {}

            # Store the file
            all_files[iteration][log_type][rank] = file

    # Find the latest iteration
    latest_iteration = max(all_files.keys()) if all_files else None

    # Populate latest_files with data from the latest iteration
    if latest_iteration is not None:
        latest_files = all_files[latest_iteration]

    return {
        "all_files": all_files,          # All files grouped by iteration
        "latest_iteration": latest_iteration,  # Just the latest iteration number
        "latest_files": latest_files      # Files from the latest iteration
    }


def load_latest_iteration(folder):

    # Usage
    result = find_latest_vtu_files(folder)

    data = dict()
    for log_type in result["latest_files"]:

        df = pd.DataFrame()

        for rank in result["latest_files"][log_type]:

            filename = result["latest_files"][log_type][rank]

            with open(folder + "/" + filename, "r") as f:
                vtu_content = f.read()

            df_temp = vtu_to_dataframe(vtu_content)
            df_temp["rank"] = rank

            df = pd.concat([df, df_temp])

        data[log_type] = df

    return data
