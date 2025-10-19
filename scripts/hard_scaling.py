#!/usr/bin/env python3

import subprocess
import os
from time import sleep
import datetime
import pytz

SCRIPT_TEMPLATE = """#!/bin/bash
#SBATCH -J {{MODE}}_{{NUM_RANKS}}_hardscaling
#SBATCH -o ./slurm-%x.%j.out
#SBATCH -D {{BIN_FOLDER}}
#SBATCH --get-user-env
#SBATCH --clusters={{CLUSTER}}
#SBATCH --partition={{PARTITION}}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node={{NTASKS_PER_NODE}}
#SBATCH --cpus-per-task=1
#SBATCH --mem=16000mb
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=manuel.lerchner@tum.de
#SBATCH --qos=cm4_tiny

module load slurm_setup
module load intel
module load intel-mpi

OUTPUT_DIR="strong_scaling/{{TIME}}/output_{{MODE}}/{{NUM_RANKS}}ranks"
mkdir -p $OUTPUT_DIR
cd $OUTPUT_DIR

make -j

export OMP_NUM_THREADS=1
mpirun -n {{NUM_RANKS}} ../../../../cellcollectives -mode {{MODE}} -end_radius {{END_RADIUS}} -LAMBDA {{LAMBDA}} -log_every_colony_radius_delta 5
"""

BIN_FOLDER = "../code/cpp/build/src"
END_RADIUS = 100

LAMBDAS = [1e-3]
MODES = ["hard", "soft"]

# Combine both ranges of ranks
MPI_RANKS = [1, 2, 4, 8, 16, 24, 48, 64, 96, 112]

time = datetime.datetime.now().timestamp() * 1000


def launch_job(mode, num_ranks, cluster, partition, lambda_val):
    ntasks_per_node = max(num_ranks, 24)

    script = SCRIPT_TEMPLATE
    script = script.replace("{{MODE}}", str(mode))
    script = script.replace("{{NUM_RANKS}}", str(num_ranks))
    script = script.replace("{{BIN_FOLDER}}", BIN_FOLDER)
    script = script.replace("{{CLUSTER}}", cluster)
    script = script.replace("{{PARTITION}}", partition)
    script = script.replace("{{END_RADIUS}}", str(END_RADIUS))
    script = script.replace("{{LAMBDA}}", f"{lambda_val:.1e}")
    script = script.replace("{{TIME}}", str(time))
    script = script.replace("{{NTASKS_PER_NODE}}", str(ntasks_per_node))

    filename = f"job_{mode}_{num_ranks}ranks_{lambda_val:.0e}.sh"
    with open(filename, "w") as f:
        f.write(script)

    print(
        f"Submitting job: mode={mode}, ranks={num_ranks}, lambda={lambda_val:.1e}")
    subprocess.call(["sbatch", filename])
    os.remove(filename)
    sleep(2)


if __name__ == "__main__":
    for mode in MODES:
        for lambda_val in LAMBDAS:
            for ranks in MPI_RANKS:
                cluster, partition = ("cm4", "cm4_tiny")
                launch_job(mode, ranks, cluster, partition, lambda_val)
