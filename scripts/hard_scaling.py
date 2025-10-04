#!/usr/bin/env python3

import subprocess
import os
from time import sleep


SCRIPT_TEMPLATE = """#!/bin/bash
#SBATCH -J {{MODE}}_{{NUM_RANKS}}_hardscaling
#SBATCH -o ./slurm-%x.%j.out
#SBATCH -D {{BIN_FOLDER}}
#SBATCH --get-user-env
#SBATCH --clusters={{CLUSTER}}
#SBATCH --partition={{PARTITION}}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node={{NUM_RANKS}}
#SBATCH --cpus-per-task=1
#SBATCH --mem=16000mb
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=manuel.lerchner@tum.de
#SBATCH --qos=cm4_tiny

# create unique output dir and move into it
module load slurm_setup
  
module load intel
module load intel-mpi


OUTPUT_DIR="strong_scaling/{{TIME}}/output_{{MODE}}/{{NUM_RANKS}}ranks"
mkdir -p $OUTPUT_DIR
cd $OUTPUT_DIR

make -j

mpirun -n {{NUM_RANKS}} ../../../../cellcollectives -mode {{MODE}} -end_radius {{END_RADIUS}} -LAMBDA {{LAMBDA}} -log_every_colony_radius_delta 5
"""

BIN_FOLDER = "../code/cpp/build/src"
END_RADIUS = 100

LAMBDAS = [1e-3]  # adjust if you want multiple lambdas
MPI_RANKS = [24,  48, 64, 96, 112]
MODES = ["hard", "soft"]

import datetime
time = datetime.datetime.now(tz=pytz.utc).timestamp() * 1000


def launch_job(mode, num_ranks, cluster, partition, lambda_val):
    script = SCRIPT_TEMPLATE
    script = script.replace("{{MODE}}", str(mode))
    script = script.replace("{{NUM_RANKS}}", str(num_ranks))
    script = script.replace("{{BIN_FOLDER}}", BIN_FOLDER)
    script = script.replace("{{CLUSTER}}", cluster)
    script = script.replace("{{PARTITION}}", partition)
    script = script.replace("{{END_RADIUS}}", str(END_RADIUS))
    script = script.replace("{{LAMBDA}}", f"{lambda_val:.1e}")
    script = script.replace("{{TIME}}", str(time))

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
                # cluster/partition choice based on number of ranks
                cluster, partition = ("cm4", "cm4_tiny")

                launch_job(mode, ranks, cluster, partition, lambda_val)
