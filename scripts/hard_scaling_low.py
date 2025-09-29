#!/usr/bin/env python3

import subprocess
import os
from time import sleep


SCRIPT_TEMPLATE = """#!/bin/bash
#SBATCH -J HardScaling_{{MODE}}_{{NUM_RANKS}}ranks
#SBATCH -o ./slurm-%x.%j.out
#SBATCH -D {{BIN_FOLDER}}
#SBATCH --get-user-env
#SBATCH --clusters={{CLUSTER}}
#SBATCH --partition={{PARTITION}}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=1
#SBATCH --mem=16000mb
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=manuel.lerchner@tum.de
#SBATCH --qos=cm4_tiny

# create unique output dir and move into it
module load slurm_setup
module load gcc/14.2.0 
module load mpi.intel/2019.12_gcc  

OUTPUT_DIR="hard_scaling/output_{{MODE}}/{{NUM_RANKS}}ranks"
mkdir -p $OUTPUT_DIR
cd $OUTPUT_DIR

make -j

mpirun -n {{NUM_RANKS}} ../../cellcollectives -mode {{MODE}} -end_radius {{END_RADIUS}} -LAMBDA {{LAMBDA}} -log_frequency_seconds 10
"""

BIN_FOLDER = "../code/cpp/build/src"
END_RADIUS = 100

LAMBDAS = [1e-3]  # adjust if you want multiple lambdas
MPI_RANKS = [1]
MODES = ["hard"]


def launch_job(mode, num_ranks, cluster, partition, lambda_val):
    script = SCRIPT_TEMPLATE
    script = script.replace("{{MODE}}", str(mode))
    script = script.replace("{{NUM_RANKS}}", str(num_ranks))
    script = script.replace("{{BIN_FOLDER}}", BIN_FOLDER)
    script = script.replace("{{CLUSTER}}", cluster)
    script = script.replace("{{PARTITION}}", partition)
    script = script.replace("{{END_RADIUS}}", str(END_RADIUS))
    script = script.replace("{{LAMBDA}}", f"{lambda_val:.1e}")

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
