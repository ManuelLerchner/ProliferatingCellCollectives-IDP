#!/bin/bash
#SBATCH -J growth
#SBATCH -o ./slurm-%x.%j.out
#SBATCH -D ./
#SBATCH --get-user-env
#SBATCH --clusters=inter
#SBATCH --partition=cm4_inter
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=112
#SBATCH --cpus-per-task=1
#SBATCH --mem=16000mb
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=manuel.lerchner@tum.de

# create unique output dir and move into it
module load slurm_setup
module load gcc/14.2.0 
module load mpi.intel/2019.12_gcc  
module load python/3.13.3  

python3 growth_comparison.py 
