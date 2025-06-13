#! /bin/bash

cd build/src

cmake ..
make -j 16

export OMP_NUM_THREADS=12
mpirun -np 16 ./cellcollectives $@