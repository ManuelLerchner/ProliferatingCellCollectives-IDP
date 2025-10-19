#! /bin/bash

cd build/src

cmake ..
make -j 16

export OMP_NUM_THREADS=1
mpirun -np 18 ./cellcollectives $@