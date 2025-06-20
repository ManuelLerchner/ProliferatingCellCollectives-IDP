#include "ParticleMPI.h"

#include <petscsys.h>

void createParticleMPIType(MPI_Datatype* particle_type) {
  const int num_blocks = 7;
  int block_lengths[num_blocks] = {1, 1, 3, 4, 1, 1, 1};

  MPI_Datatype types[num_blocks] = {MPI_LONG_LONG, MPI_LONG_LONG, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};

  MPI_Aint displacements[num_blocks];
  ParticleData pd;  // Instance to calculate displacements

  MPI_Get_address(&pd.gID, &displacements[0]);
  MPI_Get_address(&pd.localID, &displacements[1]);
  MPI_Get_address(&pd.position, &displacements[2]);
  MPI_Get_address(&pd.quaternion, &displacements[3]);
  MPI_Get_address(&pd.length, &displacements[4]);
  MPI_Get_address(&pd.l0, &displacements[5]);
  MPI_Get_address(&pd.diameter, &displacements[6]);

  // Make displacements relative to the start of the struct
  MPI_Aint base_address = displacements[0];
  for (int i = 0; i < num_blocks; ++i) {
    displacements[i] -= base_address;
  }

  MPI_Type_create_struct(num_blocks, block_lengths, displacements, types, particle_type);
  MPI_Type_commit(particle_type);
}