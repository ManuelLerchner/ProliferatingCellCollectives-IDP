#include "ParticleMPI.h"

#include <petscsys.h>

#include "simulation/ParticleData.h"

void createParticleMPIType(MPI_Datatype* particle_type) {
  const int count = 11;
  int blocklengths[] = {1, 3, 4, 3, 3, 3, 3, 1, 1, 1, 1};
  MPI_Aint displacements[count];
  MPI_Datatype types[] = {MPIU_INT, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};

  // Get addresses of each member
  ParticleData pd;
  MPI_Aint base_address;
  MPI_Get_address(&pd, &base_address);
  MPI_Get_address(&pd.gID, &displacements[0]);
  MPI_Get_address(&pd.position, &displacements[1]);
  MPI_Get_address(&pd.quaternion, &displacements[2]);
  MPI_Get_address(&pd.force, &displacements[3]);
  MPI_Get_address(&pd.torque, &displacements[4]);
  MPI_Get_address(&pd.velocityLinear, &displacements[5]);
  MPI_Get_address(&pd.velocityAngular, &displacements[6]);
  MPI_Get_address(&pd.impedance, &displacements[7]);
  MPI_Get_address(&pd.length, &displacements[8]);
  MPI_Get_address(&pd.l0, &displacements[9]);
  MPI_Get_address(&pd.diameter, &displacements[10]);

  // Make displacements relative to the start of the struct
  for (int i = 0; i < count; ++i) {
    displacements[i] -= base_address;
  }

  MPI_Type_create_struct(count, blocklengths, displacements, types, particle_type);
  MPI_Type_commit(particle_type);
}