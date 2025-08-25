#pragma once

#include <petsc.h>

#include <array>

// A plain-old-data (POD) struct for serializing Particle objects for MPI communication.
struct ParticleData {
  PetscInt gID;
  std::array<double, 3> position;
  std::array<double, 4> quaternion;
  double length;
  double l0;
  double diameter;
  int age;
};

inline void createParticleMPIType(MPI_Datatype* particle_type) {
  const int count = 7;
  int blocklengths[] = {1, 3, 4, 1, 1, 1, 1};
  MPI_Aint displacements[count];
  MPI_Datatype types[] = {MPIU_INT, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_INT};

  // Get addresses of each member
  ParticleData pd;
  MPI_Aint base_address;
  MPI_Get_address(&pd, &base_address);
  MPI_Get_address(&pd.gID, &displacements[0]);
  MPI_Get_address(&pd.position, &displacements[1]);
  MPI_Get_address(&pd.quaternion, &displacements[2]);
  MPI_Get_address(&pd.length, &displacements[3]);
  MPI_Get_address(&pd.l0, &displacements[4]);
  MPI_Get_address(&pd.diameter, &displacements[5]);
  MPI_Get_address(&pd.age, &displacements[6]);

  // Make displacements relative to the start of the struct
  for (int i = 0; i < count; ++i) {
    displacements[i] -= base_address;
  }

  MPI_Type_create_struct(count, blocklengths, displacements, types, particle_type);
  MPI_Type_commit(particle_type);
}