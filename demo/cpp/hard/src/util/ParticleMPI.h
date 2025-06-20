#pragma once
#include <mpi.h>
#include <petsc.h>

#include "simulation/Particle.h"

// A C-style struct for efficient MPI communication. This struct holds the
// persistent geometric and physical properties of a particle needed for
// reconstructing it on a different processor. Transient data like force,
// torque, or rank-specific identifiers like localID are not included.
struct ParticleData {
  PetscInt gID;
  PetscInt localID;
  double position[Particle::POSITION_SIZE];
  double quaternion[Particle::QUATERNION_SIZE];
  double length;
  double l0;
  double diameter;
};

void createParticleMPIType(MPI_Datatype* particle_type);
