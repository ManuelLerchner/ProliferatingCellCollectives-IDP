#pragma once
#include <mpi.h>
#include <petsc.h>

#include "simulation/Particle.h"
#include "simulation/ParticleData.h"

void createParticleMPIType(MPI_Datatype* particle_type);
