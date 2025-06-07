#pragma once

#include <petsc.h>

#include <vector>

#include "Particle.h"
#include "util/PetscRaii.h"

MatWrapper calculate_MobilityMatrix(std::vector<Particle>& local_particles, PetscInt global_num_particles, ISLocalToGlobalMappingWrapper& ltog_map);

MatWrapper calculate_QuaternionMap(std::vector<Particle>& local_particles, PetscInt global_num_particles, ISLocalToGlobalMappingWrapper& ltog_map);