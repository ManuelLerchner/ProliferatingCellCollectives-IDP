#pragma once

#include <petsc.h>

#include <vector>

#include "Constraint.h"
#include "simulation/Particle.h"
#include "util/PetscRaii.h"

MatWrapper calculate_Jacobian(
    const std::vector<Constraint>& local_constraints,
    PetscInt local_num_bodies,
    ISLocalToGlobalMapping col_map_6d,
    ISLocalToGlobalMapping constraint_map_N);

VecWrapper create_phi_vector(const std::vector<Constraint>& local_constraints, ISLocalToGlobalMapping constraint_map_N);

MatWrapper calculate_MobilityMatrix(const std::vector<Particle>& local_particles, PetscInt global_num_particles, double xi, ISLocalToGlobalMappingWrapper& col_map_6d);

MatWrapper calculate_QuaternionMap(const std::vector<Particle>& local_particles, ISLocalToGlobalMappingWrapper& row_map_7d, ISLocalToGlobalMappingWrapper& col_map_6d);