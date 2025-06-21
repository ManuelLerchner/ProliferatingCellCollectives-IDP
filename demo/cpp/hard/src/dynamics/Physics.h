#pragma once

#include <petsc.h>

#include <vector>

#include "Constraint.h"
#include "simulation/Particle.h"
#include "util/PetscRaii.h"

MatWrapper calculate_Jacobian(
    const std::vector<Constraint>& local_constraints,
    const std::vector<Particle>& local_particles,
    const ISLocalToGlobalMappingWrapper& col_map_6d,
    const ISLocalToGlobalMappingWrapper& constraint_map_N);

VecWrapper create_phi_vector(const std::vector<Constraint>& local_constraints, const ISLocalToGlobalMappingWrapper& constraint_map_N);

MatWrapper calculate_MobilityMatrix(const std::vector<Particle>& local_particles, PetscInt global_num_particles, double xi, const ISLocalToGlobalMappingWrapper& col_map_6d);

MatWrapper calculate_QuaternionMap(const std::vector<Particle>& local_particles, const ISLocalToGlobalMappingWrapper& row_map_7d, const ISLocalToGlobalMappingWrapper& col_map_6d);

MatWrapper calculate_stress_matrix(const std::vector<Constraint>& local_constraints, const std::vector<Particle>& local_particles, const ISLocalToGlobalMappingWrapper& length_map, const ISLocalToGlobalMappingWrapper& constraint_map);
