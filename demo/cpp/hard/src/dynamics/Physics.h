#pragma once

#include <petsc.h>

#include <vector>

#include "Constraint.h"
#include "simulation/Particle.h"
#include "util/PetscRaii.h"

MatWrapper calculate_Jacobian(
    const std::vector<Constraint>& local_constraints,
    const std::vector<Particle>& local_particles);

VecWrapper create_phi_vector(const std::vector<Constraint>& local_constraints);

MatWrapper calculate_MobilityMatrix(const std::vector<Particle>& local_particles, PetscInt global_num_particles, double xi);

MatWrapper calculate_QuaternionMap(const std::vector<Particle>& local_particles);

MatWrapper calculate_stress_matrix(
    const std::vector<Constraint>& local_constraints,
    const std::vector<Particle>& local_particles);
