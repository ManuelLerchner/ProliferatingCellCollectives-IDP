#pragma once

#include <petsc.h>

#include <vector>

#include "Constraint.h"
#include "simulation/Particle.h"
#include "util/PetscRaii.h"

void calculate_jacobian_local(
    MatWrapper& D,
    const std::vector<Constraint>& local_constraints,
    int offset);

void create_phi_vector_local(VecWrapper& phi, const std::vector<Constraint>& local_constraints, int size);

void calculate_stress_matrix_local(
    MatWrapper& L,
    const std::vector<Constraint>& local_constraints,
    int offset);

MatWrapper calculate_MobilityMatrix(const std::vector<Particle>& local_particles, double xi);

MatWrapper calculate_QuaternionMap(const std::vector<Particle>& local_particles);
