#pragma once

#include <petsc.h>

#include <vector>

#include "Constraint.h"
#include "simulation/Particle.h"
#include "util/Config.h"
#include "util/PetscRaii.h"

VecWrapper getLengthVector(const std::vector<Particle>& local_particles);
VecWrapper getLdotVector(const std::vector<Particle>& local_particles);

void calculate_ldot_inplace(const MatWrapper& L, const VecWrapper& l, const VecWrapper& gamma, double lambda, double tau, VecWrapper& ldot_curr_out, VecWrapper& stress_curr_out, VecWrapper& impedance_curr_out);

void calculate_external_velocities(VecWrapper& U_ext, VecWrapper& F_ext_workspace, const std::vector<Particle>& local_particles, const MatWrapper& M, double dt, int constraint_iterations, PhysicsConfig physics_config);

void calculate_jacobian_local(
    MatWrapper& D,
    const std::vector<Constraint>& local_constraints,
    PetscInt offset);

void create_phi_vector_local(VecWrapper& phi, const std::vector<Constraint>& local_constraints, PetscInt offset);

void create_gamma_vector_local(VecWrapper& gamma, const std::vector<Constraint>& local_constraints, PetscInt offset);

void calculate_stress_matrix_local(
    MatWrapper& L,
    const std::vector<Constraint>& local_constraints,
    PetscInt offset);

MatWrapper calculate_MobilityMatrix(const std::vector<Particle>& local_particles, double xi);

MatWrapper calculate_QuaternionMap(const std::vector<Particle>& local_particles);
