#include "PhysicsEngine.h"

#include <petsc.h>

#include <iostream>

#include "Constraint.h"
#include "MappingManager.h"
#include "Physics.h"
#include "simulation/Particle.h"
#include "solver/BBPGD.h"
#include "util/PetscRaii.h"

PhysicsEngine::PhysicsEngine(PhysicsConfig physics_config, SolverConfig solver_config) : physics_config(physics_config), solver_config(solver_config) {}

PhysicsEngine::PhysicsMatrices PhysicsEngine::calculateMatrices(const std::vector<Particle>& local_particles, const std::vector<Constraint>& local_constraints, Mappings mappings) {
  PetscInt local_num_bodies = local_particles.size();
  PetscInt local_num_constraints = local_constraints.size();

  MatWrapper D = calculate_Jacobian(local_constraints, local_particles, mappings.col_map_6d, mappings.constraint_map);
  // PetscPrintf(PETSC_COMM_WORLD, "D Matrix:\n");
  // MatView(D.get(), PETSC_VIEWER_STDOUT_WORLD);

  MatWrapper M = calculate_MobilityMatrix(local_particles, local_num_bodies, physics_config.xi, mappings.col_map_6d);
  // PetscPrintf(PETSC_COMM_WORLD, "Mobility Matrix M:\n");
  // MatView(M.get(), PETSC_VIEWER_STDOUT_WORLD);

  MatWrapper G = calculate_QuaternionMap(local_particles, mappings.row_map_7d, mappings.col_map_6d);
  // PetscPrintf(PETSC_COMM_WORLD, "Quaternion Map G:\n");
  // MatView(G.get(), PETSC_VIEWER_STDOUT_WORLD);

  VecWrapper phi = create_phi_vector(local_constraints, mappings.constraint_map);
  // PetscPrintf(PETSC_COMM_WORLD, "Phi Vector:\n");
  // VecView(phi.get(), PETSC_VIEWER_STDOUT_WORLD);

  return {.D = std::move(D), .M = std::move(M), .G = std::move(G), .phi = std::move(phi)};
}

VecWrapper estimate_phi_dot(const MatWrapper& D, const MatWrapper& M, const VecWrapper& gamma) {
  // phi_dot = D^T @ G @ M @ D @ gamma

  // t1 = D @ gamma
  VecWrapper t1;
  MatCreateVecs(D.get(), NULL, t1.get_ref());
  MatMult(D, gamma.get(), t1.get());

  // t2 = M @ t1
  VecWrapper t2;
  MatCreateVecs(M.get(), NULL, t2.get_ref());
  MatMult(M, t1.get(), t2.get());

  // t3 = D^T @ t2

  VecWrapper phi_dot;
  MatCreateVecs(D.get(), phi_dot.get_ref(), NULL);
  MatMultTranspose(D, t2.get(), phi_dot.get());

  return std::move(phi_dot);
}

VecWrapper PhysicsEngine::solveConstraints(const PhysicsMatrices& matrices, double dt) {
  PetscInt phi_size;
  VecGetLocalSize(matrices.phi.get(), &phi_size);

  // Check if all constraints are already satisfied (zero residual case)
  PetscReal phi_norm;
  VecNorm(matrices.phi.get(), NORM_2, &phi_norm);
  if (phi_norm < 1e-16) {
    PetscPrintf(PETSC_COMM_WORLD, "INFO: All constraints satisfied (phi_norm=%g) - returning zero forces\n", phi_norm);
    VecWrapper zero_solution;
    VecDuplicate(matrices.phi.get(), zero_solution.get_ref());
    VecZeroEntries(zero_solution.get());
    return zero_solution;
  }

  auto gradient = [&](const VecWrapper& gamma) -> VecWrapper {
    // phi_next = phi + dt * phi_dot
    auto phi_dot = estimate_phi_dot(matrices.D, matrices.M, gamma);

    // phi_next = phi + dt * phi_dot
    VecWrapper phi_next;
    VecDuplicate(matrices.phi.get(), phi_next.get_ref());
    VecCopy(matrices.phi.get(), phi_next.get());
    VecAXPY(phi_next.get(), dt, phi_dot.get());

    return phi_next;
  };

  auto residual = [&](const VecWrapper& gradient_val, const VecWrapper& gamma) -> double {
    // projected_gradient = gamma > 0 ? gradient_val : min(0, gradient_val)
    // residual = norm_inf(projected_gradient)

    const PetscScalar *gamma_array, *grad_array;

    VecGetArrayRead(gamma.get(), &gamma_array);
    VecGetArrayRead(gradient_val.get(), &grad_array);

    VecWrapper projected;
    VecDuplicate(gradient_val.get(), projected.get_ref());

    PetscScalar* proj_array;
    VecGetArray(projected.get(), &proj_array);

    PetscInt n;
    VecGetLocalSize(gamma.get(), &n);
    for (PetscInt i = 0; i < n; i++) {
      proj_array[i] = (PetscRealPart(gamma_array[i]) > 0)
                          ? grad_array[i]
                          : std::min(0.0, PetscRealPart(grad_array[i]));
    }
    VecRestoreArray(projected.get(), &proj_array);

    VecRestoreArrayRead(gradient_val.get(), &grad_array);
    VecRestoreArrayRead(gamma.get(), &gamma_array);

    double norm;
    VecNorm(projected.get(), NORM_INFINITY, &norm);
    return norm;
  };

  VecWrapper gamma0;
  VecDuplicate(matrices.phi.get(), gamma0.get_ref());
  VecZeroEntries(gamma0.get());

  VecWrapper gamma = BBPGD(gradient, residual, gamma0, solver_config);

  // Calculate forces
  VecWrapper df;
  MatCreateVecs(matrices.D.get(), NULL, df.get_ref());
  MatMult(matrices.D.get(), gamma.get(), df.get());

  VecWrapper dU;
  MatCreateVecs(matrices.M.get(), NULL, dU.get_ref());
  MatMult(matrices.M.get(), df.get(), dU.get());

  // Calculate deltaC
  VecWrapper deltaC;
  MatCreateVecs(matrices.G.get(), NULL, deltaC.get_ref());
  MatMult(matrices.G.get(), dU.get(), deltaC.get());

  return std::move(deltaC);
}
