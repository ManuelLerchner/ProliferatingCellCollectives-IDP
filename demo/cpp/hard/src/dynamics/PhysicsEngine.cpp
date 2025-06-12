#include "PhysicsEngine.h"

#include <petsc.h>

#include <iostream>

#include "Constraint.h"
#include "MappingManager.h"
#include "Physics.h"
#include "simulation/Particle.h"
#include "simulation/ParticleManager.h"
#include "solver/BBPGD.h"
#include "util/PetscRaii.h"

PhysicsEngine::PhysicsEngine(PhysicsConfig physics_config, SolverConfig solver_config) : physics_config(physics_config), solver_config(solver_config) {}

PhysicsEngine::PhysicsMatrices PhysicsEngine::calculateMatrices(const std::vector<Particle>& local_particles, const std::vector<Constraint>& local_constraints) {
  auto mappings = createMappings(local_particles, local_constraints);

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

std::tuple<VecWrapper, VecWrapper, VecWrapper> calculate_forces(const MatWrapper& D, const MatWrapper& M, const MatWrapper& G, const VecWrapper& gamma) {
  // f = D @ gamma
  VecWrapper F;
  MatCreateVecs(D.get(), NULL, F.get_ref());
  MatMult(D, gamma.get(), F.get());

  // U = M @ f
  VecWrapper U;
  MatCreateVecs(M.get(), NULL, U.get_ref());
  MatMult(M, F.get(), U.get());

  // deltaC = G @ U
  VecWrapper deltaC;
  MatCreateVecs(G.get(), NULL, deltaC.get_ref());
  MatMult(G, U.get(), deltaC.get());

  return {std::move(F), std::move(U), std::move(deltaC)};
}

double residual(const VecWrapper& gradient_val, const VecWrapper& gamma) {
  // projected_gradient = gamma > 0 ? gradient_val : min(0, gradient_val)
  // residum = norm_inf(projected_gradient)

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

PhysicsEngine::SolverSolution PhysicsEngine::solveConstraintsSingleConstraint(ParticleManager& particle_manager, double dt) {
  auto local_constraints = particle_manager.constraint_generator->generateConstraints(particle_manager.local_particles, 0);
  auto matrices = calculateMatrices(particle_manager.local_particles, local_constraints);

  PetscInt phi_size;
  VecGetLocalSize(matrices.phi.get(), &phi_size);

  // Check if all constraints are already satisfied (zero residum case)
  double min_seperation;
  VecMin(matrices.phi.get(), NULL, &min_seperation);
  double max_overlap = -min_seperation;

  if (max_overlap < solver_config.tolerance) {
    return {.constraints = local_constraints, .constraint_iterations = 0, .bbpgd_iterations = 0, .residum = max_overlap};
  }

  auto gradient = [&](const VecWrapper& gamma) -> VecWrapper {
    auto phi_dot = estimate_phi_dot(matrices.D, matrices.M, gamma);

    // phi_next = phi + dt * phi_dot
    VecWrapper phi_next;
    VecDuplicate(matrices.phi.get(), phi_next.get_ref());
    VecCopy(matrices.phi.get(), phi_next.get());
    VecAXPY(phi_next.get(), dt, phi_dot.get());

    return phi_next;
  };

  VecWrapper gamma0;
  VecDuplicate(matrices.phi.get(), gamma0.get_ref());
  VecZeroEntries(gamma0.get());

  auto [gamma, bbpgd_iterations, res] = BBPGD(gradient, residual, gamma0, solver_config);

  auto [F, U, deltaC] = calculate_forces(matrices.D, matrices.M, matrices.G, gamma);

  PhysicsEngine::PhysicsSolution solution = {.deltaC = std::move(deltaC), .f = std::move(F), .u = std::move(U)};
  particle_manager.moveLocalParticlesFromSolution(solution);

  return {.constraints = local_constraints, .constraint_iterations = 1, .bbpgd_iterations = bbpgd_iterations, .residum = res};
}

PhysicsEngine::SolverSolution PhysicsEngine::solveConstraintsRecursiveConstraints(ParticleManager& particle_manager, double dt) {
  VecWrapper GAMMA_PREV;
  VecCreate(PETSC_COMM_WORLD, GAMMA_PREV.get_ref());
  VecSetSizes(GAMMA_PREV, 0, PETSC_DETERMINE);
  VecSetType(GAMMA_PREV, VECSTANDARD);
  VecSetFromOptions(GAMMA_PREV);

  VecWrapper PHI_PREV;
  VecCreate(PETSC_COMM_WORLD, PHI_PREV.get_ref());
  VecSetSizes(PHI_PREV, 0, PETSC_DETERMINE);
  VecSetType(PHI_PREV, VECSTANDARD);
  VecSetFromOptions(PHI_PREV);

  MatWrapper D_PREV;
  MatCreate(PETSC_COMM_WORLD, D_PREV.get_ref());
  MatSetSizes(D_PREV, particle_manager.local_particles.size() * 6, 0, PETSC_DETERMINE, PETSC_DETERMINE);
  MatSetType(D_PREV, MATAIJ);
  MatSetFromOptions(D_PREV);

  std::vector<Constraint> all_constraints;
  int constraint_iterations = 0;
  long long bbpgd_iterations = 0;
  double res = 0;

  while (constraint_iterations < MAX_CONSTRAINT_ITERATIONS) {
    std::vector<Constraint> current_constraints = particle_manager.constraint_generator->generateConstraints(particle_manager.local_particles, constraint_iterations);
    all_constraints.insert(all_constraints.end(), current_constraints.begin(), current_constraints.end());

    auto matrices = calculateMatrices(particle_manager.local_particles, current_constraints);

    // stack PHI with matrices.phi
    Vec arr_p[2] = {PHI_PREV.get(), matrices.phi.get()};
    VecWrapper PHI_TEMP;
    VecConcatenate(2, arr_p, PHI_TEMP.get_ref(), NULL);
    PHI_PREV = std::move(PHI_TEMP);

    // get maximal overlap
    double min_seperation;
    VecMin(PHI_PREV.get(), NULL, &min_seperation);
    double max_overlap = -min_seperation;
    if (max_overlap < solver_config.tolerance) {
      break;
    }

    if (constraint_iterations > 0) {
      PetscPrintf(PETSC_COMM_WORLD, "\n  Recursive Constraint Iteration: %d | Constraints: %4d | Overlap: %f | Res: %f", constraint_iterations, all_constraints.size(), max_overlap, res);
    }

    // pad gamma with 0
    VecWrapper zero_pad;
    VecDuplicate(matrices.phi.get(), zero_pad.get_ref());
    VecZeroEntries(zero_pad.get());
    Vec arr_g[2] = {GAMMA_PREV.get(), zero_pad.get()};
    VecWrapper GAMMA_TEMP;
    VecConcatenate(2, arr_g, GAMMA_TEMP.get_ref(), NULL);
    GAMMA_PREV = std::move(GAMMA_TEMP);

    // stack D with matrices.D
    MatWrapper D_TEMP1;
    Mat mats[2] = {D_PREV.get(), matrices.D.get()};
    MatCreateNest(PETSC_COMM_WORLD, 1, NULL, 2, NULL, mats, D_TEMP1.get_ref());
    MatAssemblyBegin(D_TEMP1.get(), MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(D_TEMP1.get(), MAT_FINAL_ASSEMBLY);
    MatWrapper D_TEMP2;
    MatConvert(D_TEMP1.get(), MATAIJ, MAT_INITIAL_MATRIX, D_TEMP2.get_ref());
    D_PREV = std::move(D_TEMP2);

    auto gradient = [&](const VecWrapper& gamma_curr) -> VecWrapper {
      // print shape of gamma_curr and GAMMA_PREV
      VecWrapper gamma_diff;
      VecDuplicate(gamma_curr.get(), gamma_diff.get_ref());
      VecCopy(gamma_curr.get(), gamma_diff.get());
      VecAXPY(gamma_diff.get(), -1.0, GAMMA_PREV.get());

      auto phi_dot = estimate_phi_dot(D_PREV, matrices.M, gamma_diff);

      // phi_next = phi + dt * phi_dot
      VecWrapper phi_next;
      VecDuplicate(PHI_PREV.get(), phi_next.get_ref());
      VecCopy(PHI_PREV.get(), phi_next.get());
      VecAXPY(phi_next.get(), dt, phi_dot.get());

      return std::move(phi_next);
    };

    // solve for gamma
    auto [GAMMA_NEXT, bbpgd_iterations_temp, res_temp] = BBPGD(gradient, residual, GAMMA_PREV, solver_config);
    bbpgd_iterations += bbpgd_iterations_temp;
    res = res_temp;

    VecWrapper gamma_diff;
    VecDuplicate(GAMMA_NEXT.get(), gamma_diff.get_ref());
    VecCopy(GAMMA_NEXT.get(), gamma_diff.get());
    VecAXPY(gamma_diff.get(), -1.0, GAMMA_PREV.get());

    // calculate forces
    auto [df, dU, deltaC] = calculate_forces(D_PREV, matrices.M, matrices.G, gamma_diff);

    // prepare for next iteration
    particle_manager.moveLocalParticlesFromSolution({.deltaC = deltaC, .f = df, .u = dU});
    PHI_PREV = gradient(GAMMA_NEXT);
    GAMMA_PREV = std::move(GAMMA_NEXT);

    constraint_iterations++;
  }

  return {.constraints = all_constraints, .constraint_iterations = constraint_iterations, .bbpgd_iterations = bbpgd_iterations, .residum = res};
}
