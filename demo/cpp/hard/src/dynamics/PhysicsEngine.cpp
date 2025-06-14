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
#include "util/PetscUtil.h"

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

void estimate_phi_dot_inplace(
    const MatWrapper& D,
    const MatWrapper& M,
    const VecWrapper& gamma,
    VecWrapper& t1_workspace,
    VecWrapper& t2_workspace,
    VecWrapper& phi_dot_out) {
  // This function assumes all workspace and output vectors are correctly pre-allocated.

  // Step 1: t1_workspace = D * gamma
  PetscCallAbort(PETSC_COMM_WORLD, MatMult(D.get(), gamma.get(), t1_workspace.get()));

  // Step 2: t2_workspace = M * t1_workspace
  PetscCallAbort(PETSC_COMM_WORLD, MatMult(M.get(), t1_workspace.get(), t2_workspace.get()));

  // Step 3: phi_dot_out = D^T * t2_workspace
  PetscCallAbort(PETSC_COMM_WORLD, MatMultTranspose(D.get(), t2_workspace.get(), phi_dot_out.get()));

  // No return, the result is in phi_dot_out.
}

std::tuple<VecWrapper, VecWrapper, VecWrapper> calculate_forces(const MatWrapper& D, const MatWrapper& M, const MatWrapper& G, const VecWrapper& gamma) {
  // f = D @ gamma
  VecWrapper F;
  PetscCallAbort(PETSC_COMM_WORLD, MatCreateVecs(D.get(), NULL, F.get_ref()));
  PetscCallAbort(PETSC_COMM_WORLD, MatMult(D, gamma.get(), F.get()));

  // U = M @ f
  VecWrapper U;
  PetscCallAbort(PETSC_COMM_WORLD, MatCreateVecs(M.get(), NULL, U.get_ref()));
  PetscCallAbort(PETSC_COMM_WORLD, MatMult(M, F.get(), U.get()));

  // deltaC = G @ U
  VecWrapper deltaC;
  PetscCallAbort(PETSC_COMM_WORLD, MatCreateVecs(G.get(), NULL, deltaC.get_ref()));
  PetscCallAbort(PETSC_COMM_WORLD, MatMult(G, U.get(), deltaC.get()));

  return {std::move(F), std::move(U), std::move(deltaC)};
}

double residual(const VecWrapper& gradient_val, const VecWrapper& gamma) {
  // Get raw arrays (no copies, just access)
  const PetscScalar *grad_array, *gamma_array;
  PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(gradient_val.get(), &grad_array));
  PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(gamma.get(), &gamma_array));

  PetscInt n;
  PetscCallAbort(PETSC_COMM_WORLD, VecGetLocalSize(gamma.get(), &n));

  // Compute local residual (no temporary Vec needed)
  double local_max = 0.0;
  for (PetscInt i = 0; i < n; i++) {
    const double g_i = PetscRealPart(grad_array[i]);
    const double proj_g_i = (PetscRealPart(gamma_array[i]) > 0)
                                ? g_i
                                : std::min(0.0, g_i);
    local_max = std::max(local_max, std::abs(proj_g_i));
  }

  // Restore arrays (critical for PETSc consistency)
  PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(gradient_val.get(), &grad_array));
  PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(gamma.get(), &gamma_array));

  // Global reduction (MPI_MAX for NORM_INFINITY)
  double global_max;
  MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, PETSC_COMM_WORLD);
  return global_max;
}

PhysicsEngine::SolverSolution PhysicsEngine::solveConstraintsSingleConstraint(ParticleManager& particle_manager, double dt) {
  auto local_constraints = particle_manager.constraint_generator->generateConstraints(particle_manager.local_particles, 0);
  auto matrices = calculateMatrices(particle_manager.local_particles, local_constraints);

  PetscInt phi_size;
  PetscCallAbort(PETSC_COMM_WORLD, VecGetLocalSize(matrices.phi.get(), &phi_size));

  // Check if all constraints are already satisfied (zero residual case)
  double min_seperation;
  PetscCallAbort(PETSC_COMM_WORLD, VecMin(matrices.phi.get(), NULL, &min_seperation));
  double max_overlap = -min_seperation;

  if (max_overlap < solver_config.tolerance) {
    return {.constraints = local_constraints, .constraint_iterations = 0, .bbpgd_iterations = 0, .residual = max_overlap};
  }

  // create a workspace for phi_dot and t1 and t2
  VecWrapper phi_dot;
  VecWrapper t1_workspace;
  VecWrapper t2_workspace;
  PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(matrices.phi.get(), phi_dot.get_ref()));
  PetscCallAbort(PETSC_COMM_WORLD, MatCreateVecs(matrices.D.get(), NULL, t1_workspace.get_ref()));
  PetscCallAbort(PETSC_COMM_WORLD, MatCreateVecs(matrices.M.get(), NULL, t2_workspace.get_ref()));

  auto gradient = [&](const VecWrapper& gamma, VecWrapper& phi_next_out) {
    estimate_phi_dot_inplace(matrices.D, matrices.M, gamma, t1_workspace, t2_workspace, phi_dot);

    // phi_next = phi + dt * phi_dot
    PetscCallAbort(PETSC_COMM_WORLD, VecCopy(matrices.phi.get(), phi_next_out.get()));
    PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(phi_next_out.get(), dt, phi_dot.get()));
  };

  VecWrapper gamma0;
  PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(matrices.phi.get(), gamma0.get_ref()));
  PetscCallAbort(PETSC_COMM_WORLD, VecZeroEntries(gamma0.get()));

  auto [gamma, bbpgd_iterations, res] = BBPGD(gradient, residual, gamma0, solver_config);

  auto [F, U, deltaC] = calculate_forces(matrices.D, matrices.M, matrices.G, gamma);

  PhysicsEngine::PhysicsSolution solution = {.deltaC = std::move(deltaC), .f = std::move(F), .u = std::move(U)};
  particle_manager.moveLocalParticlesFromSolution(solution);

  return {.constraints = local_constraints, .constraint_iterations = 1, .bbpgd_iterations = bbpgd_iterations, .residual = res};
}

PhysicsEngine::SolverSolution PhysicsEngine::solveConstraintsRecursiveConstraints(ParticleManager& particle_manager, double dt) {
  VecWrapper GAMMA_PREV;
  PetscCallAbort(PETSC_COMM_WORLD, VecCreate(PETSC_COMM_WORLD, GAMMA_PREV.get_ref()));
  PetscCallAbort(PETSC_COMM_WORLD, VecSetSizes(GAMMA_PREV, 0, PETSC_DETERMINE));
  PetscCallAbort(PETSC_COMM_WORLD, VecSetType(GAMMA_PREV, VECSTANDARD));
  PetscCallAbort(PETSC_COMM_WORLD, VecSetFromOptions(GAMMA_PREV));

  VecWrapper PHI_PREV;
  PetscCallAbort(PETSC_COMM_WORLD, VecCreate(PETSC_COMM_WORLD, PHI_PREV.get_ref()));
  PetscCallAbort(PETSC_COMM_WORLD, VecSetSizes(PHI_PREV, 0, PETSC_DETERMINE));
  PetscCallAbort(PETSC_COMM_WORLD, VecSetType(PHI_PREV, VECSTANDARD));
  PetscCallAbort(PETSC_COMM_WORLD, VecSetFromOptions(PHI_PREV));

  MatWrapper D_PREV;
  PetscCallAbort(PETSC_COMM_WORLD, MatCreate(PETSC_COMM_WORLD, D_PREV.get_ref()));
  PetscCallAbort(PETSC_COMM_WORLD, MatSetSizes(D_PREV, particle_manager.local_particles.size() * 6, 0, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCallAbort(PETSC_COMM_WORLD, MatSetType(D_PREV, MATMPIAIJ));
  PetscCallAbort(PETSC_COMM_WORLD, MatSetFromOptions(D_PREV));
  PetscCallAbort(PETSC_COMM_WORLD, MatAssemblyBegin(D_PREV, MAT_FINAL_ASSEMBLY));
  PetscCallAbort(PETSC_COMM_WORLD, MatAssemblyEnd(D_PREV, MAT_FINAL_ASSEMBLY));

  VecWrapper gamma_diff_workspace;
  VecWrapper phi_dot_workspace;
  VecWrapper t1_workspace;
  VecWrapper t2_workspace;

  std::vector<Constraint> all_constraints;
  int constraint_iterations = 0;
  long long bbpgd_iterations = 0;

  double res = std::numeric_limits<double>::infinity();

  while (constraint_iterations < solver_config.max_recursive_iterations) {
    std::vector<Constraint> current_constraints = particle_manager.constraint_generator->generateConstraints(particle_manager.local_particles, constraint_iterations);
    all_constraints.insert(all_constraints.end(), current_constraints.begin(), current_constraints.end());

    if (constraint_iterations > 0) {
      PetscInt total_constraints = all_constraints.size();

      // calculate total memory usage
      PetscLogDouble total_memory;
      PetscCallAbort(PETSC_COMM_WORLD, PetscMemoryGetCurrentUsage(&total_memory));

      MPI_Allreduce(&total_constraints, &total_constraints, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD);
      PetscPrintf(PETSC_COMM_WORLD, "\n  Recursive Constraint Iteration: %d | Constraints: %4d | Memory: %4.2f MB", constraint_iterations, total_constraints, total_memory / 1024 / 1024);
    }

    auto matrices = calculateMatrices(particle_manager.local_particles, current_constraints);

    // stack PHI with matrices.phi
    PHI_PREV = concatenateVectors(PHI_PREV, matrices.phi);

    // get maximal overlap
    double min_seperation;
    PetscCallAbort(PETSC_COMM_WORLD, VecMin(PHI_PREV.get(), NULL, &min_seperation));
    double max_overlap = -min_seperation;
    if (max_overlap < solver_config.tolerance) {
      break;
    }

    // pad gamma with 0
    VecWrapper zero_pad;
    PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(matrices.phi, zero_pad.get_ref()));
    PetscCallAbort(PETSC_COMM_WORLD, VecSet(zero_pad, 0.0));
    GAMMA_PREV = concatenateVectors(GAMMA_PREV, zero_pad);

    // stack D with matrices.D
    D_PREV = horizontallyStackMatrices(D_PREV, matrices.D);

    if (constraint_iterations == 0) {
      VecDuplicate(GAMMA_PREV.get(), gamma_diff_workspace.get_ref());
      VecDuplicate(PHI_PREV.get(), phi_dot_workspace.get_ref());
      MatCreateVecs(D_PREV.get(), NULL, t1_workspace.get_ref());
      MatCreateVecs(matrices.M.get(), NULL, t2_workspace.get_ref());
    } else {
      // A simple check. If sizes changed, we must recreate.
      PetscInt prev_size, curr_size;
      VecGetSize(gamma_diff_workspace, &prev_size);
      VecGetSize(GAMMA_PREV, &curr_size);
      if (prev_size != curr_size) {
        VecDestroy(gamma_diff_workspace.get_ref());
        VecDestroy(phi_dot_workspace.get_ref());
        VecDestroy(t1_workspace.get_ref());
        VecDestroy(t2_workspace.get_ref());
        VecDuplicate(GAMMA_PREV.get(), gamma_diff_workspace.get_ref());
        VecDuplicate(PHI_PREV.get(), phi_dot_workspace.get_ref());
        MatCreateVecs(D_PREV.get(), NULL, t1_workspace.get_ref());
        MatCreateVecs(matrices.M.get(), NULL, t2_workspace.get_ref());
      }
    }

    auto gradient = [&](const VecWrapper& gamma_curr, VecWrapper& phi_next_out) {
      VecCopy(gamma_curr.get(), gamma_diff_workspace.get());
      VecAXPY(gamma_diff_workspace.get(), -1.0, GAMMA_PREV.get());

      estimate_phi_dot_inplace(D_PREV, matrices.M, gamma_diff_workspace,
                               t1_workspace, t2_workspace, phi_dot_workspace);

      // phi_next_out = PHI_PREV + dt * phi_dot_workspace
      PetscCallAbort(PETSC_COMM_WORLD, VecCopy(PHI_PREV.get(), phi_next_out.get()));
      PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(phi_next_out.get(), dt, phi_dot_workspace.get()));
    };

    // solve for gamma
    auto [GAMMA_NEXT, bbpgd_iterations_temp, res_temp] = BBPGD(gradient, residual, GAMMA_PREV, solver_config);
    res = res_temp;
    bbpgd_iterations += bbpgd_iterations_temp;

    VecWrapper gamma_diff;
    PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(GAMMA_NEXT.get(), gamma_diff.get_ref()));
    PetscCallAbort(PETSC_COMM_WORLD, VecCopy(GAMMA_NEXT.get(), gamma_diff.get()));
    PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(gamma_diff.get(), -1.0, GAMMA_PREV.get()));

    // calculate forces
    auto [df, dU, deltaC] = calculate_forces(D_PREV, matrices.M, matrices.G, gamma_diff);

    // prepare for next iteration
    particle_manager.moveLocalParticlesFromSolution({.deltaC = deltaC, .f = df, .u = dU});
    gradient(GAMMA_NEXT, PHI_PREV);
    GAMMA_PREV = std::move(GAMMA_NEXT);
    D_PREV = std::move(D_PREV);

    constraint_iterations++;
  }

  return {.constraints = all_constraints, .constraint_iterations = constraint_iterations, .bbpgd_iterations = bbpgd_iterations, .residual = res};
}
