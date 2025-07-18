#include "PhysicsEngine.h"

#include <petsc.h>

#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <optional>

#include "Constraint.h"
#include "Physics.h"
#include "logger/ParticleLogger.h"
#include "simulation/Particle.h"
#include "simulation/ParticleManager.h"
#include "solver/BBPGD.h"
#include "util/DynamicPetsc.h"
#include "util/PetscRaii.h"

PhysicsEngine::PhysicsEngine(PhysicsConfig physics_config, SolverConfig solver_config) : physics_config(physics_config), solver_config(solver_config), collision_detector(CollisionDetector(solver_config.tolerance, solver_config.linked_cell_size)) {
  std::random_device rd;
  gen = std::mt19937(42);
}

void updateMatrices(DynamicVecWrapper& PHI_PREV, DynamicVecWrapper& GAMMA_PREV, DynamicMatWrapper& D_PREV, DynamicMatWrapper& L_PREV, const std::vector<Constraint>& new_constraints, const std::vector<Constraint>& old_constraints, int last_recursive_iteration) {
  // First ensure capacity for new constraints

  bool PHI_PREV_cleared = PHI_PREV.ensureCapacity(new_constraints.size());
  bool GAMMA_PREV_cleared = GAMMA_PREV.ensureCapacity(new_constraints.size());
  bool D_PREV_cleared = D_PREV.ensureCapacity(new_constraints.size());
  bool L_PREV_cleared = L_PREV.ensureCapacity(new_constraints.size());

  if (D_PREV_cleared || L_PREV_cleared) {
    MatMPIAIJSetPreallocation(D_PREV, 8 * last_recursive_iteration, NULL, 2 * last_recursive_iteration, NULL);
    MatMPIAIJSetPreallocation(L_PREV, 8 * last_recursive_iteration, NULL, 2 * last_recursive_iteration, NULL);
    MatSetOption(D_PREV, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    MatSetOption(L_PREV, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
  }

  PetscInt ownership_start, ownership_end;
  PetscCallAbort(PETSC_COMM_WORLD, MatGetOwnershipRangeColumn(D_PREV, &ownership_start, &ownership_end));

  // Calculate offset for new data
  PetscInt col_offset = ownership_start + PHI_PREV.getSize();

// Populate matrices and vectors with new data
#pragma omp parallel sections
  {
#pragma omp section
    {
      if (PHI_PREV_cleared) {
        create_phi_vector_local(PHI_PREV, old_constraints, ownership_start);
        PHI_PREV.incrementSize(old_constraints.size());
      }

      create_phi_vector_local(PHI_PREV, new_constraints, col_offset);
      VecAssemblyBegin(PHI_PREV);
      PHI_PREV.incrementSize(new_constraints.size());
    }

#pragma omp section
    {
      if (GAMMA_PREV_cleared) {
        create_gamma_vector_local(GAMMA_PREV, old_constraints, ownership_start);
        GAMMA_PREV.incrementSize(old_constraints.size());
      }

      create_gamma_vector_local(GAMMA_PREV, new_constraints, col_offset);

      VecAssemblyBegin(GAMMA_PREV);
      GAMMA_PREV.incrementSize(new_constraints.size());
    }
#pragma omp section
    {
      if (D_PREV_cleared) {
        calculate_jacobian_local(D_PREV, old_constraints, ownership_start);
        D_PREV.incrementSize(old_constraints.size());
      }

      calculate_jacobian_local(D_PREV, new_constraints, col_offset);
      MatAssemblyBegin(D_PREV, MAT_FINAL_ASSEMBLY);
      D_PREV.incrementSize(new_constraints.size());
    }
#pragma omp section
    {
      if (L_PREV_cleared) {
        calculate_stress_matrix_local(L_PREV, old_constraints, ownership_start);
        L_PREV.incrementSize(old_constraints.size());
      }

      calculate_stress_matrix_local(L_PREV, new_constraints, col_offset);
      MatAssemblyBegin(L_PREV, MAT_FINAL_ASSEMBLY);
      L_PREV.incrementSize(new_constraints.size());
    }
  }

  // Final assembly

  VecAssemblyEnd(PHI_PREV);
  VecAssemblyEnd(GAMMA_PREV);
  MatAssemblyEnd(D_PREV, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(L_PREV, MAT_FINAL_ASSEMBLY);

  // Get info about number of mallocs
  // MatInfo info;
  // PetscCallAbort(PETSC_COMM_WORLD, MatGetInfo(D_PREV, MAT_GLOBAL_SUM, &info));
  // PetscPrintf(PETSC_COMM_WORLD, "Number of mallocs: %d\n", (PetscInt)info.mallocs);
  // PetscPrintf(PETSC_COMM_WORLD, "nz_allocated: %d, nz_used: %d, nz_unneeded: %d\n", (PetscInt)info.nz_allocated, (PetscInt)info.nz_used, (PetscInt)info.nz_unneeded);

  // MatInfo info2;
  // PetscCallAbort(PETSC_COMM_WORLD, MatGetInfo(L_PREV, MAT_GLOBAL_SUM, &info2));
  // PetscPrintf(PETSC_COMM_WORLD, "Number of mallocs: %d\n", (PetscInt)info2.mallocs);
  // PetscPrintf(PETSC_COMM_WORLD, "nz_allocated: %d, nz_used: %d, nz_unneeded: %d\n", (PetscInt)info2.nz_allocated, (PetscInt)info2.nz_used, (PetscInt)info2.nz_unneeded);
}

VecWrapper getLengthVector(const std::vector<Particle>& local_particles) {
  // l is a (global_num_particles, 1) vector
  VecWrapper l;
  VecCreate(PETSC_COMM_WORLD, l.get_ref());
  VecSetSizes(l, local_particles.size(), PETSC_DETERMINE);
  VecSetFromOptions(l);

  // Set values using global indices
  for (int i = 0; i < local_particles.size(); i++) {
    PetscCallAbort(PETSC_COMM_WORLD, VecSetValue(l, local_particles[i].getGID(), local_particles[i].getLength(), INSERT_VALUES));
  }

  VecAssemblyBegin(l);
  VecAssemblyEnd(l);

  return l;
}

void PhysicsEngine::calculate_external_velocities(VecWrapper& U_ext, VecWrapper& F_ext_workspace, const std::vector<Particle>& local_particles, const MatWrapper& M, double dt, int constraint_iterations) {
  PetscCallAbort(PETSC_COMM_WORLD, VecZeroEntries(U_ext));

  // Create a vector for external forces
  // PetscCallAbort(PETSC_COMM_WORLD, VecZeroEntries(F_ext_workspace));

  // // U_ext = M * F_ext
  // PetscCallAbort(PETSC_COMM_WORLD, MatMult(M, F_ext_workspace, U_ext));

  // Add Brownian motion
  if (physics_config.temperature > 0 && constraint_iterations == 0) {
    for (const auto& particle : local_particles) {
      auto brownian_vel = particle.calculateBrownianVelocity(physics_config.temperature, physics_config.monolayer, physics_config.xi, dt, gen);

      PetscInt ix[] = {particle.getGID() * 6, particle.getGID() * 6 + 1, particle.getGID() * 6 + 2, particle.getGID() * 6 + 3, particle.getGID() * 6 + 4, particle.getGID() * 6 + 5};

      PetscCallAbort(PETSC_COMM_WORLD, VecSetValues(U_ext, 6, ix, brownian_vel.data(), ADD_VALUES));
    }
  }

  PetscCallAbort(PETSC_COMM_WORLD, VecAssemblyBegin(U_ext));
  PetscCallAbort(PETSC_COMM_WORLD, VecAssemblyEnd(U_ext));
}

void estimate_phi_dot_movement_inplace(
    const MatWrapper& D,
    const MatWrapper& M,
    const VecWrapper& U_known,
    const VecWrapper& gamma,
    VecWrapper& F_g,
    VecWrapper& U_c,
    VecWrapper& U_total,
    VecWrapper& phi_dot_out) {
  // phi_dot = D^T (U_known + M * D * gamma)

  // Step 1: F_g = D * gamma
  PetscCallAbort(PETSC_COMM_WORLD, MatMult(D, gamma, F_g));

  // Step 2: U_c = M * F_g
  PetscCallAbort(PETSC_COMM_WORLD, MatMult(M, F_g, U_c));

  // Step 3: U_total = U_known + U_c
  PetscCallAbort(PETSC_COMM_WORLD, VecWAXPY(U_total, 1.0, U_known, U_c));

  // Step 4: phi_dot_out = D^T * U_total
  PetscCallAbort(PETSC_COMM_WORLD, MatMultTranspose(D, U_total, phi_dot_out));
}

void calculate_impedance_inplace(VecWrapper& stresses_and_impedance_out, double lambda) {
  // I is a (num_bodies, 1) vector
  // I = np.exp(-lamb * stress_matrix)
  // This function modifies the input vector in-place.

  VecScale(stresses_and_impedance_out, -lambda);
  VecExp(stresses_and_impedance_out);
}

void calculate_growth_rate_vector(const VecWrapper& l, VecWrapper& sigma_and_impedance_out, double lambda, double tau, VecWrapper& growth_rates_out) {
  // growth rate is a (num_bodies, 1) vector
  // growth_rate = l / tau * I

  calculate_impedance_inplace(sigma_and_impedance_out, lambda);

  // sigma_and_impedance_out now contains the impedance
  VecPointwiseMult(growth_rates_out, l, sigma_and_impedance_out);
  VecScale(growth_rates_out, 1 / tau);
}

void calculate_ldot_inplace(const MatWrapper& L, const VecWrapper& l, const VecWrapper& gamma, double lambda, double tau, VecWrapper& ldot_curr_out, VecWrapper& impedance_curr_out) {
  // Use impedance_curr_out as a temporary vector to store stresses (sigma)
  PetscCallAbort(PETSC_COMM_WORLD, MatMult(L, gamma, impedance_curr_out));

  // We now have stresses in impedance_curr_out.
  // We need to calculate the growth rate, which will overwrite ldot_curr_out,
  // and the final impedance, which will overwrite the stresses in impedance_curr_out.
  calculate_growth_rate_vector(l, impedance_curr_out, lambda, tau, ldot_curr_out);
};

void estimate_phi_dot_growth_inplace(
    const MatWrapper& Sigma,
    const VecWrapper& ldot,
    VecWrapper& phi_dot_growth_result) {
  PetscCallAbort(PETSC_COMM_WORLD, MatMultTranspose(Sigma, ldot, phi_dot_growth_result));
}

void calculate_forces(VecWrapper& F, VecWrapper& U, VecWrapper& deltaC, const MatWrapper& D, const MatWrapper& M, const MatWrapper& G, const VecWrapper& U_ext, const VecWrapper& gamma) {
  // f = D @ gamma
  PetscCallAbort(PETSC_COMM_WORLD, MatMult(D, gamma, F));

  // U = M @ f
  PetscCallAbort(PETSC_COMM_WORLD, MatMult(M, F, U));

  // U = U + U_ext
  PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(U, 1.0, U_ext));

  // deltaC = G @ U
  PetscCallAbort(PETSC_COMM_WORLD, MatMult(G, U, deltaC));
}

double residual(const VecWrapper& gradient_val, const VecWrapper& gamma, int N) {
  // This function computes the infinity norm of the projected gradient.
  // The projection is defined as:
  //   projected_gradient_i = gradient_val_i, if gamma_i > 0
  //   projected_gradient_i = min(0, gradient_val_i), if gamma_i <= 0

  // Get local size and pointers to vector data
  PetscInt n_local;
  const PetscScalar *grad_array, *gamma_array;
  PetscCallAbort(PETSC_COMM_WORLD, VecGetLocalSize(gradient_val, &n_local));
  PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(gradient_val, &grad_array));
  PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(gamma, &gamma_array));

  // Compute projected gradient component-wise
  double res = 0.0;  // Initialize to 0 since we're taking max of absolute values

#pragma omp parallel for reduction(max : res)
  for (PetscInt i = 0; i < N; ++i) {
    double v;
    if (PetscRealPart(gamma_array[i]) > 0) {
      v = grad_array[i];
    } else {
      v = std::min(0.0, PetscRealPart(grad_array[i]));
    }
    res = std::max(res, std::abs(v));  // Take absolute value for infinity norm
  }

  // Restore arrays
  PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(gradient_val, &grad_array));
  PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(gamma, &gamma_array));

  // Compute the infinity norm of the projected gradient: max(abs(projected_gradient_i))
  auto res_global = globalReduce(res, MPI_MAX);

  return res_global;
}

struct RecursiveSolverWorkspaces {
  RecursiveSolverWorkspaces(const MatWrapper& D_PREV, const MatWrapper& M, const MatWrapper& G, const MatWrapper& L_PREV, const VecWrapper& GAMMA_PREV, const VecWrapper& PHI_PREV, const VecWrapper& l) {
    // movement
    gamma_diff_workspace = VecWrapper::Like(GAMMA_PREV);
    phi_dot_movement_workspace = VecWrapper::Like(PHI_PREV);
    F_g_workspace = VecWrapper::FromMat(D_PREV);
    U_c_workspace = VecWrapper::FromMat(M);
    U_total_workspace = VecWrapper::FromMat(D_PREV);

    // growth
    phi_dot_growth_result = VecWrapper::FromMatRows(L_PREV);

    ldot_curr_workspace = VecWrapper::Like(l);
    ldot_diff_workspace = VecWrapper::Like(l);
    ldot_prev = VecWrapper::Like(l);
    VecZeroEntries(ldot_prev);

    impedance_curr_workspace = VecWrapper::Like(l);

    U_ext = VecWrapper::FromMat(M);
    F_ext_workspace = VecWrapper::FromMat(M);

    // force
    df = VecWrapper::FromMat(D_PREV);
    du = VecWrapper::FromMat(M);
    dC = VecWrapper::FromMat(G);
  }

  VecWrapper gamma_diff_workspace;
  VecWrapper phi_dot_movement_workspace;
  VecWrapper phi_dot_growth_result;
  VecWrapper F_g_workspace;
  VecWrapper U_c_workspace;
  VecWrapper U_total_workspace;
  VecWrapper ldot_curr_workspace;
  VecWrapper ldot_diff_workspace;
  VecWrapper ldot_prev;
  VecWrapper impedance_curr_workspace;
  VecWrapper U_ext;
  VecWrapper F_ext_workspace;
  VecWrapper df, du, dC;
};

enum class SolverState {
  RUNNING,
  CONVERGED,
  TOO_MANY_CONSTRAINTS,
  NOT_CONVERGED,
};

void PhysicsEngine::updateConstraintsFromSolution(std::vector<Constraint>& constraints, const VecWrapper& gamma, const VecWrapper& phi) {
  std::vector<PetscInt> indices;

  int ownership_start;
  int ownership_end;
  PetscCallAbort(PETSC_COMM_WORLD, VecGetOwnershipRange(gamma, &ownership_start, &ownership_end));

  for (int i = 0; i < constraints.size(); ++i) {
    indices.push_back(ownership_start + i);
  }

  // Scatter the dC vector
  Vec gamma_local;
  VecScatter gamma_scatter;
  IS gamma_is;
  scatterVectorToLocal(gamma, indices, gamma_local, gamma_scatter, gamma_is);

  // Get array pointer
  const PetscScalar* gamma_array;
  VecGetArrayRead(gamma_local, &gamma_array);

  Vec phi_local;
  VecScatter phi_scatter;
  IS phi_is;
  scatterVectorToLocal(phi, indices, phi_local, phi_scatter, phi_is);

  const PetscScalar* phi_array;
  VecGetArrayRead(phi_local, &phi_array);

  // Process each particle (7 values per particle)
  for (int i = 0; i < constraints.size(); ++i) {
    constraints[i].gamma = PetscRealPart(gamma_array[i]);
    constraints[i].signed_distance = PetscRealPart(phi_array[i]);
  }

  // Clean up
  VecRestoreArrayRead(gamma_local, &gamma_array);
  VecRestoreArrayRead(phi_local, &phi_array);
  cleanupScatteredResources(gamma_local, gamma_scatter, gamma_is);
  cleanupScatteredResources(phi_local, phi_scatter, phi_is);
}

PhysicsEngine::SolverSolution PhysicsEngine::solveConstraintsRecursiveConstraints(ParticleManager& particle_manager, double dt, int iter, std::function<void()> exchangeGhostParticles, vtk::ParticleLogger& particle_logger, vtk::ConstraintLogger& constraint_logger) {
  exchangeGhostParticles();

  MatWrapper M = calculate_MobilityMatrix(particle_manager.local_particles, physics_config.xi);
  MatWrapper G = calculate_QuaternionMap(particle_manager.local_particles);

  int global_num_particles = globalReduce<int>(particle_manager.local_particles.size(), MPI_SUM);

  int global_max_particles = globalReduce<int>(particle_manager.local_particles.size(), MPI_MAX);

  int alloc_size = solver_config.getMinPreallocationSize(global_max_particles);

  DynamicVecWrapper GAMMA = DynamicVecWrapper(alloc_size, solver_config.growth_factor);
  DynamicVecWrapper PHI = DynamicVecWrapper(alloc_size, solver_config.growth_factor);
  DynamicMatWrapper D_PREV = DynamicMatWrapper(particle_manager.local_particles.size() * 6, alloc_size, solver_config.growth_factor);
  DynamicMatWrapper L_PREV = DynamicMatWrapper(particle_manager.local_particles.size(), alloc_size, solver_config.growth_factor);

  std::optional<RecursiveSolverWorkspaces> workspaces;

  // Create length mapping for consistent indexing
  VecWrapper l = getLengthVector(particle_manager.local_particles);

  int constraint_iterations = 0;
  long long bbpgd_iterations = 0;

  double res = 0;
  double max_overlap = 0;
  long long bbpgd_iterations_this_step = 0;

  // Reset collision detector state
  collision_detector.reset();

  std::vector<Constraint> all_constraints_set;

  SolverState solver_state = SolverState::RUNNING;

  while (constraint_iterations < solver_config.max_recursive_iterations) {
    // Use a larger tolerance for initial collision detection
    double tolerance = iter == 0 ? 0.1 : 0.01;
    auto new_constraints = collision_detector.detectCollisions(particle_manager, constraint_iterations, tolerance);

    max_overlap = globalReduce(std::accumulate(new_constraints.begin(), new_constraints.end(), 0.0,
                                               [](double acc, const Constraint& c) {
                                                 return std::max(acc, c.signed_distance < 0 ? -c.signed_distance : 0);
                                               }),
                               MPI_MAX);

    // Check convergence
    if (max_overlap < solver_config.allowed_overlap && constraint_iterations > 0) {
      solver_state = SolverState::CONVERGED;
      break;
    }

    // Calculate matrices only for the newly identified constraints
    updateMatrices(PHI, GAMMA, D_PREV, L_PREV, new_constraints, all_constraints_set, last_recursive_iteration);

    all_constraints_set.insert(all_constraints_set.end(), new_constraints.begin(), new_constraints.end());

    auto global_total_constraints = globalReduce<size_t>(all_constraints_set.size(), MPI_SUM);
    auto global_new_constraints_count = globalReduce<size_t>(new_constraints.size(), MPI_SUM);

    // memory
    PetscLogDouble memory_usage;
    PetscCallAbort(PETSC_COMM_WORLD, PetscMemoryGetCurrentUsage(&memory_usage));

    double logged_overlap = std::max(0.0, max_overlap);
    PetscPrintf(PETSC_COMM_WORLD, "\r  Solver Iteration: %4d | Constraints: %6ld (New: %3ld) | Overlap: %8.2e | Residual: %8.2e | BBPGD Iters: %4lld | Memory: %4.2f MB",
                constraint_iterations, global_total_constraints, global_new_constraints_count, logged_overlap, res, bbpgd_iterations_this_step, memory_usage / 1024 / 1024);

    // Recreate workspaces if needed
    bool local_need_to_recreate_workspaces = !workspaces.has_value() || vecSize(GAMMA) != vecSize(workspaces->gamma_diff_workspace);
    int global_recreate_flag = globalReduce(local_need_to_recreate_workspaces ? 1 : 0, MPI_MAX);

    if (global_recreate_flag) {
      workspaces.emplace(D_PREV, M, G, L_PREV, GAMMA, PHI, l);
    }

    // initialize solver

    VecWrapper gamma_old = VecWrapper::Like(GAMMA);
    VecCopy(GAMMA, gamma_old);

    // Calculate external velocities
    calculate_external_velocities(workspaces->U_ext, workspaces->F_ext_workspace, particle_manager.local_particles, M, dt, constraint_iterations);

    // Gradient
    auto gradient = [&](const VecWrapper& gamma_curr, VecWrapper& phi_next_out) {

#pragma omp sections
      {
#pragma omp section
        {
          // --- MOVEMENT PART ---
          // gamma_diff = gamma_curr - gamma_old
          VecWAXPY(workspaces->gamma_diff_workspace, -1.0, gamma_old, gamma_curr);

          // phi_dot_movement = D * M * D^T * gamma_diff
          estimate_phi_dot_movement_inplace(D_PREV, M, workspaces->U_ext, workspaces->gamma_diff_workspace, workspaces->F_g_workspace, workspaces->U_c_workspace, workspaces->U_total_workspace, workspaces->phi_dot_movement_workspace);
        }

#pragma omp section
        {
          // --- GROWTH PART ---
          // ldot_curr = growth_rate(gamma_curr)
          calculate_ldot_inplace(L_PREV, l, gamma_curr, physics_config.getLambdaDimensionless(), physics_config.TAU, workspaces->ldot_curr_workspace, workspaces->impedance_curr_workspace);

          // ldot_diff = ldot_curr - ldot_prev
          VecWAXPY(workspaces->ldot_diff_workspace, -1.0, workspaces->ldot_prev, workspaces->ldot_curr_workspace);

          // phi_dot_growth = -L^T * ldot_diff
          estimate_phi_dot_growth_inplace(L_PREV, workspaces->ldot_diff_workspace, workspaces->phi_dot_growth_result);
        }
      }

      // Start with the base violation: phi_next_out = phi
      PetscCallAbort(PETSC_COMM_WORLD, VecCopy(PHI, phi_next_out));

      Vec vecs_to_add[] = {workspaces->phi_dot_movement_workspace, workspaces->phi_dot_growth_result};
      PetscScalar scales[] = {dt, -dt};
      PetscCallAbort(PETSC_COMM_WORLD, VecMAXPY(phi_next_out, 2, scales, vecs_to_add));
    };

    // Solver
    int N = GAMMA.getSize();
    auto bbpgd_result_recursive = BBPGD(gradient, residual, GAMMA, solver_config.tolerance, solver_config.max_bbpgd_iterations, N);

    res = bbpgd_result_recursive.residual;
    bbpgd_iterations_this_step = bbpgd_result_recursive.bbpgd_iterations;
    bbpgd_iterations += bbpgd_iterations_this_step;

    // Update
    PetscCallAbort(PETSC_COMM_WORLD, VecWAXPY(workspaces->gamma_diff_workspace, -1.0, gamma_old, GAMMA));

    calculate_forces(workspaces->df, workspaces->du, workspaces->dC, D_PREV, M, G, workspaces->U_ext, workspaces->gamma_diff_workspace);
    calculate_ldot_inplace(L_PREV, l, GAMMA, physics_config.getLambdaDimensionless(), physics_config.TAU, workspaces->ldot_curr_workspace, workspaces->impedance_curr_workspace);

    // Move
    particle_manager.moveLocalParticlesFromSolution({.dC = workspaces->dC, .f = workspaces->df, .u = workspaces->du});

    // update phi_prev
    gradient(GAMMA, PHI);

    std::swap(workspaces->ldot_curr_workspace, workspaces->ldot_prev);

    // Update constraints with current solution values
    updateConstraintsFromSolution(all_constraints_set, GAMMA, PHI);

    // log substep
    // particle_logger.log(particle_manager.local_particles);
    // constraint_logger.log(all_constraints_set);

    constraint_iterations++;
    exchangeGhostParticles();
  }

  last_recursive_iteration = constraint_iterations;

  switch (solver_state) {
    case SolverState::CONVERGED:
      PetscPrintf(PETSC_COMM_WORLD, "\n  \033[92mConverged in %d iterations\033[0m | Residual: %4.2e | Overlap: %8.2e | BBPGD Iters: %4lld", constraint_iterations, res, max_overlap, bbpgd_iterations);
      break;
    case SolverState::TOO_MANY_CONSTRAINTS:
      PetscPrintf(PETSC_COMM_WORLD, "\n  \033[91mToo many constraints: %ld | Constraints per particle: %4.2f\033[0m | Residual: %4.2e | Overlap: %8.2e | BBPGD Iters: %4lld", globalReduce<size_t>(all_constraints_set.size(), MPI_SUM), globalReduce<size_t>(all_constraints_set.size(), MPI_SUM) / (double)global_num_particles, res, max_overlap, bbpgd_iterations);
      break;
    case SolverState::NOT_CONVERGED:
      PetscPrintf(PETSC_COMM_WORLD, "\n  \033[91mDid not converge in %d iterations\033[0m | Residual: %4.2e | Overlap: %8.2e | BBPGD Iters: %4lld", constraint_iterations, res, max_overlap, bbpgd_iterations);
      break;
    case SolverState::RUNNING:
      PetscPrintf(PETSC_COMM_WORLD, "\n  \033[91mRunning\033[0m | Residual: %4.2e | Overlap: %8.2e | BBPGD Iters: %4lld", res, max_overlap, bbpgd_iterations);
      break;
  }

  particle_manager.growLocalParticlesFromSolution({.dL = workspaces->ldot_prev, .impedance = workspaces->impedance_curr_workspace});

  return {.constraints = all_constraints_set, .constraint_iterations = constraint_iterations, .bbpgd_iterations = bbpgd_iterations, .residual = res, .max_overlap = max_overlap};
}

void PhysicsEngine::updateCollisionDetectorBounds(const std::array<double, 3>& min_bounds, const std::array<double, 3>& max_bounds) {
  collision_detector.updateBounds(min_bounds, max_bounds);
}

SpatialGrid PhysicsEngine::getCollisionDetectorSpatialGrid() {
  return collision_detector.getSpatialGrid();
}