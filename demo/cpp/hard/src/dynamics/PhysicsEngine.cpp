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
  gen = std::mt19937(rd());
}

void updateMatrices(DynamicVecWrapper& PHI_PREV, DynamicVecWrapper& GAMMA_PREV, DynamicMatWrapper& D_PREV, DynamicMatWrapper& L_PREV, const std::vector<Constraint>& new_constraints, PetscMPIInt global_new_constraints_count) {
  // Ensure capacity for new constraints
  PHI_PREV.ensureCapacity(global_new_constraints_count);
  GAMMA_PREV.ensureCapacity(global_new_constraints_count);
  D_PREV.ensureCapacity(global_new_constraints_count);
  L_PREV.ensureCapacity(global_new_constraints_count);

  // Calculate offset for new data
  PetscInt col_offset = PHI_PREV.getSize();
  size_t local_new_constraints_count = new_constraints.size();
  PetscCallAbort(PETSC_COMM_WORLD, MPI_Exscan(&local_new_constraints_count, &col_offset, 1, MPI_INT, MPI_SUM, PETSC_COMM_WORLD));

  // Populate matrices and vectors with new data
  create_phi_vector_local(PHI_PREV, new_constraints, col_offset);

  calculate_jacobian_local(D_PREV, new_constraints, col_offset);

  calculate_stress_matrix_local(L_PREV, new_constraints, col_offset);

  // Update sizes
  PHI_PREV.updateSize(global_new_constraints_count);
  GAMMA_PREV.updateSize(global_new_constraints_count);
  D_PREV.updateSize(global_new_constraints_count);
  L_PREV.updateSize(global_new_constraints_count);

  // Finalize assembly
  VecAssemblyBegin(PHI_PREV);
  VecAssemblyBegin(GAMMA_PREV);
  MatAssemblyBegin(D_PREV, MAT_FINAL_ASSEMBLY);
  MatAssemblyBegin(L_PREV, MAT_FINAL_ASSEMBLY);

  VecAssemblyEnd(PHI_PREV);
  VecAssemblyEnd(GAMMA_PREV);
  MatAssemblyEnd(D_PREV, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(L_PREV, MAT_FINAL_ASSEMBLY);
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

void PhysicsEngine::calculate_external_velocities(VecWrapper& U_ext, VecWrapper& F_ext_workspace, const std::vector<Particle>& local_particles, const MatWrapper& M, double dt) {
  // Create a vector for external forces
  PetscCallAbort(PETSC_COMM_WORLD, VecSet(F_ext_workspace, 0.0));

  // Add gravity
  if (physics_config.gravity.x != 0.0 || physics_config.gravity.y != 0.0 || physics_config.gravity.z != 0.0) {
    for (size_t i = 0; i < local_particles.size(); ++i) {
      PetscInt idx[3] = {static_cast<PetscInt>(i * 6 + 0), static_cast<PetscInt>(i * 6 + 1), static_cast<PetscInt>(i * 6 + 2)};
      auto [grav_force_x, grav_force_y, grav_force_z] = local_particles[i].calculateGravitationalForce({physics_config.gravity.x, physics_config.gravity.y, physics_config.gravity.z});
      PetscScalar vals[3] = {grav_force_x, grav_force_y, grav_force_z};
      PetscCallAbort(PETSC_COMM_WORLD, VecSetValuesLocal(F_ext_workspace, 3, idx, vals, ADD_VALUES));
    }
    PetscCallAbort(PETSC_COMM_WORLD, VecAssemblyBegin(F_ext_workspace));
    PetscCallAbort(PETSC_COMM_WORLD, VecAssemblyEnd(F_ext_workspace));
  }
  // U_ext = M * F_ext
  PetscCallAbort(PETSC_COMM_WORLD, MatMult(M, F_ext_workspace, U_ext));

  // Add Brownian motion
  if (physics_config.temperature > 0) {
    PetscScalar* u_ext_array;
    PetscCallAbort(PETSC_COMM_WORLD, VecGetArray(U_ext, &u_ext_array));

    std::normal_distribution<double> dist(0.0, 1.0);

    for (size_t i = 0; i < local_particles.size(); ++i) {
      auto brownian_vel = local_particles[i].calculateBrownianVelocity(physics_config.temperature, physics_config.xi, dt, dist, gen);
      for (int j = 0; j < 6; ++j) {
        u_ext_array[i * 6 + j] += brownian_vel[j];
      }
    }

    PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArray(U_ext, &u_ext_array));
  }

  if (physics_config.monolayer) {
    PetscScalar* u_array;
    PetscCallAbort(PETSC_COMM_WORLD, VecGetArray(U_ext, &u_array));

    for (int i = 0; i < local_particles.size(); i++) {
      u_array[i * 6 + 2] = 0;  // vz
      u_array[i * 6 + 3] = 0;  // omegax
      u_array[i * 6 + 4] = 0;  // omegay
    }

    PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArray(U_ext, &u_array));
  }
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

double residual(const VecWrapper& gradient_val, const VecWrapper& gamma) {
  // This function computes the infinity norm of the projected gradient.
  // The projection is defined as:
  //   projected_gradient_i = gradient_val_i, if gamma_i > 0
  //   projected_gradient_i = min(0, gradient_val_i), if gamma_i <= 0

  Vec projected_gradient;
  PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(gradient_val, &projected_gradient));

  // Get local size and pointers to vector data
  PetscInt n_local;
  const PetscScalar *grad_array, *gamma_array;
  PetscScalar* pg_array;
  PetscCallAbort(PETSC_COMM_WORLD, VecGetLocalSize(gradient_val, &n_local));
  PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(gradient_val, &grad_array));
  PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(gamma, &gamma_array));
  PetscCallAbort(PETSC_COMM_WORLD, VecGetArray(projected_gradient, &pg_array));

  // Compute projected gradient component-wise
  for (PetscInt i = 0; i < n_local; ++i) {
    const double g_i = PetscRealPart(grad_array[i]);
    if (PetscRealPart(gamma_array[i]) > 0) {
      pg_array[i] = g_i;
    } else {
      pg_array[i] = std::min(0.0, g_i);
    }
  }

  // Restore arrays
  PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(gradient_val, &grad_array));
  PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(gamma, &gamma_array));
  PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArray(projected_gradient, &pg_array));

  // Compute the infinity norm of the projected gradient: max(abs(projected_gradient_i))
  PetscReal norm_val;
  PetscCallAbort(PETSC_COMM_WORLD, VecNorm(projected_gradient, NORM_INFINITY, &norm_val));

  // Clean up
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&projected_gradient));

  return (double)norm_val;
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

struct pair_hash {
  template <class T1, class T2>
  std::size_t operator()(const std::pair<T1, T2>& p) const {
    auto h1 = std::hash<T1>{}(p.first);
    auto h2 = std::hash<T2>{}(p.second);

    return h1 ^ h2;
  }
};

PhysicsEngine::SolverSolution PhysicsEngine::solveConstraintsRecursiveConstraints(ParticleManager& particle_manager, double dt, int iter, std::function<void()> exchangeGhostParticles, vtk::ParticleLogger& particle_logger, vtk::ConstraintLogger& constraint_logger) {
  MatWrapper M = calculate_MobilityMatrix(particle_manager.local_particles, physics_config.xi);
  MatWrapper G = calculate_QuaternionMap(particle_manager.local_particles);

  int global_num_particles = globalReduce<int>(particle_manager.local_particles.size(), MPI_SUM);
  int alloc_size = solver_config.getMinPreallocationSize(global_num_particles);

  DynamicVecWrapper GAMMA = DynamicVecWrapper(alloc_size, solver_config.growth_factor);
  DynamicVecWrapper PHI = DynamicVecWrapper(alloc_size, solver_config.growth_factor);
  DynamicMatWrapper D_PREV = DynamicMatWrapper(particle_manager.local_particles.size() * 6, alloc_size, solver_config.growth_factor);
  DynamicMatWrapper L_PREV = DynamicMatWrapper(particle_manager.local_particles.size(), alloc_size, solver_config.growth_factor);

  std::optional<RecursiveSolverWorkspaces> workspaces;

  // Create length mapping for consistent indexing
  VecWrapper l = getLengthVector(particle_manager.local_particles);

  std::vector<Constraint> all_constraints;
  int constraint_iterations = 0;
  long long bbpgd_iterations = 0;

  double res = 0;
  double max_overlap = 0;
  long long bbpgd_iterations_this_step = 0;

  // hash map counting constraints betwen pairs of particles
  std::unordered_map<std::pair<int, int>, int, pair_hash> global_constraint_count;

  SolverState solver_state = SolverState::RUNNING;
  while (constraint_iterations < solver_config.max_recursive_iterations) {
    exchangeGhostParticles();

    double future_colission_tolerance = constraint_iterations == 0 ? 0.1 : 0;
    auto new_constraints = collision_detector.detectCollisions(
        particle_manager, constraint_iterations, future_colission_tolerance);

    // Filter constraints to include at most `max_constraints_per_pair` constraints per particle pair.
    std::vector<Constraint> filtered_constraints;
    for (const auto& constraint : new_constraints) {
      auto pair = std::make_pair(constraint.gidI, constraint.gidJ);
      if (pair.first > pair.second) std::swap(pair.first, pair.second);
      global_constraint_count[pair]++;
      if (global_constraint_count[pair] <= solver_config.max_constraints_per_pair) {
        filtered_constraints.push_back(constraint);
      } else {
        solver_state = SolverState::TOO_MANY_CONSTRAINTS;
      }
    }

    all_constraints.insert(all_constraints.end(), filtered_constraints.begin(), filtered_constraints.end());

    auto global_total_constraints = globalReduce<size_t>(all_constraints.size(), MPI_SUM);
    auto global_new_constraints_count = globalReduce<size_t>(filtered_constraints.size(), MPI_SUM);

    // Calculate matrices only for the newly identified constraints
    updateMatrices(PHI, GAMMA, D_PREV, L_PREV, filtered_constraints, global_new_constraints_count);

    double min;
    VecMin(PHI, NULL, &min);

    // Logging
    max_overlap = -min;

    // memory
    PetscLogDouble memory_usage;
    PetscCallAbort(PETSC_COMM_WORLD, PetscMemoryGetCurrentUsage(&memory_usage));

    double logged_overlap = std::max(0.0, max_overlap);
    PetscPrintf(PETSC_COMM_WORLD, "\r  Solver Iteration: %4d | Constraints: %6ld (New: %3ld) | Overlap: %8.2e | Residual: %8.2e | BBPGD Iters: %4lld | Memory: %4.2f MB",
                constraint_iterations, global_total_constraints, global_new_constraints_count, logged_overlap, res, bbpgd_iterations_this_step, memory_usage / 1024 / 1024);

    // Check convergence
    if (max_overlap < 5 * solver_config.tolerance && constraint_iterations > 0) {
      solver_state = SolverState::CONVERGED;
      break;
    }

    // Recreate workspaces if needed
    bool local_need_to_recreate_workspaces = !workspaces.has_value() || vecSize(GAMMA) != vecSize(workspaces->gamma_diff_workspace);
    int global_recreate_flag = globalReduce(local_need_to_recreate_workspaces ? 1 : 0, MPI_MAX);

    if (global_recreate_flag) {
      workspaces.emplace(D_PREV, M, G, L_PREV, GAMMA, PHI, l);
    }

    // initialize solver

    VecWrapper gamma_old = VecWrapper::Like(GAMMA);
    GAMMA.copyTo(gamma_old);

    // Calculate external velocities
    calculate_external_velocities(workspaces->U_ext, workspaces->F_ext_workspace, particle_manager.local_particles, M, dt);

    // Gradient
    auto gradient = [&](const VecWrapper& gamma_curr, VecWrapper& phi_next_out) {
      // --- MOVEMENT PART ---
      // gamma_diff = gamma_curr - gamma_old
      VecWAXPY(workspaces->gamma_diff_workspace, -1.0, gamma_old, gamma_curr);

      // phi_dot_movement = D * M * D^T * gamma_diff
      estimate_phi_dot_movement_inplace(D_PREV, M, workspaces->U_ext, workspaces->gamma_diff_workspace, workspaces->F_g_workspace, workspaces->U_c_workspace, workspaces->U_total_workspace, workspaces->phi_dot_movement_workspace);

      // --- GROWTH PART ---
      // ldot_curr = growth_rate(gamma_curr)
      calculate_ldot_inplace(L_PREV, l, gamma_curr, physics_config.getLambdaDimensionless(), physics_config.TAU, workspaces->ldot_curr_workspace, workspaces->impedance_curr_workspace);

      // ldot_diff = ldot_curr - ldot_prev
      VecWAXPY(workspaces->ldot_diff_workspace, -1.0, workspaces->ldot_prev, workspaces->ldot_curr_workspace);

      // phi_dot_growth = -L^T * ldot_diff
      estimate_phi_dot_growth_inplace(L_PREV, workspaces->ldot_diff_workspace, workspaces->phi_dot_growth_result);

      // Start with the base violation: phi_next_out = phi
      PetscCallAbort(PETSC_COMM_WORLD, VecCopy(PHI, phi_next_out));

      // Add movement term: phi_next_out += dt * phi_dot_movement
      PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(phi_next_out, dt, workspaces->phi_dot_movement_workspace));

      // Add growth term: phi_next_out -= dt * phi_dot_growth
      PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(phi_next_out, -dt, workspaces->phi_dot_growth_result));
    };

    // Solver
    auto bbpgd_result_recursive = BBPGD(gradient, residual, GAMMA, solver_config);

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

    // log substep
    particle_logger.log(particle_manager.local_particles);
    constraint_logger.log(all_constraints);

    constraint_iterations++;
  }

  switch (solver_state) {
    case SolverState::CONVERGED:
      PetscPrintf(PETSC_COMM_WORLD, "\n  \033[92mConverged in %d iterations\033[0m | Residual: %4.2e | Overlap: %8.2e", constraint_iterations, res, max_overlap);
      break;
    case SolverState::TOO_MANY_CONSTRAINTS:
      PetscPrintf(PETSC_COMM_WORLD, "\n  \033[91mToo many constraints: %ld | Constraints per particle: %4.2f\033[0m | Residual: %4.2e | Overlap: %8.2e", globalReduce<size_t>(all_constraints.size(), MPI_SUM), globalReduce<size_t>(all_constraints.size(), MPI_SUM) / (double)global_num_particles, res, max_overlap);
      break;
    case SolverState::NOT_CONVERGED:
      PetscPrintf(PETSC_COMM_WORLD, "\n  \033[91mDid not converge in %d iterations\033[0m | Residual: %4.2e | Overlap: %8.2e", constraint_iterations, res, max_overlap);
      break;
    case SolverState::RUNNING:
      PetscPrintf(PETSC_COMM_WORLD, "\n  \033[91mRunning\033[0m | Residual: %4.2e | Overlap: %8.2e", res, max_overlap);
      break;
  }

  particle_manager.growLocalParticlesFromSolution({.dL = workspaces->ldot_prev, .impedance = workspaces->impedance_curr_workspace});

  return {.constraints = all_constraints, .constraint_iterations = constraint_iterations, .bbpgd_iterations = bbpgd_iterations, .residual = res};
}

void PhysicsEngine::updateCollisionDetectorBounds(const std::array<double, 3>& min_bounds, const std::array<double, 3>& max_bounds) {
  collision_detector.updateBounds(min_bounds, max_bounds);
}

SpatialGrid PhysicsEngine::getCollisionDetectorSpatialGrid() {
  return collision_detector.getSpatialGrid();
}