#include "PhysicsEngine.h"

#include <petsc.h>

#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>

#include "Constraint.h"
#include "Physics.h"
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

  // Finalize assembly
  VecAssemblyBegin(PHI_PREV);
  VecAssemblyEnd(PHI_PREV);
  VecAssemblyBegin(GAMMA_PREV);
  VecAssemblyEnd(GAMMA_PREV);
  MatAssemblyBegin(D_PREV, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(D_PREV, MAT_FINAL_ASSEMBLY);
  MatAssemblyBegin(L_PREV, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(L_PREV, MAT_FINAL_ASSEMBLY);

  // Update sizes
  PHI_PREV.updateSize(global_new_constraints_count);
  GAMMA_PREV.updateSize(global_new_constraints_count);
  D_PREV.updateSize(global_new_constraints_count);
  L_PREV.updateSize(global_new_constraints_count);
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
  auto map_func = [](PetscInt i, PetscScalar grad_i, PetscScalar gamma_i) -> double {
    const double g_i = PetscRealPart(grad_i);
    const double proj_g_i = (PetscRealPart(gamma_i) > 0)
                                ? g_i
                                : std::min(0.0, g_i);

    return std::abs(proj_g_i);
  };

  auto res = reduceVector<double>(gradient_val, map_func, MPI_MAX, gamma);

  if (res == std::numeric_limits<double>::lowest()) {
    res = 0.0;
  }

  return res;
}

namespace {
struct GradientCalculator {
  const PhysicsEngine& physics_engine;  // For config access
  double dt;
  const MatWrapper& D_PREV;
  const MatWrapper& M;
  const MatWrapper& L_PREV;
  const VecWrapper& U_ext;
  const VecWrapper& l;
  const VecWrapper& GAMMA_PREV;
  const VecWrapper& ldot_prev;
  const VecWrapper& PHI_PREV;

  // Workspaces
  VecWrapper& gamma_diff_workspace;
  VecWrapper& phi_dot_movement_workspace;
  VecWrapper& F_g_workspace;
  VecWrapper& U_c_workspace;
  VecWrapper& U_total_workspace;
  VecWrapper& ldot_curr_workspace;
  VecWrapper& impedance_curr_workspace;
  VecWrapper& ldot_diff_workspace;
  VecWrapper& phi_dot_growth_result;

  void operator()(const VecWrapper& gamma_curr, VecWrapper& phi_next_out) const {
    // --- MOVEMENT PART ---
    // gamma_diff = gamma_curr - GAMMA_PREV
    VecWAXPY(gamma_diff_workspace, -1.0, GAMMA_PREV, gamma_curr);

    // phi_dot_movement = D * M * D^T * gamma_diff
    estimate_phi_dot_movement_inplace(D_PREV, M, U_ext, gamma_diff_workspace, F_g_workspace, U_c_workspace, U_total_workspace, phi_dot_movement_workspace);

    // --- GROWTH PART ---
    // ldot_curr = growth_rate(gamma_curr)
    calculate_ldot_inplace(L_PREV, l, gamma_curr, physics_engine.physics_config.getLambdaDimensionless(), physics_engine.physics_config.TAU, ldot_curr_workspace, impedance_curr_workspace);

    // ldot_diff = ldot_curr - ldot_prev
    VecWAXPY(ldot_diff_workspace, -1.0, ldot_prev, ldot_curr_workspace);

    // phi_dot_growth = -L^T * ldot_diff
    estimate_phi_dot_growth_inplace(L_PREV, ldot_diff_workspace, phi_dot_growth_result);

    // Start with the base violation: phi_next_out = PHI_PREV
    PetscCallAbort(PETSC_COMM_WORLD, VecCopy(PHI_PREV, phi_next_out));

    // Add movement term: phi_next_out += dt * phi_dot_movement
    PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(phi_next_out, dt, phi_dot_movement_workspace));

    // Add growth term: phi_next_out -= dt * phi_dot_growth
    PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(phi_next_out, -dt, phi_dot_growth_result));
  }
};

struct RecursiveSolverWorkspaces {
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

void initializeWorkspaces(const MatWrapper& D_PREV, const MatWrapper& M, const MatWrapper& G, const MatWrapper& L_PREV, const VecWrapper& GAMMA_PREV, const VecWrapper& PHI_PREV, const VecWrapper& l, RecursiveSolverWorkspaces& workspaces) {
  // movement
  workspaces.gamma_diff_workspace = VecWrapper::Like(GAMMA_PREV);
  workspaces.phi_dot_movement_workspace = VecWrapper::Like(PHI_PREV);
  workspaces.F_g_workspace = VecWrapper::FromMat(D_PREV);
  workspaces.U_c_workspace = VecWrapper::FromMat(M);
  workspaces.U_total_workspace = VecWrapper::FromMat(D_PREV);

  // growth
  workspaces.phi_dot_growth_result = VecWrapper::FromMatRows(L_PREV);

  workspaces.ldot_curr_workspace = VecWrapper::Like(l);
  workspaces.ldot_diff_workspace = VecWrapper::Like(l);
  workspaces.ldot_prev = VecWrapper::Like(l);
  VecZeroEntries(workspaces.ldot_prev);

  workspaces.impedance_curr_workspace = VecWrapper::Like(l);

  workspaces.U_ext = VecWrapper::FromMat(M);
  workspaces.F_ext_workspace = VecWrapper::FromMat(M);

  // force

  workspaces.df = VecWrapper::FromMat(D_PREV);
  workspaces.du = VecWrapper::FromMat(M);
  workspaces.dC = VecWrapper::FromMat(G);
}
}  // namespace

PhysicsEngine::SolverSolution PhysicsEngine::solveConstraintsRecursiveConstraints(ParticleManager& particle_manager, double dt, int iter, std::function<void()> exchangeGhostParticles) {
  MatWrapper M = calculate_MobilityMatrix(particle_manager.local_particles, physics_config.xi);
  MatWrapper G = calculate_QuaternionMap(particle_manager.local_particles);

  DynamicVecWrapper GAMMA_PREV = DynamicVecWrapper(solver_config);
  DynamicVecWrapper PHI_PREV = DynamicVecWrapper(solver_config);
  DynamicMatWrapper D_PREV = DynamicMatWrapper(solver_config, particle_manager.local_particles.size() * 6);
  DynamicMatWrapper L_PREV = DynamicMatWrapper(solver_config, particle_manager.local_particles.size());

  RecursiveSolverWorkspaces workspaces;
  bool workspaces_initialized = false;

  bool force_vectors_initialized = false;

  // Create length mapping for consistent indexing
  VecWrapper l = getLengthVector(particle_manager.local_particles);

  std::vector<Constraint> all_constraints;
  int constraint_iterations = 0;
  long long bbpgd_iterations = 0;

  double res = 0;
  long long bbpgd_iterations_this_step = 0;

  bool converged = false;
  while (constraint_iterations < solver_config.max_recursive_iterations) {
    exchangeGhostParticles();
    auto new_constraints = collision_detector.detectCollisions(
        particle_manager, constraint_iterations);

    all_constraints.insert(all_constraints.end(), new_constraints.begin(), new_constraints.end());

    auto global_total_constraints = globalReduce<size_t>(all_constraints.size(), MPI_SUM);
    auto global_new_constraints_count = globalReduce<size_t>(new_constraints.size(), MPI_SUM);

    // Calculate matrices only for the newly identified constraints
    updateMatrices(PHI_PREV, GAMMA_PREV, D_PREV, L_PREV, new_constraints, global_new_constraints_count);

    auto max_overlap = -reduceVector<double>(PHI_PREV, [](PetscInt i, PetscScalar v) -> double { return PetscRealPart(v); }, MPI_MIN);
    auto violated_count = reduceVector<int>(PHI_PREV, [](PetscInt i, PetscScalar v) -> int { return (PetscRealPart(v) < 0) ? 1 : 0; }, MPI_SUM);

    double logged_overlap = std::max(0.0, max_overlap);
    PetscPrintf(PETSC_COMM_WORLD, "\n  RecurIter: %2d | Constraints: %4ld (New: %3ld, Violated: %3d) | Overlap: %8.2e | Residual: %8.2e | BBPGD Iters: %5lld",
                constraint_iterations, global_total_constraints, global_new_constraints_count, violated_count, logged_overlap, res, bbpgd_iterations_this_step);

    converged = max_overlap < solver_config.tolerance && constraint_iterations > 0;
    if (converged) {
      break;
    }

    bool local_need_to_recreate_workspaces = !workspaces_initialized || vecSize(GAMMA_PREV) != vecSize(workspaces.gamma_diff_workspace);

    int global_recreate_flag = globalReduce(local_need_to_recreate_workspaces ? 1 : 0, MPI_MAX);

    if (global_recreate_flag) {
      initializeWorkspaces(D_PREV, M, G, L_PREV, GAMMA_PREV, PHI_PREV, l, workspaces);
      workspaces_initialized = true;
    }

    calculate_external_velocities(workspaces.U_ext, workspaces.F_ext_workspace, particle_manager.local_particles, M, dt);

    GradientCalculator gradient_calculator{*this, dt, D_PREV, M, L_PREV, workspaces.U_ext, l, GAMMA_PREV, workspaces.ldot_prev, PHI_PREV, workspaces.gamma_diff_workspace, workspaces.phi_dot_movement_workspace, workspaces.F_g_workspace, workspaces.U_c_workspace, workspaces.U_total_workspace, workspaces.ldot_curr_workspace, workspaces.impedance_curr_workspace, workspaces.ldot_diff_workspace, workspaces.phi_dot_growth_result};

    VecWrapper GAMMA_SOLVED = VecWrapper::Like(GAMMA_PREV);
    GAMMA_PREV.copyTo(GAMMA_SOLVED);

    auto bbpgd_result_recursive = BBPGD(gradient_calculator, residual, GAMMA_SOLVED, solver_config);

    res = bbpgd_result_recursive.residual;
    bbpgd_iterations_this_step = bbpgd_result_recursive.bbpgd_iterations;
    bbpgd_iterations += bbpgd_iterations_this_step;

    PetscCallAbort(PETSC_COMM_WORLD, VecWAXPY(workspaces.gamma_diff_workspace, -1.0, GAMMA_PREV, GAMMA_SOLVED));

    calculate_forces(workspaces.df, workspaces.du, workspaces.dC, D_PREV, M, G, workspaces.U_ext, workspaces.gamma_diff_workspace);

    calculate_ldot_inplace(L_PREV, l, GAMMA_SOLVED, physics_config.getLambdaDimensionless(), physics_config.TAU, workspaces.ldot_curr_workspace, workspaces.impedance_curr_workspace);

    particle_manager.moveLocalParticlesFromSolution({.dC = workspaces.dC, .f = workspaces.df, .u = workspaces.du});

    gradient_calculator(GAMMA_SOLVED, PHI_PREV);

    PetscCallAbort(PETSC_COMM_WORLD, VecCopy(GAMMA_SOLVED, GAMMA_PREV));

    std::swap(workspaces.ldot_curr_workspace, workspaces.ldot_prev);

    constraint_iterations++;
  }

  if (converged) {
    PetscPrintf(PETSC_COMM_WORLD, "\n  Converged in %d iterations | Residual: %4.2e\n", constraint_iterations, res);
  } else {
    PetscPrintf(PETSC_COMM_WORLD, "\n  Did not converge in %d iterations | Residual: %4.2e\n", constraint_iterations, res);
  }

  particle_manager.growLocalParticlesFromSolution({.dL = workspaces.ldot_prev, .impedance = workspaces.impedance_curr_workspace});

  return {.constraints = all_constraints, .constraint_iterations = constraint_iterations, .bbpgd_iterations = bbpgd_iterations, .residual = res};
}

void PhysicsEngine::updateCollisionDetectorBounds(const std::array<double, 3>& min_bounds, const std::array<double, 3>& max_bounds) {
  collision_detector.updateBounds(min_bounds, max_bounds);
}

SpatialGrid PhysicsEngine::getCollisionDetectorSpatialGrid() {
  return collision_detector.getSpatialGrid();
}