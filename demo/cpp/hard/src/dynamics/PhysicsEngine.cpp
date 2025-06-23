#include "PhysicsEngine.h"

#include <petsc.h>

#include <cassert>
#include <cmath>
#include <iostream>

#include "Constraint.h"
#include "Physics.h"
#include "simulation/Particle.h"
#include "simulation/ParticleManager.h"
#include "solver/BBPGD.h"
#include "util/PetscRaii.h"
#include "util/PetscUtil.h"

constexpr int PARTICLE_DOFS = 6;

PhysicsEngine::PhysicsEngine(PhysicsConfig physics_config, SolverConfig solver_config) : physics_config(physics_config), solver_config(solver_config) {
  std::random_device rd;
  gen = std::mt19937(rd());
}

PhysicsEngine::PhysicsMatrices PhysicsEngine::calculateMatrices(const std::vector<Particle>& local_particles, const std::vector<Constraint>& local_constraints) {
  MatWrapper D = calculate_Jacobian(local_constraints, local_particles);
  // PetscPrintf(PETSC_COMM_WORLD, "D Matrix:\n");
  // MatView(D.get(), PETSC_VIEWER_STDOUT_WORLD);

  MatWrapper M = calculate_MobilityMatrix(local_particles, physics_config.xi);
  // PetscPrintf(PETSC_COMM_WORLD, "Mobility Matrix M:\n");
  // MatView(M.get(), PETSC_VIEWER_STDOUT_WORLD);

  MatWrapper G = calculate_QuaternionMap(local_particles);
  // PetscPrintf(PETSC_COMM_WORLD, "Quaternion Map G:\n");
  // MatView(G.get(), PETSC_VIEWER_STDOUT_WORLD);

  VecWrapper phi = create_phi_vector(local_constraints);
  // PetscPrintf(PETSC_COMM_WORLD, "Phi Vector:\n");
  // VecView(phi.get(), PETSC_VIEWER_STDOUT_WORLD);

  MatWrapper S = calculate_stress_matrix(local_constraints, local_particles);
  // PetscPrintf(PETSC_COMM_WORLD, "Stress Matrix S:\n");
  // MatView(S.get(), PETSC_VIEWER_STDOUT_WORLD);

  MatAssemblyEnd(D, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(G, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(S, MAT_FINAL_ASSEMBLY);
  VecAssemblyEnd(phi);

  // --- DEBUG: Assert column counts ---
  auto assert_column_counts = [](const MatWrapper& mat, const char* name, PetscInt expected_nnz) {
    MatWrapper mat_T;
    PetscCallAbort(PETSC_COMM_WORLD, MatTranspose(mat.get(), MAT_INITIAL_MATRIX, mat_T.get_ref()));

    PetscInt r_start, r_end;
    PetscCallAbort(PETSC_COMM_WORLD, MatGetOwnershipRange(mat_T.get(), &r_start, &r_end));

    for (PetscInt i = r_start; i < r_end; ++i) {
      PetscInt nnz;
      const PetscInt* cols;
      const PetscScalar* vals;
      PetscCallAbort(PETSC_COMM_WORLD, MatGetRow(mat_T.get(), i, &nnz, &cols, &vals));
      if (nnz != expected_nnz) {
        PetscPrintf(PETSC_COMM_WORLD, "Assertion failed for matrix %s: column %d has %d non-zeros, expected %d\n", name, i, nnz, expected_nnz);
      }
      assert(nnz == expected_nnz);
      PetscCallAbort(PETSC_COMM_WORLD, MatRestoreRow(mat_T.get(), i, &nnz, &cols, &vals));
    }
  };

  assert_column_counts(D, "D", 12);
  assert_column_counts(S, "S", 2);

  return {.D = std::move(D), .M = std::move(M), .G = std::move(G), .S = std::move(S), .phi = std::move(phi)};
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

void PhysicsEngine::apply_monolayer_constraints(VecWrapper& U, int n_local) {
  if (!physics_config.monolayer) {
    return;
  }

  PetscScalar* u_array;
  PetscCallAbort(PETSC_COMM_WORLD, VecGetArray(U.get(), &u_array));

#pragma omp parallel for
  for (int i = 0; i < n_local; i++) {
    u_array[i * PARTICLE_DOFS + 2] = 0;  // vz
    u_array[i * PARTICLE_DOFS + 3] = 0;  // omegax
    u_array[i * PARTICLE_DOFS + 4] = 0;  // omegay
  }

  PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArray(U.get(), &u_array));
}

void PhysicsEngine::calculate_external_velocities(VecWrapper& U_ext, VecWrapper& F_ext_workspace, const std::vector<Particle>& local_particles, const MatWrapper& M, double dt) {
  // Create a vector for external forces
  PetscCallAbort(PETSC_COMM_WORLD, VecSet(F_ext_workspace.get(), 0.0));

  // Add gravity
  if (physics_config.gravity.x != 0.0 || physics_config.gravity.y != 0.0 || physics_config.gravity.z != 0.0) {
    for (size_t i = 0; i < local_particles.size(); ++i) {
      PetscInt idx[3];
      idx[0] = i * PARTICLE_DOFS + 0;
      idx[1] = i * PARTICLE_DOFS + 1;
      idx[2] = i * PARTICLE_DOFS + 2;
      PetscScalar vals[3];
      vals[0] = local_particles[i].getVolume() * physics_config.gravity.x;
      vals[1] = local_particles[i].getVolume() * physics_config.gravity.y;
      vals[2] = local_particles[i].getVolume() * physics_config.gravity.z;
      PetscCallAbort(PETSC_COMM_WORLD, VecSetValuesLocal(F_ext_workspace.get(), 3, idx, vals, ADD_VALUES));
    }
    PetscCallAbort(PETSC_COMM_WORLD, VecAssemblyBegin(F_ext_workspace.get()));
    PetscCallAbort(PETSC_COMM_WORLD, VecAssemblyEnd(F_ext_workspace.get()));
  }

  // U_ext = M * F_ext
  PetscCallAbort(PETSC_COMM_WORLD, MatMult(M.get(), F_ext_workspace.get(), U_ext.get()));

  // Add Brownian motion
  if (physics_config.temperature > 0) {
    PetscScalar* u_ext_array;
    PetscCallAbort(PETSC_COMM_WORLD, VecGetArray(U_ext.get(), &u_ext_array));

    std::normal_distribution<double> dist(0.0, 1.0);
    const double k_B = 1;

#pragma omp parallel for
    for (size_t i = 0; i < local_particles.size(); ++i) {
      const auto& p = local_particles[i];
      const double L = p.getLength();
      const double r = p.getDiameter() / 2.0;
      const double aspect_ratio = L / p.getDiameter();
      const double log_p = log(aspect_ratio);

      // Slender body theory geometric factors (assuming viscosity mu=1.0)
      const double geom_para = (2 * M_PI * L) / (log_p + 0.20);
      const double geom_perp = (4 * M_PI * L) / (log_p + 0.84);
      const double geom_rot = (M_PI * L * L * L) / (3 * (log_p - 0.66));

      // Total drag, scaled by the friction coefficient
      const double gamma_para = physics_config.xi * geom_para;
      const double gamma_perp = physics_config.xi * geom_perp;
      const double gamma_rot = physics_config.xi * geom_rot;

      // Average translational drag
      const double gamma_trans_avg = (gamma_para + 2 * gamma_perp) / 3.0;

      // Mobilities
      const double mobility_trans = 1.0 / gamma_trans_avg;
      const double mobility_rot = 1.0 / gamma_rot;

      // Brownian velocities
      const double trans_coeff = sqrt(2.0 * k_B * physics_config.temperature * mobility_trans * 1.0 / dt);
      const double rot_coeff = sqrt(2.0 * k_B * physics_config.temperature * mobility_rot * 1.0 / dt);

      // Apply translational Brownian velocity
      u_ext_array[i * PARTICLE_DOFS + 0] += trans_coeff * dist(gen);
      u_ext_array[i * PARTICLE_DOFS + 1] += trans_coeff * dist(gen);
      u_ext_array[i * PARTICLE_DOFS + 2] += trans_coeff * dist(gen);

      // Apply rotational Brownian velocity
      u_ext_array[i * PARTICLE_DOFS + 3] += rot_coeff * dist(gen);
      u_ext_array[i * PARTICLE_DOFS + 4] += rot_coeff * dist(gen);
      u_ext_array[i * PARTICLE_DOFS + 5] += rot_coeff * dist(gen);
    }

    PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArray(U_ext.get(), &u_ext_array));
  }

  apply_monolayer_constraints(U_ext, local_particles.size());
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
  PetscCallAbort(PETSC_COMM_WORLD, MatMult(D.get(), gamma.get(), F_g.get()));

  // Step 2: U_c = M * F_g
  PetscCallAbort(PETSC_COMM_WORLD, MatMult(M.get(), F_g.get(), U_c.get()));

  // Step 3: U_total = U_known + U_c
  PetscCallAbort(PETSC_COMM_WORLD, VecCopy(U_c.get(), U_total.get()));
  PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(U_total.get(), 1.0, U_known.get()));

  // Step 4: phi_dot_out = D^T * U_total
  PetscCallAbort(PETSC_COMM_WORLD, MatMultTranspose(D.get(), U_total.get(), phi_dot_out.get()));

  // No return, the result is in phi_dot_out.
}

void calculate_impedance_vector(const VecWrapper& stresses, double lambda, VecWrapper& impedance_out) {
  // I is a (num_bodies, 1) vector
  // I = np.exp(-lamb * stress_matrix)

  VecCopy(stresses, impedance_out);

  VecScale(impedance_out, -lambda);
  VecExp(impedance_out);
}

void calculate_growth_rate_vector(const VecWrapper& l, const VecWrapper& sigma, double lambda, double tau, VecWrapper& growth_rates_out, VecWrapper& impedance_out) {
  // growth rate is a (num_bodies, 1) vector
  // growth_rate = l / tau * I

  VecCopy(l, growth_rates_out);

  calculate_impedance_vector(sigma, lambda, impedance_out);

  VecPointwiseMult(growth_rates_out, l, impedance_out);
  VecScale(growth_rates_out, 1 / tau);
}

void calculate_ldot(const MatWrapper& L, const VecWrapper& l, const VecWrapper& gamma, double lambda, double tau, VecWrapper& ldot_curr_out, VecWrapper& impedance_curr_out) {
  VecWrapper sigma;
  PetscCallAbort(PETSC_COMM_WORLD, MatCreateVecs(L.get(), NULL, sigma.get_ref()));
  PetscCallAbort(PETSC_COMM_WORLD, MatMult(L.get(), gamma.get(), sigma.get()));

  calculate_growth_rate_vector(l, sigma, lambda, tau, ldot_curr_out, impedance_curr_out);
};

void estimate_phi_dot_growth_inplace(
    const MatWrapper& Sigma,
    const VecWrapper& ldot,
    VecWrapper& phi_dot_growth_result) {
  PetscCallAbort(PETSC_COMM_WORLD, MatMultTranspose(Sigma.get(), ldot.get(), phi_dot_growth_result.get()));
  PetscCallAbort(PETSC_COMM_WORLD, VecScale(phi_dot_growth_result.get(), -1.0));
}

void calculate_forces(VecWrapper& F, VecWrapper& U, VecWrapper& deltaC, const MatWrapper& D, const MatWrapper& M, const MatWrapper& G, const VecWrapper& U_ext, const VecWrapper& gamma) {
  // f = D @ gamma
  PetscCallAbort(PETSC_COMM_WORLD, MatMult(D, gamma.get(), F.get()));

  // U = M @ f
  PetscCallAbort(PETSC_COMM_WORLD, MatMult(M, F.get(), U.get()));

  // U = U + U_ext
  PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(U.get(), 1.0, U_ext.get()));

  // deltaC = G @ U
  PetscCallAbort(PETSC_COMM_WORLD, MatMult(G, U.get(), deltaC.get()));
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
    VecCopy(gamma_curr.get(), gamma_diff_workspace.get());
    VecAXPY(gamma_diff_workspace.get(), -1.0, GAMMA_PREV.get());

    // phi_dot_movement = D * M * D^T * gamma_diff
    estimate_phi_dot_movement_inplace(D_PREV, M, U_ext, gamma_diff_workspace, F_g_workspace, U_c_workspace, U_total_workspace, phi_dot_movement_workspace);

    // --- GROWTH PART ---
    // ldot_curr = growth_rate(gamma_curr)
    calculate_ldot(L_PREV, l, gamma_curr, physics_engine.physics_config.getLambdaDimensionless(), physics_engine.physics_config.TAU, ldot_curr_workspace, impedance_curr_workspace);

    // ldot_diff = ldot_curr - ldot_prev
    VecCopy(ldot_curr_workspace.get(), ldot_diff_workspace.get());
    VecAXPY(ldot_diff_workspace.get(), -1.0, ldot_prev.get());

    // phi_dot_growth = -L^T * ldot_diff
    estimate_phi_dot_growth_inplace(L_PREV, ldot_diff_workspace, phi_dot_growth_result);

    // --- COMBINE AND REGULARIZE ---
    // Start with the base violation: phi_next_out = PHI_PREV
    PetscCallAbort(PETSC_COMM_WORLD, VecCopy(PHI_PREV.get(), phi_next_out.get()));

    // Add movement term: phi_next_out += dt * phi_dot_movement
    PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(phi_next_out.get(), dt, phi_dot_movement_workspace.get()));

    // Add growth term: phi_next_out += dt * phi_dot_growth
    PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(phi_next_out.get(), dt, phi_dot_growth_result.get()));
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
  VecZeroEntries(workspaces.ldot_prev.get());

  workspaces.impedance_curr_workspace = VecWrapper::Like(l);

  workspaces.U_ext = VecWrapper::FromMat(M);
  workspaces.F_ext_workspace = VecWrapper::FromMat(M);

  // force

  workspaces.df = VecWrapper::FromMat(D_PREV);
  workspaces.du = VecWrapper::FromMat(M);
  workspaces.dC = VecWrapper::FromMat(G);
}
}  // namespace

PhysicsEngine::SolverSolution PhysicsEngine::solveConstraintsSingleConstraint(ParticleManager& particle_manager, double dt) {
  std::unordered_set<Constraint, ConstraintHash, ConstraintEqual> all_constraints;
  auto local_constraints = particle_manager.constraint_generator->generateConstraints(particle_manager.local_particles, all_constraints, 0);
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

  // --- External Velocities ---
  VecWrapper U_ext;
  PetscCallAbort(PETSC_COMM_WORLD, MatCreateVecs(matrices.M.get(), NULL, U_ext.get_ref()));
  VecWrapper F_ext_workspace;
  PetscCallAbort(PETSC_COMM_WORLD, MatCreateVecs(matrices.M.get(), NULL, F_ext_workspace.get_ref()));
  calculate_external_velocities(U_ext, F_ext_workspace, particle_manager.local_particles, matrices.M, dt);

  // create a workspace for phi_dot and t1 and t2
  auto phi_dot = VecWrapper::Like(matrices.phi);
  auto F_g = VecWrapper::FromMat(matrices.D);
  auto U_c = VecWrapper::FromMat(matrices.M);
  auto U_total = VecWrapper::FromMat(matrices.D);

  auto gradient = [&](const VecWrapper& gamma, VecWrapper& phi_next_out) {
    estimate_phi_dot_movement_inplace(matrices.D, matrices.M, U_ext, gamma, F_g, U_c, U_total, phi_dot);

    // phi_next = phi + dt * phi_dot
    PetscCallAbort(PETSC_COMM_WORLD, VecCopy(matrices.phi.get(), phi_next_out.get()));
    PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(phi_next_out.get(), dt, phi_dot.get()));
  };

  auto gamma0 = VecWrapper::Like(matrices.phi);
  PetscCallAbort(PETSC_COMM_WORLD, VecZeroEntries(gamma0.get()));

  auto [gamma, bbpgd_iterations, res] = BBPGD(gradient, residual, gamma0, solver_config);

  auto F = VecWrapper::FromMat(matrices.D);
  auto U = VecWrapper::FromMat(matrices.M);
  auto deltaC = VecWrapper::FromMat(matrices.G);
  calculate_forces(F, U, deltaC, matrices.D, matrices.M, matrices.G, U_ext, gamma);

  PhysicsEngine::MovementSolution solution = {.dC = deltaC, .f = F, .u = U};
  particle_manager.moveLocalParticlesFromSolution(solution);

  return {.constraints = local_constraints, .constraint_iterations = 1, .bbpgd_iterations = bbpgd_iterations, .residual = res};
}

PhysicsEngine::SolverSolution PhysicsEngine::solveConstraintsRecursiveConstraints(ParticleManager& particle_manager, double dt, int iter) {
  VecWrapper GAMMA_PREV = VecWrapper::CreateEmpty();
  VecWrapper PHI_PREV = VecWrapper::CreateEmpty();
  MatWrapper D_PREV = MatWrapper::CreateEmpty(particle_manager.local_particles.size() * PARTICLE_DOFS);
  MatWrapper L_PREV = MatWrapper::CreateEmpty(particle_manager.local_particles.size());

  RecursiveSolverWorkspaces workspaces;
  bool workspaces_initialized = false;

  bool force_vectors_initialized = false;

  // Create length mapping for consistent indexing
  VecWrapper l = getLengthVector(particle_manager.local_particles);

  std::unordered_set<Constraint, ConstraintHash, ConstraintEqual> all_constraints;
  int constraint_iterations = 0;
  long long bbpgd_iterations = 0;

  double res = 0;
  long long bbpgd_iterations_this_step = 0;

  bool converged = false;
  while (constraint_iterations < solver_config.max_recursive_iterations) {
    auto new_constraints = particle_manager.constraint_generator->generateConstraints(
        particle_manager.local_particles, all_constraints, constraint_iterations);

    // Add the new constraints to the master set
    all_constraints.insert(new_constraints.begin(), new_constraints.end());

    // Calculate matrices only for the newly identified constraints
    auto matrices = calculateMatrices(particle_manager.local_particles, new_constraints);

    // stack PHI with matrices.phi
    PHI_PREV = concatenateVectors(PHI_PREV, matrices.phi);

    double min_seperation;
    PetscCallAbort(PETSC_COMM_WORLD, VecMin(PHI_PREV.get(), NULL, &min_seperation));
    double max_overlap = -min_seperation;

    // Gather statistics for logging
    PetscInt local_total_constraints = all_constraints.size();
    PetscInt local_new_constraints = new_constraints.size();

    PetscLogDouble total_memory;
    PetscCallAbort(PETSC_COMM_WORLD, PetscMemoryGetCurrentUsage(&total_memory));

    // Print the comprehensive log line
    double logged_overlap = std::max(0.0, max_overlap);
    PetscPrintf(PETSC_COMM_WORLD, "\n  RecurIter: %2d | Constraints: %4d (New: %3d, Violated: %3d) | Overlap: %8.2e | Residual: %8.2e | BBPGD Iters: %5lld | Mem: %4.2f MB",
                constraint_iterations, globalReduce(local_total_constraints, MPI_SUM), globalReduce(local_new_constraints, MPI_SUM), reduceVector<PetscInt>(PHI_PREV, [](PetscInt i, PetscScalar v) -> PetscInt { return (PetscRealPart(v) < 0) ? 1 : 0; }, MPI_SUM), logged_overlap, res, bbpgd_iterations_this_step, total_memory / 1024 / 1024);

    converged = max_overlap < solver_config.tolerance && constraint_iterations > 0;
    if (converged) {
      break;
    }

    // pad gamma with 0
    VecWrapper zero_pad;
    PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(matrices.phi, zero_pad.get_ref()));
    PetscCallAbort(PETSC_COMM_WORLD, VecSet(zero_pad, 0.0));
    GAMMA_PREV = concatenateVectors(GAMMA_PREV, zero_pad);

    // stack D with matrices.D
    D_PREV = horizontallyStackMatrices(D_PREV, matrices.D);

    // stack L with matrices.S
    L_PREV = horizontallyStackMatrices(L_PREV, matrices.S);

    bool local_need_to_recreate_workspaces = !workspaces_initialized || vecSize(GAMMA_PREV) != vecSize(workspaces.gamma_diff_workspace);

    // Ensure all ranks agree on whether to recreate workspaces
    int global_recreate_flag = globalReduce(local_need_to_recreate_workspaces ? 1 : 0, MPI_MAX);

    if (global_recreate_flag) {
      initializeWorkspaces(D_PREV, matrices.M, matrices.G, L_PREV, GAMMA_PREV, PHI_PREV, l, workspaces);
      workspaces_initialized = true;
    }

    // --- External Velocities ---
    calculate_external_velocities(workspaces.U_ext, workspaces.F_ext_workspace, particle_manager.local_particles, matrices.M, dt);

    GradientCalculator gradient_calculator{*this, dt, D_PREV, matrices.M, L_PREV, workspaces.U_ext, l, GAMMA_PREV, workspaces.ldot_prev, PHI_PREV, workspaces.gamma_diff_workspace, workspaces.phi_dot_movement_workspace, workspaces.F_g_workspace, workspaces.U_c_workspace, workspaces.U_total_workspace, workspaces.ldot_curr_workspace, workspaces.impedance_curr_workspace, workspaces.ldot_diff_workspace, workspaces.phi_dot_growth_result};

    // solve for gamma
    auto [GAMMA_NEXT, bbpgd_iterations_temp, res_temp] = BBPGD(gradient_calculator, residual, GAMMA_PREV, solver_config);

    res = res_temp;
    bbpgd_iterations_this_step = bbpgd_iterations_temp;
    bbpgd_iterations += bbpgd_iterations_temp;

    VecWrapper gamma_diff = VecWrapper::Like(GAMMA_NEXT);
    PetscCallAbort(PETSC_COMM_WORLD, VecCopy(GAMMA_NEXT.get(), gamma_diff.get()));
    PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(gamma_diff.get(), -1.0, GAMMA_PREV.get()));

    // calculate forces

    calculate_forces(workspaces.df, workspaces.du, workspaces.dC, D_PREV, matrices.M, matrices.G, workspaces.U_ext, gamma_diff);

    calculate_ldot(L_PREV, l, GAMMA_NEXT, physics_config.getLambdaDimensionless(), physics_config.TAU, workspaces.ldot_curr_workspace, workspaces.impedance_curr_workspace);

    // prepare for next iteration
    particle_manager.moveLocalParticlesFromSolution({.dC = workspaces.dC, .f = workspaces.df, .u = workspaces.du});

    gradient_calculator(GAMMA_NEXT, PHI_PREV);
    GAMMA_PREV = std::move(GAMMA_NEXT);
    VecCopy(workspaces.ldot_curr_workspace.get(), workspaces.ldot_prev.get());

    constraint_iterations++;
  }

  if (converged) {
    PetscPrintf(PETSC_COMM_WORLD, "\n  Converged in %d iterations | Residual: %4.2e\n", constraint_iterations, res);
  } else {
    PetscPrintf(PETSC_COMM_WORLD, "\n  Did not converge in %d iterations | Residual: %4.2e\n", constraint_iterations, res);
  }

  particle_manager.growLocalParticlesFromSolution({.dL = workspaces.ldot_curr_workspace, .impedance = workspaces.impedance_curr_workspace});

  return {.constraints = std::vector<Constraint>(all_constraints.begin(), all_constraints.end()), .constraint_iterations = constraint_iterations, .bbpgd_iterations = bbpgd_iterations, .residual = res};
}
