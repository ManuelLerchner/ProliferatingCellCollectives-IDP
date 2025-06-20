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

  MatWrapper D = calculate_Jacobian(local_constraints, local_particles, mappings.velocityL2GMap, mappings.constraintL2GMap);
  // PetscPrintf(PETSC_COMM_WORLD, "D Matrix:\n");
  // MatView(D.get(), PETSC_VIEWER_STDOUT_WORLD);

  MatWrapper M = calculate_MobilityMatrix(local_particles, local_num_bodies, physics_config.xi, mappings.velocityL2GMap);
  // PetscPrintf(PETSC_COMM_WORLD, "Mobility Matrix M:\n");
  // MatView(M.get(), PETSC_VIEWER_STDOUT_WORLD);

  MatWrapper G = calculate_QuaternionMap(local_particles, mappings.configL2GMap, mappings.velocityL2GMap);
  // PetscPrintf(PETSC_COMM_WORLD, "Quaternion Map G:\n");
  // MatView(G.get(), PETSC_VIEWER_STDOUT_WORLD);

  VecWrapper phi = create_phi_vector(local_constraints, mappings.constraintL2GMap);
  // PetscPrintf(PETSC_COMM_WORLD, "Phi Vector:\n");
  // VecView(phi.get(), PETSC_VIEWER_STDOUT_WORLD);

  auto length_map = create_length_map(local_particles);
  MatWrapper S = calculate_stress_matrix(local_constraints, local_particles, length_map, mappings.constraintL2GMap);
  // PetscPrintf(PETSC_COMM_WORLD, "Stress Matrix S:\n");
  // MatView(S.get(), PETSC_VIEWER_STDOUT_WORLD);

  return {.D = std::move(D), .M = std::move(M), .G = std::move(G), .S = std::move(S), .phi = std::move(phi)};
}

VecWrapper getLengthVector(const std::vector<Particle>& local_particles, ISLocalToGlobalMapping length_map) {
  // l is a (global_num_particles, 1) vector
  VecWrapper l;
  VecCreate(PETSC_COMM_WORLD, l.get_ref());
  VecSetSizes(l, local_particles.size(), PETSC_DETERMINE);
  VecSetFromOptions(l);

  // Set the local-to-global mapping to ensure consistency with other vectors
  VecSetLocalToGlobalMapping(l, length_map);

  // Set values using local indices - PETSc will handle the global mapping
  for (int i = 0; i < local_particles.size(); i++) {
    PetscCallAbort(PETSC_COMM_WORLD, VecSetValueLocal(l, i, local_particles[i].getLength(), INSERT_VALUES));
  }

  VecAssemblyBegin(l);
  VecAssemblyEnd(l);

  return l;
}

void estimate_phi_dot_movement_inplace(
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

  // create a workspace for phi_dot and t1 and t2
  VecWrapper phi_dot;
  VecWrapper t1_workspace;
  VecWrapper t2_workspace;
  PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(matrices.phi.get(), phi_dot.get_ref()));
  PetscCallAbort(PETSC_COMM_WORLD, MatCreateVecs(matrices.D.get(), NULL, t1_workspace.get_ref()));
  PetscCallAbort(PETSC_COMM_WORLD, MatCreateVecs(matrices.M.get(), NULL, t2_workspace.get_ref()));

  auto gradient = [&](const VecWrapper& gamma, VecWrapper& phi_next_out) {
    estimate_phi_dot_movement_inplace(matrices.D, matrices.M, gamma, t1_workspace, t2_workspace, phi_dot);

    // phi_next = phi + dt * phi_dot
    PetscCallAbort(PETSC_COMM_WORLD, VecCopy(matrices.phi.get(), phi_next_out.get()));
    PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(phi_next_out.get(), dt, phi_dot.get()));
  };

  VecWrapper gamma0;
  PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(matrices.phi.get(), gamma0.get_ref()));
  PetscCallAbort(PETSC_COMM_WORLD, VecZeroEntries(gamma0.get()));

  auto [gamma, bbpgd_iterations, res] = BBPGD(gradient, residual, gamma0, solver_config);

  auto [F, U, deltaC] = calculate_forces(matrices.D, matrices.M, matrices.G, gamma);

  PhysicsEngine::MovementSolution solution = {.dC = std::move(deltaC), .f = std::move(F), .u = std::move(U)};
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

  MatWrapper L_PREV;
  PetscCallAbort(PETSC_COMM_WORLD, MatCreate(PETSC_COMM_WORLD, L_PREV.get_ref()));
  PetscCallAbort(PETSC_COMM_WORLD, MatSetSizes(L_PREV, particle_manager.local_particles.size(), 0, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCallAbort(PETSC_COMM_WORLD, MatSetType(L_PREV, MATMPIAIJ));
  PetscCallAbort(PETSC_COMM_WORLD, MatSetFromOptions(L_PREV));
  PetscCallAbort(PETSC_COMM_WORLD, MatAssemblyBegin(L_PREV, MAT_FINAL_ASSEMBLY));
  PetscCallAbort(PETSC_COMM_WORLD, MatAssemblyEnd(L_PREV, MAT_FINAL_ASSEMBLY));

  VecWrapper gamma_diff_workspace;
  VecWrapper phi_dot_movement_workspace;
  VecWrapper phi_dot_growth_result;

  VecWrapper t1_workspace;
  VecWrapper t2_workspace;

  VecWrapper ldot_curr_workspace;
  VecWrapper ldot_diff_workspace;
  VecWrapper ldot_prev;

  VecWrapper impedance_curr_workspace;

  // Create length mapping for consistent indexing
  auto length_map = create_length_map(particle_manager.local_particles);
  VecWrapper l = getLengthVector(particle_manager.local_particles, length_map.get());
  VecWrapper l_current;

  VecDuplicate(l.get(), l_current.get_ref());
  VecDuplicate(l.get(), ldot_curr_workspace.get_ref());
  VecDuplicate(l.get(), ldot_diff_workspace.get_ref());
  VecDuplicate(l.get(), ldot_prev.get_ref());
  VecDuplicate(l.get(), impedance_curr_workspace.get_ref());

  std::unordered_set<Constraint, ConstraintHash, ConstraintEqual> all_constraints;
  int constraint_iterations = 0;
  long long bbpgd_iterations = 0;

  double res = std::numeric_limits<double>::infinity();

  const double LAMBDA_DIMENSIONLESS = (physics_config.TAU / (physics_config.xi * physics_config.l0 * physics_config.l0)) * physics_config.LAMBDA;

  bool converged = false;
  while (constraint_iterations < solver_config.max_recursive_iterations) {
    std::vector<Constraint> new_constraints = particle_manager.constraint_generator->generateConstraints(
        particle_manager.local_particles, all_constraints, constraint_iterations);

    // Add the new constraints to the master set
    all_constraints.insert(new_constraints.begin(), new_constraints.end());

    // Calculate matrices only for the newly identified constraints
    auto matrices = calculateMatrices(particle_manager.local_particles, new_constraints);

    // stack PHI with matrices.phi
    PHI_PREV = concatenateVectors(PHI_PREV, matrices.phi);

    // get maximal overlap
    double min_seperation;
    PetscCallAbort(PETSC_COMM_WORLD, VecMin(PHI_PREV.get(), NULL, &min_seperation));
    double max_overlap = -min_seperation;
    converged = max_overlap < solver_config.tolerance && constraint_iterations > 0;

    PetscInt total_constraints = all_constraints.size();

    // calculate total memory usage
    PetscLogDouble total_memory;
    PetscCallAbort(PETSC_COMM_WORLD, PetscMemoryGetCurrentUsage(&total_memory));

    std::vector<Constraint> all_constraints_vec(all_constraints.begin(), all_constraints.end());
    PetscInt local_violated_constraint_count = std::count_if(all_constraints_vec.begin(), all_constraints_vec.end(), [](const Constraint& constraint) { return constraint.violated; });
    PetscInt global_violated_constraint_count;
    MPI_Allreduce(&local_violated_constraint_count, &global_violated_constraint_count, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD);

    MPI_Allreduce(&total_constraints, &total_constraints, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD);
    PetscPrintf(PETSC_COMM_WORLD, "\n  Recursive Constraint Iteration: %d | Constraints: %4d | Violated: %4d | Memory: %4.2f MB | Residual: %4.2e", constraint_iterations, total_constraints, global_violated_constraint_count, total_memory / 1024 / 1024, max_overlap);

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

    bool recreate_workspace = constraint_iterations == 0;
    if (not recreate_workspace) {
      PetscInt prev_size, curr_size;
      VecGetSize(gamma_diff_workspace, &prev_size);
      VecGetSize(GAMMA_PREV, &curr_size);
      recreate_workspace = prev_size != curr_size;
    }

    if (recreate_workspace) {
      VecDestroy(gamma_diff_workspace.get_ref());
      VecDestroy(phi_dot_movement_workspace.get_ref());
      VecDestroy(t1_workspace.get_ref());
      VecDestroy(t2_workspace.get_ref());
      VecDestroy(phi_dot_growth_result.get_ref());

      // movement
      VecDuplicate(GAMMA_PREV.get(), gamma_diff_workspace.get_ref());
      VecDuplicate(PHI_PREV.get(), phi_dot_movement_workspace.get_ref());
      MatCreateVecs(D_PREV.get(), NULL, t1_workspace.get_ref());
      MatCreateVecs(matrices.M.get(), NULL, t2_workspace.get_ref());

      // growth
      MatCreateVecs(L_PREV.get(), phi_dot_growth_result.get_ref(), NULL);
    }

    const double epsilon_movement = 1e-4;  // For the D*M*D^T part
    const double epsilon_growth = 1e-4;    // For the growth coupling part
    auto gradient = [&](const VecWrapper& gamma_curr, VecWrapper& phi_next_out) {
      // --- MOVEMENT PART ---
      // gamma_diff = gamma_curr - GAMMA_PREV
      VecCopy(gamma_curr.get(), gamma_diff_workspace.get());
      VecAXPY(gamma_diff_workspace.get(), -1.0, GAMMA_PREV.get());

      // phi_dot_movement = D * M * D^T * gamma_diff
      estimate_phi_dot_movement_inplace(D_PREV, matrices.M, gamma_diff_workspace,
                                        t1_workspace, t2_workspace, phi_dot_movement_workspace);

      // --- GROWTH PART ---
      // ldot_curr = growth_rate(gamma_curr)
      calculate_ldot(L_PREV, l, gamma_curr, LAMBDA_DIMENSIONLESS, physics_config.TAU, ldot_curr_workspace, impedance_curr_workspace);

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

      // Add the combined regularization term.
      // This is mathematically equivalent to regularizing both parts separately.
      // It adds dt * (epsilon_movement + epsilon_growth) * gamma_diff to phi_next_out
      const double total_epsilon = epsilon_movement + epsilon_growth;
      PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(phi_next_out.get(), dt * total_epsilon, gamma_diff_workspace.get()));
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
    auto [df, du, dC] = calculate_forces(D_PREV, matrices.M, matrices.G, gamma_diff);

    calculate_ldot(L_PREV, l, GAMMA_NEXT, LAMBDA_DIMENSIONLESS, physics_config.TAU, ldot_curr_workspace, impedance_curr_workspace);

    // prepare for next iteration
    particle_manager.moveLocalParticlesFromSolution({.dC = dC, .f = df, .u = du});

    gradient(GAMMA_NEXT, PHI_PREV);
    GAMMA_PREV = std::move(GAMMA_NEXT);
    VecCopy(ldot_curr_workspace.get(), ldot_prev.get());

    constraint_iterations++;
  }

  if (converged) {
    PetscPrintf(PETSC_COMM_WORLD, "\n  Converged in %d iterations | Residual: %4.2e\n", constraint_iterations, res);
  } else {
    PetscPrintf(PETSC_COMM_WORLD, "\n  Did not converge in %d iterations | Residual: %4.2e\n", constraint_iterations, res);
  }

  particle_manager.growLocalParticlesFromSolution({.dL = ldot_curr_workspace, .impedance = impedance_curr_workspace});

  return {.constraints = std::vector<Constraint>(all_constraints.begin(), all_constraints.end()), .constraint_iterations = constraint_iterations, .bbpgd_iterations = bbpgd_iterations, .residual = res};
}
