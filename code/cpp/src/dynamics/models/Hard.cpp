#include "Hard.h"

#include "dynamics/Physics.h"
#include "hard/HardModelGradient.h"
#include "hard/Workspace.h"
#include "solver/BBPGD.h"
#include "util/ArrayMath.h"
#include "util/PetscRaii.h"

auto createMatrices(const std::vector<Constraint>& all_constraints, ParticleManager& particle_manager, SimulationParameters& params) {
  auto global_max_constraints = globalReduce<size_t>(all_constraints.size(), MPI_MAX);

  MatWrapper M = calculate_MobilityMatrix(particle_manager.local_particles, params.physics_config.xi);
  MatWrapper G = calculate_QuaternionMap(particle_manager.local_particles);
  VecWrapper l = getLengthVector(particle_manager.local_particles);
  VecWrapper ldot = getLdotVector(particle_manager.local_particles);
  VecWrapper PHI = VecWrapper::Create(global_max_constraints);
  VecWrapper GAMMA = VecWrapper::Create(global_max_constraints);
  MatWrapper D_PREV = MatWrapper::CreateAIJ(global_max_constraints, 6 * particle_manager.local_particles.size());
  MatWrapper L_PREV = MatWrapper::CreateAIJ(global_max_constraints, particle_manager.local_particles.size());

  PetscInt ownership_start, ownership_end;
  PetscCallAbort(PETSC_COMM_WORLD, MatGetOwnershipRange(D_PREV, &ownership_start, &ownership_end));

  // Calculate offset for new data
  PetscInt col_offset = ownership_start;

  create_phi_vector_local(PHI, all_constraints, col_offset);
  create_gamma_vector_local(GAMMA, all_constraints, col_offset);
  calculate_jacobian_local(D_PREV, all_constraints, col_offset);
  calculate_stress_matrix_local(L_PREV, all_constraints, col_offset);

  VecAssemblyEnd(PHI);
  VecAssemblyEnd(GAMMA);
  MatAssemblyEnd(D_PREV, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(L_PREV, MAT_FINAL_ASSEMBLY);

  return std::make_tuple(
      std::move(M),
      std::move(G),
      std::move(l),
      std::move(ldot),
      std::move(GAMMA),
      std::move(PHI),
      std::move(D_PREV),
      std::move(L_PREV));
}

void updateConstraintsFromSolution(std::vector<Constraint>& constraints, const VecWrapper& gamma, const VecWrapper& phi) {
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

ParticleManager::SolverSolution solveHardModel(ParticleManager& particle_manager, CollisionDetector& collision_detector, SimulationParameters& params, double dt, int iter, std::function<void()> exchangeGhostParticles, vtk::ParticleLogger& particle_logger, vtk::ConstraintLogger& constraint_logger) {
  int constraint_iterations = 0;
  size_t total_bbpgd_iterations = 0;

  double res = 0.0;
  double max_overlap = 0.0;

  // Reset collision detector state
  collision_detector.reset();

  std::vector<Constraint> all_constraints_set;

  for (auto& p : particle_manager.local_particles) {
    p.reset();
  }

  std::optional<std::shared_ptr<vtk::BBPGDLogger>> bbpgd_logger;

  if (iter % 10 == 0 && params.sim_config.log_bbgd_trace) {
    bbpgd_logger = std::make_shared<vtk::BBPGDLogger>("logs", "bbpgdtrace", true);
  }

  while (constraint_iterations < params.solver_config.max_recursive_iterations) {
    if (bbpgd_logger.has_value()) {
      (*bbpgd_logger)->set_recursive_iteration(constraint_iterations);
    }

    // Use a larger tolerance for initial collision detection
    exchangeGhostParticles();

    auto new_constraints = collision_detector.detectCollisions(particle_manager, constraint_iterations, constraint_iterations == 0 ? .25 : 0.0);

    all_constraints_set.insert(all_constraints_set.end(), new_constraints.begin(), new_constraints.end());

    max_overlap = globalReduce(std::accumulate(all_constraints_set.begin(), all_constraints_set.end(), 0.0,
                                               [](double acc, const Constraint& c) {
                                                 return std::max(acc, c.signed_distance < 0 ? -c.signed_distance : 0);
                                               }),
                               MPI_MAX);

    // Check convergence
    if (max_overlap <= params.solver_config.tolerance && constraint_iterations > 0) {
      break;
    }

    auto [M, G, l, ldot, GAMMA, PHI, D_PREV, L_PREV] = createMatrices(all_constraints_set, particle_manager, params);

    // Calculate external velocities

    VecWrapper U_ext = VecWrapper::Create(6 * particle_manager.local_particles.size());

    if (constraint_iterations == 0) {
      calculate_external_velocities(U_ext, particle_manager.local_particles, M, dt, constraint_iterations, params.physics_config);
    }

    VecWrapper gamma_old = VecWrapper::Like(GAMMA);
    VecCopy(GAMMA, gamma_old);

    // Create gradient object
    HardModelGradient hardgradient(D_PREV, M, L_PREV, U_ext, PHI, gamma_old, l, ldot, params, dt);

    // Solver
    auto bbpgd_result_recursive = BBPGD(hardgradient, GAMMA, params.solver_config.tolerance, params.solver_config.max_bbpgd_iterations, bbpgd_logger);

    res = bbpgd_result_recursive.residual;
    total_bbpgd_iterations += bbpgd_result_recursive.bbpgd_iterations;

    auto& workspace = hardgradient.workspaces_;

    VecWrapper dC = VecWrapper::FromMat(G);
    MatMult(G, workspace.U_total_workspace, dC);

    // Move
    particle_manager.moveLocalParticlesFromSolution({.dC = dC, .f = workspace.F_g_workspace, .u = workspace.U_total_workspace}, dt);

    // Grow
    particle_manager.setGrowParamsFromSolution({.dL = workspace.ldot_curr_workspace, .impedance = workspace.impedance_curr_workspace, .stress = workspace.stress_curr_workspace});

    // Update constraints with current solution values
    updateConstraintsFromSolution(all_constraints_set, GAMMA, workspace.phi_next_out);

    constraint_iterations++;

    // Logging
    double logged_overlap = std::max(0.0, max_overlap);

    PetscLogDouble memory_usage;
    PetscCallAbort(PETSC_COMM_WORLD, PetscMemoryGetCurrentUsage(&memory_usage));
  }

  if (bbpgd_logger.has_value()) {
    (*bbpgd_logger)->log();
  }

  particle_manager.grow(dt);

  if (constraint_iterations == params.solver_config.max_recursive_iterations) {
    PetscPrintf(PETSC_COMM_WORLD, "\n  Warning: Maximum number of constraint iterations reached (%ld). Solution may not be fully converged.\n", params.solver_config.max_recursive_iterations);
  }

  return {.constraints = all_constraints_set, .constraint_iterations = constraint_iterations, .bbpgd_iterations = total_bbpgd_iterations, .residual = res, .max_overlap = max_overlap};
}