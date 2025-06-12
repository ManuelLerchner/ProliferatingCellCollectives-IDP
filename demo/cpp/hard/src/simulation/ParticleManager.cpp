#include "ParticleManager.h"

#include <petsc.h>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

#include "dynamics/Constraint.h"
#include "dynamics/Physics.h"
#include "dynamics/PhysicsEngine.h"
#include "logger/VTK.h"
#include "spatial/ConstraintGenerator.h"

ParticleManager::ParticleManager(PhysicsConfig physics_config, SolverConfig solver_config) {
  constraint_generator = std::make_unique<ConstraintGenerator>(solver_config.tolerance, 2.5 * physics_config.l0);
  physics_engine = std::make_unique<PhysicsEngine>(physics_config, solver_config);

  vtk_logger_ = vtk::createParticleLogger("./vtk_output");
  constraint_loggers_ = vtk::createConstraintLogger("./vtk_output");
}

void ParticleManager::queueNewParticle(Particle p) {
  new_particle_buffer.push_back(p);
}

void ParticleManager::commitNewParticles() {
  PetscInt num_to_add_local = new_particle_buffer.size();

  PetscInt first_id_for_this_rank;
  MPI_Scan(&num_to_add_local, &first_id_for_this_rank, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD);
  first_id_for_this_rank -= num_to_add_local;

  first_id_for_this_rank += this->global_particle_count;

  std::vector<PetscInt> new_ids(num_to_add_local);
  std::iota(new_ids.begin(), new_ids.end(), first_id_for_this_rank);
  for (PetscInt i = 0; i < num_to_add_local; ++i) {
    new_particle_buffer[i].setGID(new_ids[i]);
  }

  local_particles.insert(
      local_particles.end(),
      std::make_move_iterator(new_particle_buffer.begin()),
      std::make_move_iterator(new_particle_buffer.end()));

  // Sort particles by ID to maintain ordering across all ranks
  if (new_particle_buffer.size() > 0) {
    std::sort(local_particles.begin(), local_particles.end(),
              [](const Particle& a, const Particle& b) { return a.setGID() < b.setGID(); });
  }

  new_particle_buffer.clear();

  PetscInt total_added_this_step;
  MPI_Allreduce(&num_to_add_local, &total_added_this_step, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD);

  this->global_particle_count += total_added_this_step;
}

void ParticleManager::eulerStepfromSolution(const VecWrapper& dC) {
  // Get local portion of dC vector to update particles
  const PetscScalar* dC_array;

  VecGetArrayRead(dC.get(), &dC_array);

  const PetscScalar* gamma_array;
  PetscInt local_size;
  VecGetLocalSize(dC.get(), &local_size);

  for (int i = 0; i < local_particles.size(); i++) {
    int base_offset = i * Particle::getStateSize();
    if (base_offset + Particle::getStateSize() <= local_size) {
      // Update particle state using helper function

      double dt = physics_engine->solver_config.dt;
      local_particles[i].eulerStep(dC_array, i, dt);
    }
  }

  VecRestoreArrayRead(dC.get(), &dC_array);
}

void ParticleManager::resetLocalParticles() {
  const PetscScalar* gamma_array;
  PetscInt local_size;

  for (int i = 0; i < local_particles.size(); i++) {
    local_particles[i].clearForceAndTorque();
  }
}

void ParticleManager::moveLocalParticlesFromSolution(const PhysicsEngine::PhysicsSolution& solution) {
  // Get local portion of dC vector to update particles
  const PetscScalar* dC_array;
  const PetscScalar* df_array;
  const PetscScalar* dU_array;

  VecGetArrayRead(solution.deltaC.get(), &dC_array);
  VecGetArrayRead(solution.f.get(), &df_array);
  VecGetArrayRead(solution.u.get(), &dU_array);

  const PetscScalar* gamma_array;
  PetscInt local_size;
  VecGetLocalSize(solution.deltaC.get(), &local_size);

  for (int i = 0; i < local_particles.size(); i++) {
    int base_offset = i * Particle::getStateSize();
    if (base_offset + Particle::getStateSize() <= local_size) {
      // Update particle state using helper function

      double dt = physics_engine->solver_config.dt;
      local_particles[i].eulerStep(dC_array, i, dt);

      local_particles[i].addForceAndTorque(df_array, dU_array, i);
    }
  }

  VecRestoreArrayRead(solution.deltaC.get(), &dC_array);
  VecRestoreArrayRead(solution.f.get(), &df_array);
  VecRestoreArrayRead(solution.u.get(), &dU_array);
}

void ParticleManager::run(int num_steps) {
  // Print progress information

  for (int i = 0; i < num_steps; i++) {
    commitNewParticles();
    resetLocalParticles();

    // Validate particle IDs after committing new particles
    validateParticleIDs();

    // VTK logging if enabled (initial state with no constraints)
    if (vtk_logger_ && i == 0) {
      PhysicsEngine::SolverSolution solver_solution = {.constraints = {}, .constraint_iterations = 0, .bbpgd_iterations = 0, .residum = 0.0};
      auto sim_state = createSimulationState(solver_solution);
      vtk_logger_->logTimestepComplete(physics_engine->solver_config.dt, sim_state.get());
      printProgress(i, num_steps, {.constraints = {}, .constraint_iterations = 0, .bbpgd_iterations = 0, .residum = 0.0});
    }

    // auto solver_solution = physics_engine->solveConstraintsSingleConstraint(*this, physics_engine->solver_config.dt);
    auto solver_solution = physics_engine->solveConstraintsRecursiveConstraints(*this, physics_engine->solver_config.dt);

    if (vtk_logger_) {
      auto sim_state = createSimulationState(solver_solution);

      vtk_logger_->logTimestepComplete(physics_engine->solver_config.dt, sim_state.get());
      constraint_loggers_->logTimestepComplete(physics_engine->solver_config.dt, sim_state.get());
      printProgress(i + 1, num_steps, solver_solution);
    }

    // Print progress information
  }
  PetscPrintf(PETSC_COMM_WORLD, "\n");
}

void ParticleManager::validateParticleIDs() const {
  PetscMPIInt rank, size;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  MPI_Comm_size(PETSC_COMM_WORLD, &size);

  // Check local sorting
  for (size_t i = 1; i < local_particles.size(); ++i) {
    if (local_particles[i - 1].setGID() >= local_particles[i].setGID()) {
      PetscPrintf(PETSC_COMM_WORLD, "ERROR: Local particles not sorted at rank %d: ID[%zu]=%d >= ID[%zu]=%d\n",
                  rank, i - 1, local_particles[i - 1].setGID(), i, local_particles[i].setGID());
    }
  }

  // Gather all IDs to check global uniqueness (only do this for small numbers of particles)
  if (global_particle_count < 10000) {  // Only for reasonable sizes
    std::vector<PetscInt> local_ids;
    for (const auto& p : local_particles) {
      local_ids.push_back(p.setGID());
    }

    PetscInt local_count = local_ids.size();
    std::vector<PetscInt> all_counts(size);
    MPI_Allgather(&local_count, 1, MPIU_INT, all_counts.data(), 1, MPIU_INT, PETSC_COMM_WORLD);

    std::vector<PetscInt> displacements(size);
    displacements[0] = 0;
    for (int i = 1; i < size; ++i) {
      displacements[i] = displacements[i - 1] + all_counts[i - 1];
    }

    std::vector<PetscInt> all_ids(global_particle_count);
    MPI_Allgatherv(local_ids.data(), local_count, MPIU_INT,
                   all_ids.data(), all_counts.data(), displacements.data(), MPIU_INT, PETSC_COMM_WORLD);

    // Check for duplicates on rank 0
    if (rank == 0) {
      std::sort(all_ids.begin(), all_ids.end());
      for (size_t i = 1; i < all_ids.size(); ++i) {
        if (all_ids[i - 1] == all_ids[i]) {
          PetscPrintf(PETSC_COMM_WORLD, "ERROR: Duplicate global ID found: %d\n", all_ids[i]);
        }
      }
    }
  }
}

void ParticleManager::printProgress(int current_iteration, int total_iterations, const PhysicsEngine::SolverSolution& solver_solution) const {
  PetscInt global_constraint_count;

  PetscInt local_constraint_count = solver_solution.constraints.size();
  MPI_Allreduce(&local_constraint_count, &global_constraint_count, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD);

  PetscInt local_violated_constraint_count = std::count_if(solver_solution.constraints.begin(), solver_solution.constraints.end(), [](const Constraint& constraint) { return constraint.violated; });
  PetscInt global_violated_constraint_count;
  MPI_Allreduce(&local_violated_constraint_count, &global_violated_constraint_count, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD);

  double current_time = current_iteration * physics_engine->solver_config.dt / 60;
  double total_time = total_iterations * physics_engine->solver_config.dt / 60;

  PetscPrintf(PETSC_COMM_WORLD,
              "\rProgress: %4d /%4d (%5.1f%%) | Time: %5.1f min / %5.1f min | Particles: %4d | Constraints: %4d | Violated: %4d | Overlap: %4f |    ",
              current_iteration, total_iterations,
              ((double)(current_iteration) / total_iterations) * 100.0,
              current_time, total_time,
              global_particle_count, global_constraint_count, global_violated_constraint_count, solver_solution.residum);

  // Flush output to ensure immediate display
  fflush(stdout);
}

std::unique_ptr<vtk::ParticleSimulationState> ParticleManager::createSimulationState(
    const PhysicsEngine::SolverSolution& solver_solution) const {
  auto state = std::make_unique<vtk::ParticleSimulationState>();

  // Copy particles
  state->particles = local_particles;
  state->constraints = std::move(solver_solution.constraints);

  // Initialize empty force/torque vectors (would be populated by physics engine in full implementation)
  state->forces.resize(local_particles.size());
  state->torques.resize(local_particles.size());

  for (int i = 0; i < local_particles.size(); i++) {
    state->forces[i] = local_particles[i].getForce();
    state->torques[i] = local_particles[i].getTorque();
  }

  // Calculate max overlap from constraints
  state->residum = solver_solution.residum;

  // Set other state values
  state->constraint_iterations = solver_solution.constraint_iterations;
  state->avg_bbpgd_iterations = static_cast<double>(solver_solution.bbpgd_iterations) / solver_solution.constraint_iterations;
  state->l0 = 1.0;

  return state;
}
