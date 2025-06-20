#include "ParticleManager.h"

#include <petsc.h>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <unordered_map>
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

    for (int i = 0; i < local_particles.size(); i++) {
      local_particles[i].setLocalID(i);
    }
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
      local_particles[i].eulerStepPosition(dC_array, i, dt);
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

void ParticleManager::moveLocalParticlesFromSolution(const PhysicsEngine::MovementSolution& solution) {
  double dt = physics_engine->solver_config.dt;

  for (auto& p : local_particles) {
    PetscInt gid = p.getGID();
    PetscInt base_idx = gid * Particle::getStateSize();
    std::vector<PetscInt> indices(Particle::getStateSize());
    std::iota(indices.begin(), indices.end(), base_idx);

    std::vector<PetscScalar> dC_values(Particle::getStateSize());
    std::vector<PetscScalar> f_values(Particle::getStateSize());
    std::vector<PetscScalar> u_values(Particle::getStateSize());

    VecGetValues(solution.dC.get(), Particle::getStateSize(), indices.data(), dC_values.data());
    VecGetValues(solution.f.get(), Particle::getStateSize(), indices.data(), f_values.data());
    VecGetValues(solution.u.get(), Particle::getStateSize(), indices.data(), u_values.data());

    // Create a dummy gid for the update functions, since they now operate on local data
    PetscInt dummy_gid = 0;
    p.eulerStepPosition(dC_values.data(), dummy_gid, dt);
    p.addForceAndTorque(f_values.data(), u_values.data(), dummy_gid);
  }
}

void ParticleManager::growLocalParticlesFromSolution(const PhysicsEngine::GrowthSolution& solution) {
  const PetscScalar* dL_array;
  const PetscScalar* impedance_array;

  VecGetArrayRead(solution.dL.get(), &dL_array);
  VecGetArrayRead(solution.impedance.get(), &impedance_array);

  for (int i = 0; i < local_particles.size(); i++) {
    double dt = physics_engine->solver_config.dt;
    // Use local array index since PETSc vectors are accessed with local indices
    // The local-to-global mapping is handled internally by PETSc during vector creation
    local_particles[i].eulerStepLength(dL_array, i, dt);

    local_particles[i].setImpedance(impedance_array[i]);
  }

  VecRestoreArrayRead(solution.dL.get(), &dL_array);
  VecRestoreArrayRead(solution.impedance.get(), &impedance_array);
}

void ParticleManager::divideParticles() {
  // divide particles
  for (int i = 0; i < local_particles.size(); i++) {
    if (local_particles[i].getLength() >= 2 * physics_engine->physics_config.l0) {
      // divide particle
      auto new_particle = local_particles[i].divide();
      if (new_particle) {
        queueNewParticle(new_particle.value());
      }
    }
  }
}

void ParticleManager::run(int num_steps) {
  // Print progress information

  for (int i = 0; i < num_steps; i++) {
    // divide particles
    divideParticles();

    commitNewParticles();
    resetLocalParticles();

    // Validate particle IDs after committing new particles
    validateParticleIDs();

    // VTK logging if enabled (initial state with no constraints)
    if (vtk_logger_ && i == 0) {
      PhysicsEngine::SolverSolution solver_solution = {.constraints = {}, .constraint_iterations = 0, .bbpgd_iterations = 0, .residual = 0.0};
      auto sim_state = createSimulationState(solver_solution);
      vtk_logger_->logTimestepComplete(physics_engine->solver_config.dt, sim_state.get());
      printProgress(i, num_steps, std::nullopt);
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

void ParticleManager::printProgress(int current_iteration, int total_iterations, const std::optional<PhysicsEngine::SolverSolution>& solver_solution) const {
  PetscInt global_constraint_count;

  double current_time = current_iteration * physics_engine->solver_config.dt / 60;
  double total_time = total_iterations * physics_engine->solver_config.dt / 60;

  if (solver_solution) {
    PetscInt local_constraint_count = solver_solution.value().constraints.size();
    MPI_Allreduce(&local_constraint_count, &global_constraint_count, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD);

    PetscInt local_violated_constraint_count = std::count_if(solver_solution.value().constraints.begin(), solver_solution.value().constraints.end(), [](const Constraint& constraint) { return constraint.violated; });
    PetscInt global_violated_constraint_count;
    MPI_Allreduce(&local_violated_constraint_count, &global_violated_constraint_count, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD);

    PetscPrintf(PETSC_COMM_WORLD,
                "\rProgress: %4d /%4d (%5.1f%%) | Time: %5.1f min / %5.1f min | Particles: %4d | Constraints: %4d | Violated: %4d |    ",
                current_iteration, total_iterations,
                ((double)(current_iteration) / total_iterations) * 100.0,
                current_time, total_time,
                global_particle_count, global_constraint_count, global_violated_constraint_count);

  } else {
    PetscPrintf(PETSC_COMM_WORLD,
                "\rProgress: %4d /%4d (%5.1f%%) | Time: %5.1f min / %5.1f min | Particles: %4d |    ",
                current_iteration, total_iterations,
                ((double)(current_iteration) / total_iterations) * 100.0,
                current_time, total_time,
                global_particle_count);
  }

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
  state->residual = solver_solution.residual;

  // Set other state values
  state->constraint_iterations = solver_solution.constraint_iterations;
  state->bbpgd_iterations = solver_solution.bbpgd_iterations;
  state->l0 = physics_engine->physics_config.l0;

  return state;
}
