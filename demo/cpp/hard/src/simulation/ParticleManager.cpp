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
  constraint_generator = std::make_unique<ConstraintGenerator>();

  physics_engine = std::make_unique<PhysicsEngine>(physics_config, solver_config);
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

void ParticleManager::updateLocalParticlesFromSolution(const VecWrapper& dC) {
  // Get local portion of dC vector to update particles
  const PetscScalar* dC_array;
  VecGetArrayRead(dC.get(), &dC_array);

  PetscInt local_size;
  VecGetLocalSize(dC.get(), &local_size);

  for (int i = 0; i < local_particles.size(); i++) {
    int base_offset = i * Particle::getStateSize();
    if (base_offset + Particle::getStateSize() <= local_size) {
      // Update particle state using helper function

      local_particles[i].updateState(dC_array, i, physics_engine->solver_config.dt);

      // Normalize quaternion to maintain unit length
      local_particles[i].normalizeQuaternion();
    }
  }

  VecRestoreArrayRead(dC.get(), &dC_array);
}

void ParticleManager::run(int num_steps) {
  for (int i = 0; i < num_steps; i++) {
    commitNewParticles();

    // Validate particle IDs after committing new particles
    validateParticleIDs();

    // VTK logging if enabled
    if (vtk_logger_ && i == 0) {
      auto sim_state = createSimulationState({});
      vtk_logger_->logTimestepComplete(physics_engine->solver_config.dt, sim_state.get());
    }

    // Generate real collision constraints using the collision detection system
    std::vector<Constraint> local_constraints = constraint_generator->generateConstraints(local_particles);

    std::cout << "Found " << local_constraints.size() << " constraints" << std::endl;

    auto mappings = createMappings(local_particles, local_constraints);
    auto matrices = physics_engine->calculateMatrices(local_particles, local_constraints, std::move(mappings));

    auto deltaC = physics_engine->solveConstraints(matrices, physics_engine->solver_config.dt);

    updateLocalParticlesFromSolution(deltaC);

    if (vtk_logger_) {
      auto sim_state = createSimulationState({});
      vtk_logger_->logTimestepComplete(physics_engine->solver_config.dt, sim_state.get());
    }

    for (const auto& p : local_particles) {
      p.printState();
    }
  }
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
      PetscPrintf(PETSC_COMM_WORLD, "Particle ID validation complete. Global count: %d\n", global_particle_count);
    }
  }
}

void ParticleManager::enableVTKLogging(const std::string& output_dir, int log_every_n_iterations) {
  vtk_logger_ = vtk::createParticleLogger(output_dir);
}

std::unique_ptr<vtk::ParticleSimulationState> ParticleManager::createSimulationState(
    const std::vector<Constraint>& constraints) const {
  auto state = std::make_unique<vtk::ParticleSimulationState>();

  // Copy particles
  state->particles = local_particles;
  state->constraints = constraints;

  // Initialize empty force/torque vectors (would be populated by physics engine in full implementation)
  state->forces.resize(local_particles.size(), {0.0, 0.0, 0.0});
  state->torques.resize(local_particles.size(), {0.0, 0.0, 0.0});
  state->impedance.resize(local_particles.size(), 1.0);

  // Calculate max overlap from constraints
  state->max_overlap = 0.0;
  for (const auto& constraint : constraints) {
    state->max_overlap = std::max(state->max_overlap, std::abs(constraint.delta0));
  }

  // Set other state values
  state->constraint_iterations = constraints.size();
  state->avg_bbpgd_iterations = 1;  // Would be set by solver in full implementation
  state->l0 = 1.0;                  // Default particle length

  return state;
}
