#include "ParticleManager.h"

#include <petsc.h>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

#include "Constraint.h"
#include "ParticleData.h"
#include "Physics.h"
#include "PhysicsEngine.h"

ParticleManager::ParticleManager(PhysicsConfig physics_config, SolverConfig solver_config) {
  constraint_generator = std::make_unique<ConstraintGenerator>();
  mapping_manager = std::make_unique<MappingManager>();

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
    new_particle_buffer[i].setId(new_ids[i]);
  }

  local_particles.insert(
      local_particles.end(),
      std::make_move_iterator(new_particle_buffer.begin()),
      std::make_move_iterator(new_particle_buffer.end()));

  // Sort particles by ID to maintain ordering across all ranks
  if (new_particle_buffer.size() > 0) {
    std::sort(local_particles.begin(), local_particles.end(),
              [](const Particle& a, const Particle& b) { return a.getId() < b.getId(); });
  }

  new_particle_buffer.clear();

  PetscInt total_added_this_step;
  MPI_Allreduce(&num_to_add_local, &total_added_this_step, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD);

  this->global_particle_count += total_added_this_step;
}

void ParticleManager::timeStep() {
  std::vector<Constraint> local_constraints;

  for (int i = 0; i < local_particles.size(); i += 2) {
    if (i + 1 < local_particles.size()) {
      int id_i = local_particles[i].getId();
      int id_j = local_particles[i + 1].getId();
      local_constraints.emplace_back(Constraint(
          -0.000000009999999999999999,
          id_i,
          id_j,
          {1, 0, 0},              // normI
          {-0.0000000025, 0, 0},  // posI
          {0.0000000025, 0, 0},   // posJ
          {0.00000000245, 0, 0},  // labI
          {-0.00000000245, 0, 0}  // labJ
          ));
    }
  }

  auto mappings = mapping_manager->createMappings(local_particles, local_constraints);
  auto matrices = physics_engine->calculateMatrices(local_particles, local_constraints, std::move(mappings));

  auto deltaC = physics_engine->solveConstraints(matrices, physics_engine->solver_config.dt);

  updateLocalParticlesFromSolution(deltaC);
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

void ParticleManager::run() {
  for (int i = 0; i < 2; i++) {
    commitNewParticles();

    // Validate particle IDs after committing new particles
    validateParticleIDs();

    for (const auto& p : local_particles) {
      p.printState();
    }

    timeStep();
  }
}

void ParticleManager::validateParticleIDs() const {
  PetscMPIInt rank, size;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  MPI_Comm_size(PETSC_COMM_WORLD, &size);

  // Check local sorting
  for (size_t i = 1; i < local_particles.size(); ++i) {
    if (local_particles[i - 1].getId() >= local_particles[i].getId()) {
      PetscPrintf(PETSC_COMM_WORLD, "ERROR: Local particles not sorted at rank %d: ID[%zu]=%d >= ID[%zu]=%d\n",
                  rank, i - 1, local_particles[i - 1].getId(), i, local_particles[i].getId());
    }
  }

  // Gather all IDs to check global uniqueness (only do this for small numbers of particles)
  if (global_particle_count < 10000) {  // Only for reasonable sizes
    std::vector<PetscInt> local_ids;
    for (const auto& p : local_particles) {
      local_ids.push_back(p.getId());
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
