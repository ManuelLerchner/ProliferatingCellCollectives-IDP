#include "ParticleManager.h"

#include <petsc.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include "dynamics/Constraint.h"
#include "dynamics/Physics.h"
#include "dynamics/PhysicsEngine.h"
#include "logger/VTK.h"
#include "spatial/ConstraintGenerator.h"
#include "util/MPIUtil.h"

ParticleManager::ParticleManager(SimulationConfig sim_config, PhysicsConfig physics_config, SolverConfig solver_config)
    : sim_config_(sim_config), physics_config_(physics_config), solver_config_(solver_config) {
  constraint_generator = std::make_unique<ConstraintGenerator>(solver_config.tolerance, 2.5 * physics_config.l0);
  physics_engine = std::make_unique<PhysicsEngine>(physics_config, solver_config);

  int log_every_n_steps = 1;
  if (sim_config.log_frequency_seconds > 0) {
    log_every_n_steps = std::max(1, static_cast<int>(sim_config.log_frequency_seconds / sim_config.dt));
  }

  vtk_logger_ = vtk::createParticleLogger("./vtk_output", log_every_n_steps);
  constraint_loggers_ = vtk::createConstraintLogger("./vtk_output", log_every_n_steps);
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

  this->global_particle_count += globalReduce(num_to_add_local, MPI_SUM);
}

void ParticleManager::resetLocalParticles() {
  const PetscScalar* gamma_array;
  PetscInt local_size;

  for (int i = 0; i < local_particles.size(); i++) {
    local_particles[i].clearForceAndTorque();
  }
}
// Helper function to scatter values from a parallel vector to a local array
static void scatterVectorToLocal(Vec global_vec, const std::vector<PetscInt>& indices,
                                 Vec& local_vec, VecScatter& scatter, IS& is) {
  ISCreateGeneral(PETSC_COMM_SELF, indices.size(), indices.data(), PETSC_COPY_VALUES, &is);
  VecCreateSeq(PETSC_COMM_SELF, indices.size(), &local_vec);
  VecScatterCreate(global_vec, is, local_vec, NULL, &scatter);
  VecScatterBegin(scatter, global_vec, local_vec, INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(scatter, global_vec, local_vec, INSERT_VALUES, SCATTER_FORWARD);
}

// Helper function to clean up scattered resources
static void cleanupScatteredResources(Vec& local_vec, VecScatter& scatter, IS& is) {
  VecScatterDestroy(&scatter);
  VecDestroy(&local_vec);
  ISDestroy(&is);
}

// Updated move function using helpers
void ParticleManager::moveLocalParticlesFromSolution(const PhysicsEngine::MovementSolution& solution) {
  double dt = sim_config_.dt;

  // Collect all needed global indices (7 per particle)
  std::vector<PetscInt> indicesConfig;
  std::vector<PetscInt> indicesVelocity;
  for (auto& p : local_particles) {
    for (PetscInt i = 0; i < Particle::getStateSize(); i++) {
      indicesConfig.push_back(p.getGID() * Particle::getStateSize() + i);
    }
    for (PetscInt i = 0; i < 6; i++) {
      indicesVelocity.push_back(p.getGID() * 6 + i);
    }
  }

  // Scatter the dC vector
  Vec dC_local;
  VecScatter dC_scatter;
  IS dC_is;
  scatterVectorToLocal(solution.dC.get(), indicesConfig, dC_local, dC_scatter, dC_is);

  // Get array pointer
  const PetscScalar* dC_array;
  VecGetArrayRead(dC_local, &dC_array);

  Vec dF_local;
  VecScatter dF_scatter;
  IS dF_is;
  scatterVectorToLocal(solution.f.get(), indicesVelocity, dF_local, dF_scatter, dF_is);

  const PetscScalar* dF_array;
  VecGetArrayRead(dF_local, &dF_array);

  Vec dU_local;
  VecScatter dU_scatter;
  IS dU_is;
  scatterVectorToLocal(solution.u.get(), indicesVelocity, dU_local, dU_scatter, dU_is);

  const PetscScalar* dU_array;
  VecGetArrayRead(dU_local, &dU_array);

  // Process each particle (7 values per particle)
  for (auto& p : local_particles) {
    const PetscScalar* particle_values = &dC_array[p.getLocalID() * Particle::getStateSize()];
    const PetscScalar* force_values = &dF_array[p.getLocalID() * 6];
    const PetscScalar* velocity_values = &dU_array[p.getLocalID() * 6];

    p.eulerStepPosition(particle_values, dt);
    p.addForceAndTorque(force_values);

    p.addVelocity(velocity_values);
  }

  // Clean up
  VecRestoreArrayRead(dC_local, &dC_array);
  VecRestoreArrayRead(dF_local, &dF_array);
  VecRestoreArrayRead(dU_local, &dU_array);
  cleanupScatteredResources(dC_local, dC_scatter, dC_is);
  cleanupScatteredResources(dF_local, dF_scatter, dF_is);
  cleanupScatteredResources(dU_local, dU_scatter, dU_is);
}

void ParticleManager::growLocalParticlesFromSolution(const PhysicsEngine::GrowthSolution& solution) {
  double dt = sim_config_.dt;

  // Collect needed global indices and maintain mapping
  std::vector<PetscInt> indices;
  std::vector<size_t> local_indices;  // Maps to position in local_particles
  for (size_t i = 0; i < local_particles.size(); i++) {
    indices.push_back(local_particles[i].getGID());
    local_indices.push_back(i);
  }

  // Scatter both vectors
  Vec dL_local, impedance_local;
  VecScatter dL_scatter, impedance_scatter;
  IS dL_is, impedance_is;

  scatterVectorToLocal(solution.dL.get(), indices, dL_local, dL_scatter, dL_is);
  scatterVectorToLocal(solution.impedance.get(), indices, impedance_local,
                       impedance_scatter, impedance_is);

  // Get array pointers
  const PetscScalar *dL_array, *impedance_array;
  VecGetArrayRead(dL_local, &dL_array);
  VecGetArrayRead(impedance_local, &impedance_array);

  // Process each particle using correct mapping
  for (size_t arr_idx = 0; arr_idx < local_indices.size(); arr_idx++) {
    size_t particle_idx = local_indices[arr_idx];

    local_particles[particle_idx].eulerStepLength(dL_array[arr_idx], dt);
    local_particles[particle_idx].setImpedance(impedance_array[arr_idx]);
  }

  // Clean up
  VecRestoreArrayRead(dL_local, &dL_array);
  VecRestoreArrayRead(impedance_local, &impedance_array);
  cleanupScatteredResources(dL_local, dL_scatter, dL_is);
  cleanupScatteredResources(impedance_local, impedance_scatter, impedance_is);
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

    // auto solver_solution = physics_engine->solveConstraintsSingleConstraint(*this, physics_engine->solver_config.dt);
    std::vector<Particle> particles_before_step = local_particles;
    auto solver_solution = physics_engine->solveConstraintsRecursiveConstraints(*this, sim_config_.dt, i);

    if (vtk_logger_) {
      auto sim_state = createSimulationState(solver_solution, &particles_before_step);

      vtk_logger_->logTimestepComplete(sim_config_.dt, sim_state.get());
      constraint_loggers_->logTimestepComplete(sim_config_.dt, sim_state.get());
      printProgress(i + 1, num_steps);
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

void ParticleManager::printProgress(int current_iteration, int total_iterations) const {
  PetscPrintf(PETSC_COMM_WORLD, "\rProgress: %3d / %d (%5.1f%%) | Time: %3.1f min / %3.1f min | Particles: %4d",
              current_iteration, total_iterations,
              (double)current_iteration / total_iterations * 100,
              (double)current_iteration * sim_config_.dt / 60,
              (double)total_iterations * sim_config_.dt / 60,
              global_particle_count);
}

std::unique_ptr<vtk::ParticleSimulationState> ParticleManager::createSimulationState(
    const PhysicsEngine::SolverSolution& solver_solution, const std::vector<Particle>* particles_for_geometry) const {
  auto state = std::make_unique<vtk::ParticleSimulationState>();

  // Copy particles
  if (particles_for_geometry) {
    state->particles = *particles_for_geometry;
  } else {
    state->particles = local_particles;
  }
  state->constraints = std::move(solver_solution.constraints);

  // Initialize empty force/torque vectors (would be populated by physics engine in full implementation)
  state->forces.resize(local_particles.size());
  state->torques.resize(local_particles.size());
  state->velocities_linear.resize(local_particles.size());
  state->velocities_angular.resize(local_particles.size());

  for (int i = 0; i < local_particles.size(); i++) {
    state->forces[i] = local_particles[i].getForce();
    state->torques[i] = local_particles[i].getTorque();
    state->velocities_linear[i] = local_particles[i].getVelocityLinear();
    state->velocities_angular[i] = local_particles[i].getVelocityAngular();
  }

  // Calculate max overlap from constraints
  state->residual = solver_solution.residual;

  // Set other state values
  state->constraint_iterations = solver_solution.constraint_iterations;
  state->bbpgd_iterations = solver_solution.bbpgd_iterations;
  state->l0 = physics_config_.l0;

  return state;
}
