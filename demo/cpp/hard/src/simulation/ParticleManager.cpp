#include "ParticleManager.h"

#include <petsc.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include "dynamics/Constraint.h"
#include "dynamics/Physics.h"
#include "dynamics/PhysicsEngine.h"
#include "logger/VTK.h"

ParticleManager::ParticleManager(SimulationConfig sim_config, PhysicsConfig physics_config, SolverConfig solver_config)
    : sim_config_(sim_config), physics_config_(physics_config), solver_config_(solver_config) {
  physics_engine = std::make_unique<PhysicsEngine>(physics_config, solver_config);
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
  scatterVectorToLocal(solution.dC, indicesConfig, dC_local, dC_scatter, dC_is);

  // Get array pointer
  const PetscScalar* dC_array;
  VecGetArrayRead(dC_local, &dC_array);

  Vec dF_local;
  VecScatter dF_scatter;
  IS dF_is;
  scatterVectorToLocal(solution.f, indicesVelocity, dF_local, dF_scatter, dF_is);

  const PetscScalar* dF_array;
  VecGetArrayRead(dF_local, &dF_array);

  Vec dU_local;
  VecScatter dU_scatter;
  IS dU_is;
  scatterVectorToLocal(solution.u, indicesVelocity, dU_local, dU_scatter, dU_is);

  const PetscScalar* dU_array;
  VecGetArrayRead(dU_local, &dU_array);

  // Process each particle (7 values per particle)
  for (int i = 0; i < local_particles.size(); ++i) {
    const PetscScalar* particle_values = &dC_array[i * Particle::getStateSize()];
    const PetscScalar* force_values = &dF_array[i * 6];
    const PetscScalar* velocity_values = &dU_array[i * 6];

    local_particles[i].eulerStepPosition(particle_values, dt);
    local_particles[i].addForceAndTorque(force_values);

    local_particles[i].addVelocity(velocity_values);
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

  scatterVectorToLocal(solution.dL, indices, dL_local, dL_scatter, dL_is);
  scatterVectorToLocal(solution.impedance, indices, impedance_local,
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

std::vector<Particle> ParticleManager::divideParticles() {
  std::vector<Particle> new_particles;
  for (int i = 0; i < local_particles.size(); i++) {
    auto new_particle = local_particles[i].divide(-1);
    if (new_particle) {
      new_particles.push_back(new_particle.value());
    }
  }
  return new_particles;
}

PhysicsEngine::SolverSolution ParticleManager::step(int i, std::function<void()> exchangeGhostParticles) {
  resetLocalParticles();

  std::vector<Particle> particles_before_step = local_particles;
  auto solver_solution = physics_engine->solveConstraintsRecursiveConstraints(*this, sim_config_.dt, i, exchangeGhostParticles);

  return solver_solution;
}

void ParticleManager::updateDomainBounds(const std::array<double, 3>& min_bounds, const std::array<double, 3>& max_bounds) {
  physics_engine->updateCollisionDetectorBounds(min_bounds, max_bounds);
}

void ParticleManager::printProgress(int current_iteration, int total_iterations) const {
  PetscPrintf(PETSC_COMM_WORLD, "\rProgress: %3d / %d (%5.1f%%) | Time: %3.1f min / %3.1f min | Particles: %4d",
              current_iteration, total_iterations,
              (double)current_iteration / total_iterations * 100,
              (double)current_iteration * sim_config_.dt / 60,
              (double)total_iterations * sim_config_.dt / 60,
              global_particle_count);
}
