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
#include "dynamics/models/Hard.h"
#include "dynamics/models/Soft.h"
#include "logger/ParticleLogger.h"

ParticleManager::ParticleManager(SimulationParameters& params, vtk::ParticleLogger& particle_logger, vtk::ConstraintLogger& constraint_logger, const std::string& mode)
    : params_(params), particle_logger_(particle_logger), constraint_logger_(constraint_logger), mode_(mode), collision_detector_(CollisionDetector(params.solver_config.tolerance, params.solver_config.linked_cell_size)) {
}

// Updated move function using helpers
void ParticleManager::moveLocalParticlesFromSolution(const MovementSolution& solution, double dt) {
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
    local_particles[i].setForceAndTorque(force_values);

    local_particles[i].setVelocity(velocity_values);
  }

  // Clean up
  VecRestoreArrayRead(dC_local, &dC_array);
  VecRestoreArrayRead(dF_local, &dF_array);
  VecRestoreArrayRead(dU_local, &dU_array);
  cleanupScatteredResources(dC_local, dC_scatter, dC_is);
  cleanupScatteredResources(dF_local, dF_scatter, dF_is);
  cleanupScatteredResources(dU_local, dU_scatter, dU_is);
}

void ParticleManager::setGrowParamsFromSolution(const GrowthSolution& solution) {
  // Collect needed global indices and maintain mapping
  std::vector<PetscInt> indices;
  std::vector<size_t> local_indices;  // Maps to position in local_particles
  for (size_t i = 0; i < local_particles.size(); i++) {
    indices.push_back(local_particles[i].getGID());
    local_indices.push_back(i);
  }

  // Scatter both vectors
  Vec dL_local, impedance_local, stress_local;
  VecScatter dL_scatter, impedance_scatter, stress_scatter;
  IS dL_is, impedance_is, stress_is;

  scatterVectorToLocal(solution.dL, indices, dL_local, dL_scatter, dL_is);
  scatterVectorToLocal(solution.impedance, indices, impedance_local,
                       impedance_scatter, impedance_is);
  scatterVectorToLocal(solution.stress, indices, stress_local, stress_scatter, stress_is);

  // Get array pointers
  const PetscScalar *dL_array, *impedance_array, *stress_array;
  VecGetArrayRead(dL_local, &dL_array);
  VecGetArrayRead(impedance_local, &impedance_array);
  VecGetArrayRead(stress_local, &stress_array);

  // Process each particle using correct mapping
  for (size_t arr_idx = 0; arr_idx < local_indices.size(); arr_idx++) {
    size_t particle_idx = local_indices[arr_idx];

    local_particles[particle_idx].setImpedance(impedance_array[arr_idx]);
    local_particles[particle_idx].setStress(stress_array[arr_idx]);
    local_particles[particle_idx].setLdot(dL_array[arr_idx]);
  }

  // Clean up
  VecRestoreArrayRead(dL_local, &dL_array);
  VecRestoreArrayRead(impedance_local, &impedance_array);
  VecRestoreArrayRead(stress_local, &stress_array);
  cleanupScatteredResources(dL_local, dL_scatter, dL_is);
  cleanupScatteredResources(impedance_local, impedance_scatter, impedance_is);
  cleanupScatteredResources(stress_local, stress_scatter, stress_is);
}

void ParticleManager::grow(double dt) {
  // Process each particle using correct mapping
  for (auto& p : local_particles) {
    p.eulerStepLength(p.getData().ldot, dt);
  }
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

ParticleManager::SolverSolution ParticleManager::step(int i, std::function<void()> exchangeGhostParticles) {
  for (auto& p : local_particles) {
    p.reset();
  }

  double dt = params_.sim_config.dt_s;

  if (mode_ == "soft") {
    auto solver_solution = solveSoftPotential(*this, collision_detector_, params_, dt, i, exchangeGhostParticles, particle_logger_, constraint_logger_);
    return solver_solution;
  } else if (mode_ == "hard") {
    auto solver_solution = solveHardModel(*this, collision_detector_, params_, dt, i, exchangeGhostParticles, particle_logger_, constraint_logger_);
    return solver_solution;
  }

  throw std::runtime_error("Invalid mode: " + mode_);
}

void ParticleManager::printProgress(int current_iteration, int total_iterations) const {
  PetscPrintf(PETSC_COMM_WORLD, "\rProgress: %3d / %d (%5.1f%%) | Time: %3.1f min / %3.1f min | Particles: %4d",
              current_iteration, total_iterations,
              (double)current_iteration / total_iterations * 100,
              (double)current_iteration * params_.sim_config.dt_s / 60,
              (double)total_iterations * params_.sim_config.dt_s / 60,
              global_particle_count);
}
