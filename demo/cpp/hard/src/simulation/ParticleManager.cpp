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
#include "util/ParticleMPI.h"

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

std::vector<Particle> ParticleManager::divideParticles() {
  std::vector<Particle> new_particles;
  for (int i = 0; i < local_particles.size(); i++) {
    auto new_particle = local_particles[i].divide();
    if (new_particle) {
      new_particles.push_back(new_particle.value());
    }
  }
  return new_particles;
}

PhysicsEngine::SolverSolution ParticleManager::step(int i) {
  resetLocalParticles();

  std::vector<Particle> particles_before_step = local_particles;
  auto solver_solution = physics_engine->solveConstraintsRecursiveConstraints(*this, sim_config_.dt, i);

  return solver_solution;
}

void ParticleManager::updateDomainBounds(const std::array<double, 3>& min_bounds, const std::array<double, 3>& max_bounds) {
  physics_engine->updateCollisionDetectorBounds(min_bounds, max_bounds);
}

// void ParticleManager::redistributeParticles() {
//   PetscMPIInt rank, size;
//   MPI_Comm_rank(cart_comm_, &rank);
//   MPI_Comm_size(cart_comm_, &size);

//   if (size == 1) return;

//   // Decompose along X and Y axes
//   double domain_x = domain_max_[0] - domain_min_[0];
//   double domain_y = domain_max_[1] - domain_min_[1];

//   // Enforce minimum domain size for decomposition
//   double min_total_width_x = sim_config_.min_box_size.x * dims_[0];
//   double min_total_width_y = sim_config_.min_box_size.y * dims_[1];
//   if (domain_x < min_total_width_x) {
//     domain_x = min_total_width_x;
//   }
//   if (domain_y < min_total_width_y) {
//     domain_y = min_total_width_y;
//   }

//   double rank_width_x = domain_x / dims_[0];
//   double rank_width_y = domain_y / dims_[1];

//   std::vector<std::vector<ParticleData>> send_buffers(size);
//   std::vector<Particle> keep_particles;

//   for (auto& p : local_particles) {
//     const auto& pos = p.getPosition();
//     int coord_x = static_cast<int>((pos[0] - domain_min_[0]) / rank_width_x);
//     int coord_y = static_cast<int>((pos[1] - domain_min_[1]) / rank_width_y);
//     coord_x = std::max(0, std::min(dims_[0] - 1, coord_x));
//     coord_y = std::max(0, std::min(dims_[1] - 1, coord_y));

//     int owner_rank;
//     int owner_coords[] = {coord_x, coord_y};
//     MPI_Cart_rank(cart_comm_, owner_coords, &owner_rank);

//     if (owner_rank == rank) {
//       keep_particles.push_back(std::move(p));
//     } else {
//       send_buffers[owner_rank].push_back(p.toStruct());
//     }
//   }

//   // All-to-all communication to determine send/receive counts
//   std::vector<int> send_counts(size);
//   for (int i = 0; i < size; ++i) {
//     send_counts[i] = send_buffers[i].size();
//   }
//   std::vector<int> recv_counts(size);
//   MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, cart_comm_);

//   // Prepare for Alltoallv
//   std::vector<ParticleData> send_buffer_flat;
//   std::vector<int> send_displs(size + 1, 0);
//   for (int i = 0; i < size; ++i) {
//     send_displs[i + 1] = send_displs[i] + send_counts[i];
//     send_buffer_flat.insert(send_buffer_flat.end(), send_buffers[i].begin(), send_buffers[i].end());
//   }

//   std::vector<int> recv_displs(size + 1, 0);
//   int total_recv_count = 0;
//   for (int i = 0; i < size; ++i) {
//     recv_displs[i + 1] = recv_displs[i] + recv_counts[i];
//     total_recv_count += recv_counts[i];
//   }

//   std::vector<ParticleData> recv_buffer(total_recv_count);

//   // Exchange particles
//   MPI_Alltoallv(send_buffer_flat.data(), send_counts.data(), send_displs.data(), mpi_particle_data_type_,
//                 recv_buffer.data(), recv_counts.data(), recv_displs.data(), mpi_particle_data_type_,
//                 cart_comm_);

//   // Update local particles
//   local_particles = std::move(keep_particles);
//   for (const auto& p_data : recv_buffer) {
//     local_particles.emplace_back(p_data);
//   }

//   // Update global particle count and re-sort
//   global_particle_count = globalReduce(static_cast<PetscInt>(local_particles.size()), MPI_SUM);
//   std::sort(local_particles.begin(), local_particles.end(),
//             [](const Particle& a, const Particle& b) { return a.getGID() < b.getGID(); });

//   for (int i = 0; i < local_particles.size(); i++) {
//     local_particles[i].setLocalID(i);
//   }
// }

void ParticleManager::printProgress(int current_iteration, int total_iterations) const {
  PetscPrintf(PETSC_COMM_WORLD, "\rProgress: %3d / %d (%5.1f%%) | Time: %3.1f min / %3.1f min | Particles: %4d",
              current_iteration, total_iterations,
              (double)current_iteration / total_iterations * 100,
              (double)current_iteration * sim_config_.dt / 60,
              (double)total_iterations * sim_config_.dt / 60,
              global_particle_count);
}
