#include "BacterialSystem.h"

#include <petsc.h>
#include <solver/LCP.h>

#include <iostream>
#include <unordered_map>
#include <vector>

#include "Constraint.h"
#include "Forces.h"
#include "ParticleData.h"
#include "Physics.h"

BacterialSystem::BacterialSystem(int argc, char** argv) {
  initializeSystem();
}

void BacterialSystem::initializeParticles() {
  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  // For this initial setup, we'll have rank 0 create all particles
  // and then we can distribute them later if needed.
  if (rank == 0) {
    double angle = 0;
    double l0 = 1.0;

    Bacterium p1 = {.id = 0, .position = {0.5, 0.1, 0.0}, .quaternion = {cos(angle / 2), 0, 0, sin(angle / 2)}, .length = l0, .diameter = l0 / 2};
    Bacterium p2 = {.id = 1, .position = {1.0, -0.1, 0.0}, .quaternion = {cos(angle / 2), 0, 0, sin(angle / 2)}, .length = l0, .diameter = l0 / 2};

    addParticle(p1);
    addParticle(p2);
  }

  // After rank 0 adds particles, it knows the total count.
  // We must broadcast this count to all other processes so they are in sync.
  MPI_Bcast(&global_particle_count_, 1, MPIU_INT, 0, PETSC_COMM_WORLD);
}

void BacterialSystem::initializeSystem() {
  global_particle_count_ = 0;
  local_bacteria_.clear();
  initializeParticles();
}

void BacterialSystem::addParticle(Bacterium p) {
  // In this new design, addParticle is simpler. It just adds to the local C++ vector.
  // The responsibility of assigning a unique global ID is handled here.
  // We assume for now that only rank 0 adds particles.
  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  if (rank == 0) {
    p.id = global_particle_count_;
    local_bacteria_.push_back(p);
    global_particle_count_++;
  }
}

void BacterialSystem::timeStep() {
  // --- Step 1: Assemble the global configuration vector from local data ---
  Vec configuration;
  VecCreate(PETSC_COMM_WORLD, &configuration);
  VecSetSizes(configuration, local_bacteria_.size() * COMPONENTS_PER_PARTICLE, global_particle_count_ * COMPONENTS_PER_PARTICLE);
  VecSetFromOptions(configuration);
  VecSetUp(configuration);

  // Each rank provides the values for its local bacteria
  int num_local_bacteria = local_bacteria_.size();
  if (num_local_bacteria > 0) {
    std::vector<PetscInt> indices(num_local_bacteria * COMPONENTS_PER_PARTICLE);
    std::vector<PetscScalar> values(num_local_bacteria * COMPONENTS_PER_PARTICLE);
    for (int i = 0; i < num_local_bacteria; ++i) {
      const auto& p = local_bacteria_[i];
      int base_idx = i * COMPONENTS_PER_PARTICLE;
      int global_base_idx = p.id * COMPONENTS_PER_PARTICLE;

      for (int j = 0; j < COMPONENTS_PER_PARTICLE; ++j) {
        indices[base_idx + j] = global_base_idx + j;
      }
      values[base_idx + 0] = p.position[0];
      values[base_idx + 1] = p.position[1];
      values[base_idx + 2] = p.position[2];
      values[base_idx + 3] = p.quaternion[0];
      values[base_idx + 4] = p.quaternion[1];
      values[base_idx + 5] = p.quaternion[2];
      values[base_idx + 6] = p.quaternion[3];
      values[base_idx + 7] = p.length;
    }
    VecSetValues(configuration, indices.size(), indices.data(), values.data(), INSERT_VALUES);
  }
  VecAssemblyBegin(configuration);
  VecAssemblyEnd(configuration);

  // --- Step 2: Perform Physics calculations using the new global vector ---
  std::vector<Constraint> constraints_;
  constraints_.emplace_back(Constraint(0.5, 0, 1, {1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}, {7.7, 8.8, 9.9}, {10.1, 11.2, 12.3}, {13.4, 14.5, 15.6}));

  std::unique_ptr<Mat> D = calculate_Jacobian(constraints_, global_particle_count_);

  std::unique_ptr<Mat> M = calculate_MobilityMatrix(configuration, global_particle_count_);

  PetscPrintf(PETSC_COMM_WORLD, "D Matrix:\n");
  MatView(*D, PETSC_VIEWER_STDOUT_WORLD);
  PetscPrintf(PETSC_COMM_WORLD, "Mobility Matrix M:\n");
  MatView(*M, PETSC_VIEWER_STDOUT_WORLD);

  // Clean up the temporary vector
  VecDestroy(&configuration);
}

void BacterialSystem::run() {
  for (int i = 0; i < 1; i++) {
    timeStep();
  }
}
