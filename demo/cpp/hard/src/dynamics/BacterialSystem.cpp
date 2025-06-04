#include "BacterialSystem.h"

#include <petsc.h>
#include <solver/LCP.h>

#include <unordered_map>
#include <vector>

BacterialSystem::BacterialSystem(int argc, char** argv) {
  initializeSystem();
}

void BacterialSystem::initializeParticles() {
  // Get local ownership range
  PetscInt localStart, localEnd;
  VecGetOwnershipRange(configuration_, &localStart, &localEnd);

  // Only the root process (rank 0) initializes the first particle
  if (localStart == 0) {
    double angle = 0;
    double l0 = 1;

    Bacterium initialParticle = {
        .position = {0.0, 0.0, 0.0},  // Origin
        .quaternion = {cos(angle / 2), 0, 0, sin(angle / 2)},
        .length = l0,
        .diameter = l0 / 2};

    // Set values in PETSc vectors
    PetscScalar* localConfig;
    VecGetArray(configuration_, &localConfig);

    // Position vector contains [x,y,z,q0,q1,q2,q3] per particle
    localConfig[0] = initialParticle.position[0];    // x
    localConfig[1] = initialParticle.position[1];    // y
    localConfig[2] = initialParticle.position[2];    // z
    localConfig[3] = initialParticle.quaternion[0];  // q0
    localConfig[4] = initialParticle.quaternion[1];  // q1
    localConfig[5] = initialParticle.quaternion[2];  // q2
    localConfig[6] = initialParticle.quaternion[3];  // q3
    localConfig[8] = initialParticle.length;         // l

    VecRestoreArray(configuration_, &localConfig);
  }

  // Synchronize across all processes
  VecAssemblyBegin(configuration_);
  VecAssemblyEnd(configuration_);
}

void BacterialSystem::initializeSystem() {
  // Estimate initial capacity (e.g., 100 particles)
  int particles_per_rank = 10;  // Example initial estimate
  local_estimate_ = 8 * particles_per_rank;

  VecCreateMPI(PETSC_COMM_WORLD, 8 * local_estimate_, PETSC_DETERMINE, &configuration_);
  VecSetOption(configuration_, VEC_IGNORE_NEGATIVE_INDICES, PETSC_TRUE);

  // Track active particles separately
  current_particles_ = 0;

  // Initialize first particle
  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  if (rank == 0) {
    initializeParticles();
    current_particles_ = 1;
  }

  VecView(configuration_, PETSC_VIEWER_STDOUT_WORLD);

  exit(0);
}

void BacterialSystem::addParticle(Bacterium p) {
  PetscInt local_size;
  VecGetLocalSize(configuration_, &local_size);

  if (8 * (current_particles_ + 1) > local_size) {
    // Redistribute with doubled capacity
    redistributeVectors(2 * local_size);
  }

  // Insert particle data
  PetscInt idx = 8 * current_particles_;
  PetscScalar values[8] = {p.position[0], p.position[1], p.position[2], p.quaternion[0], p.quaternion[1], p.quaternion[2], p.length};
  VecSetValues(configuration_, 8, &idx, values, INSERT_VALUES);

  current_particles_++;
  VecAssemblyBegin(configuration_);
  VecAssemblyEnd(configuration_);
}

void BacterialSystem::redistributeVectors(PetscInt new_local_size) {
  Vec new_config;
  VecCreateMPI(PETSC_COMM_WORLD, new_local_size, PETSC_DETERMINE, &new_config);

  // Create scatter context to move data
  VecScatter ctx;
  VecScatterCreateToAll(configuration_, &ctx, &new_config);
  VecScatterBegin(ctx, configuration_, new_config, INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(ctx, configuration_, new_config, INSERT_VALUES, SCATTER_FORWARD);

  VecDestroy(&configuration_);
  configuration_ = new_config;
  VecScatterDestroy(&ctx);
}

void BacterialSystem::detectContacts() {
  // Get local particles
  Vec localPositions;
  DMGetLocalVector(dm_, &localPositions);
  DMGlobalToLocalBegin(dm_, configuration_, INSERT_VALUES, localPositions);
  DMGlobalToLocalEnd(dm_, configuration_, INSERT_VALUES, localPositions);

  // Access raw array
  double* positionsArray;
  VecGetArray(localPositions, &positionsArray);

  // Spatial hashing
  std::unordered_map<int, std::vector<int>> spatialHash;
  // buildSpatialHash(positionsArray, spatialHash);

  // Check neighboring cells
  // std::vector<ContactPair> contacts;
  // for (const auto& [cell, particles] : spatialHash) {
  //   checkCellNeighbors(cell, particles, positionsArray, contacts);
  // }

  // Process contacts
  // processContacts(contacts);

  VecRestoreArray(localPositions, &positionsArray);
  DMRestoreLocalVector(dm_, &localPositions);
}

void BacterialSystem::timeStep() {
  // Solve constraints
  Vec gamma = solveLCP(jacobian_, forces_);

  // U = f @ gamma
  Vec U;
  MatMult(jacobian_, gamma, U);

  // // Update velocities: pos = pos + force
  VecAXPY(configuration_, dt_, U);

  VecDestroy(&U);
  VecDestroy(&gamma);
}

void BacterialSystem::run() {
  for (int i = 0; i < 100; i++) {
    timeStep();
  }
}