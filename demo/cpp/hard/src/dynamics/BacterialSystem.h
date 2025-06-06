#pragma once
#include <petsc.h>

#include <memory>
#include <vector>  // For std::vector

#include "Bacteria.h"
#include "ParticleData.h"

class BacterialSystem {
 public:
  BacterialSystem(int argc, char** argv);

  void initializeSystem();
  void run();

 private:
  // --- New Data Structure ---
  // Each rank stores a list of the bacteria it is responsible for.
  std::vector<Bacterium> local_bacteria_;

  // --- PETSc Objects ---
  DM dm_;
  Vec configuration_;  // Persistent configuration vector
  Vec forces_;
  Mat jacobian_;
  Vec lambda_;

  // Reusable vectors for configuration updates
  std::vector<PetscInt> indices_;
  std::vector<PetscScalar> values_;

  // Simulation parameters
  double time_;
  double dt_;
  int current_step_;

  // --- Refactored Methods ---
  void initializeParticles();
  void addParticle(Bacterium p);
  void cleanup();
  void detectContacts();
  void timeStep();

  // The global particle count, consistent across all ranks
  PetscInt global_particle_count_;

  // These are no longer needed in the new design
  // void redistributeVectors(PetscInt new_local_size);
  // int particles_per_rank_;
  // int local_estimate_;
  // int current_particles_;
};
