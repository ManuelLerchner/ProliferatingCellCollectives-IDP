#pragma once
#include <petsc.h>

#include <vector>

#include "Particle.h"
#include "util/PetscRaii.h"

class ParticleManager {
 public:
  ParticleManager();

  void queueNewParticle(Particle p);

  void run();

 private:
  // --- New Data Structure ---
  // Each rank stores a list of the bacteria it is responsible for.
  std::vector<Particle> new_particle_buffer;
  std::vector<Particle> local_particles;

  // --- PETSc Objects ---
  DM mobility_matrix;
  Vec configuration_;  // Persistent configuration vector
  Vec forces_;
  Mat jacobian_;
  Vec lambda_;

  // Reusable vectors for configuration updates
  std::vector<PetscInt> indices;
  std::vector<PetscScalar> values;

  // Simulation parameters
  double time;
  double dt;
  int current_step;

  // --- Refactored Methods ---
  void initializeParticles();
  void cleanup();
  void detectContacts();
  void timeStep();
  void commitNewParticles();

  ISLocalToGlobalMappingWrapper createLocalToGlobalMapping(int local_num_particles, int components_per_particle);
  ISLocalToGlobalMappingWrapper create_constraint_map(int local_num_constraints);

  // The global particle count, consistent across all ranks
  PetscInt global_particle_count = 0;
};
