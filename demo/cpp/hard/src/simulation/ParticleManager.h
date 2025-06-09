#pragma once
#include <petsc.h>

#include <memory>
#include <vector>

#include "util/Config.h"
#include "simulation/Particle.h"
#include "dynamics/PhysicsEngine.h"
#include "util/PetscRaii.h"

class ParticleManager {
 public:
 public:
  ParticleManager(PhysicsConfig physics_config, SolverConfig solver_config);

  void queueNewParticle(Particle p);
  void commitNewParticles();
  void timeStep();
  void run();

 private:
  void updateLocalParticlesFromSolution(const VecWrapper& solution);
  void validateParticleIDs() const;

  std::vector<Particle> local_particles;
  std::vector<Particle> new_particle_buffer;
  PetscInt global_particle_count = 0;

  // Composition instead of doing everything inline
  std::unique_ptr<PhysicsEngine> physics_engine;
  std::unique_ptr<ConstraintGenerator> constraint_generator;
  PhysicsConfig config;

  // Reusable vectors for configuration updates
  std::vector<PetscInt> indices;
  std::vector<PetscScalar> values;
};
