#pragma once
#include <petsc.h>

#include <memory>
#include <vector>

#include "dynamics/PhysicsEngine.h"
#include "logger/VTK.h"
#include "simulation/Particle.h"
#include "spatial/ConstraintGenerator.h"
#include "util/Config.h"
#include "util/PetscRaii.h"

class ParticleManager {
 public:
  ParticleManager(PhysicsConfig physics_config, SolverConfig solver_config);

  void queueNewParticle(Particle p);
  void commitNewParticles();
  void run(int num_steps);

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

  // VTK logging
  std::unique_ptr<vtk::SimulationLogger> vtk_logger_;
  std::unique_ptr<vtk::SimulationLogger> constraint_loggers_;

  // Helper method to create simulation state for VTK logging
  std::unique_ptr<vtk::ParticleSimulationState> createSimulationState(
      const std::vector<class Constraint>& constraints) const;
};
