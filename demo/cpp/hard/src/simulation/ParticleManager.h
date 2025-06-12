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

  std::unique_ptr<PhysicsEngine> physics_engine;
  std::unique_ptr<ConstraintGenerator> constraint_generator;

  std::vector<Particle> local_particles;
  void eulerStepfromSolution(const VecWrapper& dC);
  void moveLocalParticlesFromSolution(const PhysicsEngine::PhysicsSolution& solution);
  void resetLocalParticles();

 private:
  void validateParticleIDs() const;
  void printProgress(int current_iteration, int total_iterations, const PhysicsEngine::SolverSolution& solver_solution) const;

  std::vector<Particle> new_particle_buffer;
  PetscInt global_particle_count = 0;

  // Composition instead of doing everything inline
  PhysicsConfig config;

  // Reusable vectors for configuration updates
  std::vector<PetscInt> indices;
  std::vector<PetscScalar> values;

  // VTK logging
  std::unique_ptr<vtk::SimulationLogger> vtk_logger_;
  std::unique_ptr<vtk::SimulationLogger> constraint_loggers_;

  // Helper method to create simulation state for VTK logging
  std::unique_ptr<vtk::ParticleSimulationState> createSimulationState(
      const PhysicsEngine::SolverSolution& solver_solution) const;
};
