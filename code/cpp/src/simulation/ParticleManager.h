#pragma once
#include <petsc.h>

#include <array>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "dynamics/PhysicsEngine.h"
#include "logger/ConstraintLogger.h"
#include "logger/ParticleLogger.h"
#include "simulation/Particle.h"
#include "util/Config.h"

class ParticleManager {
 public:
  ParticleManager(SimulationConfig sim_config, PhysicsConfig physics_config, SolverConfig solver_config, vtk::ParticleLogger& particle_logger, vtk::ConstraintLogger& constraint_logger, const std::string& mode);

  void queueNewParticles(Particle p);
  void commitNewParticles();

  void moveLocalParticlesFromSolution(const PhysicsEngine::MovementSolution& solution);
  void growLocalParticlesFromSolution(const PhysicsEngine::GrowthSolution& solution);

  std::vector<Particle> divideParticles();
  PhysicsEngine::SolverSolution step(int i, std::function<void()> exchangeGhostParticles);

  void redistributeParticles();
  void updateDomainBounds(const std::array<double, 3>& min_bounds, const std::array<double, 3>& max_bounds);

  std::vector<Particle> local_particles;
  std::vector<Particle> ghost_particles;
  PetscInt global_particle_count = 0;

  std::unique_ptr<PhysicsEngine> physics_engine;

  SimulationConfig sim_config_;
  PhysicsConfig physics_config_;
  SolverConfig solver_config_;

 private:
  void printProgress(int current_iteration, int total_iterations) const;
  vtk::ParticleLogger& particle_logger_;
  vtk::ConstraintLogger& constraint_logger_;
  std::string mode_;
};
