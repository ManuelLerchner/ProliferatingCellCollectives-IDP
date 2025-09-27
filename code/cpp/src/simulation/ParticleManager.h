#pragma once
#include <petsc.h>

#include <array>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "logger/ConstraintLogger.h"
#include "logger/ParticleLogger.h"
#include "simulation/Particle.h"
#include "spatial/CollisionDetector.h"
#include "util/Config.h"
#include "util/PetscRaii.h"

class ParticleManager {
 public:
  struct MovementSolution {
    const VecWrapper& dC;
    const VecWrapper& f;
    const VecWrapper& u;
  };

  struct GrowthSolution {
    const VecWrapper& dL;
    const VecWrapper& impedance;
    const VecWrapper& stress;
  };

  struct SolverSolution {
    std::vector<Constraint> constraints;
    int constraint_iterations;
    size_t bbpgd_iterations;
    double residual;
    double max_overlap;
  };

  ParticleManager(SimulationParameters params, vtk::ParticleLogger& particle_logger, vtk::ConstraintLogger& constraint_logger, const std::string& mode);

  void queueNewParticles(Particle p);
  void commitNewParticles();

  void moveLocalParticlesFromSolution(const MovementSolution& solution, double dt);
  void growLocalParticlesFromSolution(const GrowthSolution& solution, double dt);

  std::vector<Particle> divideParticles();
  SolverSolution step(int i, std::function<void()> exchangeGhostParticles);

  void redistributeParticles();
  void updateDomainBounds(const std::array<double, 3>& min_bounds, const std::array<double, 3>& max_bounds);

  std::vector<Particle> local_particles;
  std::vector<Particle> ghost_particles;
  PetscInt global_particle_count = 0;

  SimulationParameters params_;
  CollisionDetector collision_detector_;

 private:
  void printProgress(int current_iteration, int total_iterations) const;
  vtk::ParticleLogger& particle_logger_;
  vtk::ConstraintLogger& constraint_logger_;
  std::string mode_;
};
