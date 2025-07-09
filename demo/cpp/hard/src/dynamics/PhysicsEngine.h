#pragma once

#include <array>
#include <functional>
#include <optional>
#include <random>
#include <vector>

#include "Constraint.h"
#include "logger/ParticleLogger.h"
#include "simulation/Particle.h"
#include "spatial/CollisionDetector.h"
#include "util/Config.h"
#include "util/PetscRaii.h"

// Forward declaration
class ParticleManager;

class PhysicsEngine {
 public:
  PhysicsEngine(PhysicsConfig physics_config, SolverConfig solver_config);

  struct PhysicsMatrices {
    MatWrapper D, S;
    VecWrapper phi;
  };

  struct MovementSolution {
    const VecWrapper& dC;
    const VecWrapper& f;
    const VecWrapper& u;
  };

  struct GrowthSolution {
    const VecWrapper& dL;
    const VecWrapper& impedance;
  };

  struct SolverSolution {
    std::vector<Constraint> constraints;
    int constraint_iterations;
    long long bbpgd_iterations;
    double residual;
  };

  SolverSolution solveConstraintsRecursiveConstraints(ParticleManager& particle_manager, double dt, int iter, std::function<void()> exchangeGhostParticles, vtk::ParticleLogger& particle_logger, vtk::ConstraintLogger& constraint_logger);

  void updateCollisionDetectorBounds(const std::array<double, 3>& min_bounds, const std::array<double, 3>& max_bounds);
  SpatialGrid getCollisionDetectorSpatialGrid();

  const PhysicsConfig physics_config;
  const SolverConfig solver_config;

  CollisionDetector collision_detector;

 private:
  void calculate_external_velocities(VecWrapper& U_ext, VecWrapper& F_ext_workspace, const std::vector<Particle>& local_particles, const MatWrapper& M, double dt);
  void apply_monolayer_constraints(VecWrapper& U, int n_local);
  void updateConstraintsFromSolution(std::vector<Constraint>& constraints, const VecWrapper& gamma, const VecWrapper& phi);

  std::mt19937 gen;
};
