#pragma once

#include <functional>
#include <optional>
#include <random>
#include <vector>

#include "Constraint.h"
#include "simulation/Particle.h"
#include "util/Config.h"
#include "util/PetscRaii.h"

// Forward declaration
class ParticleManager;

class PhysicsEngine {
 public:
  PhysicsEngine(PhysicsConfig physics_config, SolverConfig solver_config);

  struct PhysicsMatrices {
    MatWrapper D, M, G, S;
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

  PhysicsMatrices calculateMatrices(const std::vector<Particle>& local_particles, const std::vector<Constraint>& local_constraints);

  SolverSolution solveConstraintsSingleConstraint(ParticleManager& particle_manager, double dt);
  SolverSolution solveConstraintsRecursiveConstraints(ParticleManager& particle_manager, double dt,int iter);

  const PhysicsConfig physics_config;
  const SolverConfig solver_config;

 private:
  void calculate_external_velocities(VecWrapper& U_ext, VecWrapper& F_ext_workspace, const std::vector<Particle>& local_particles, const MatWrapper& M, double dt);
  void apply_monolayer_constraints(VecWrapper& U, int n_local);

  std::mt19937 gen;
};
