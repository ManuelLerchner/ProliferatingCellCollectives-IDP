#pragma once

#include <vector>

#include "Constraint.h"
#include "MappingManager.h"
#include "simulation/Particle.h"
#include "util/Config.h"
#include "util/PetscRaii.h"

#define MAX_CONSTRAINT_ITERATIONS 3

// Forward declaration
class ParticleManager;

class PhysicsEngine {
 public:
  PhysicsEngine(PhysicsConfig physics_config, SolverConfig solver_config);

  struct PhysicsMatrices {
    MatWrapper D, M, G;
    VecWrapper phi;
  };

  struct PhysicsSolution {
    const VecWrapper& deltaC;
    const VecWrapper& f;
    const VecWrapper& u;
  };

  struct SolverSolution {
    std::vector<Constraint> constraints;
    const long long constraint_iterations;
    const long long bbpgd_iterations;
    const double residum;
  };

  PhysicsMatrices calculateMatrices(const std::vector<Particle>& local_particles, const std::vector<Constraint>& local_constraints);

  SolverSolution solveConstraintsSingleConstraint(ParticleManager& particle_manager, double dt);
  SolverSolution solveConstraintsRecursiveConstraints(ParticleManager& particle_manager, double dt);

  const PhysicsConfig physics_config;
  const SolverConfig solver_config;
};
