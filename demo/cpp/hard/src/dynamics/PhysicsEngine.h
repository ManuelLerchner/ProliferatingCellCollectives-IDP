#pragma once

#include <vector>

#include "Config.h"
#include "Constraint.h"
#include "MappingManager.h"
#include "Particle.h"
#include "util/PetscRaii.h"

class PhysicsEngine {
 public:
  PhysicsEngine(PhysicsConfig physics_config, SolverConfig solver_config);

  struct PhysicsMatrices {
    MatWrapper D, M, G;
    VecWrapper phi;
  };

  PhysicsMatrices calculateMatrices(const std::vector<Particle>& local_particles, const std::vector<Constraint>& local_constraints, MappingManager::Mappings mappings);

  VecWrapper solveConstraints(const PhysicsMatrices& matrices, double dt);

  const PhysicsConfig physics_config;
  const SolverConfig solver_config;
};

class ConstraintGenerator {
 public:
  std::vector<Constraint> generateConstraints(const std::vector<Particle>& particles);
};