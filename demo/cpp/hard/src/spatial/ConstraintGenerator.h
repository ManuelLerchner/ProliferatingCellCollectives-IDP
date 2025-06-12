#pragma once

#include <vector>

#include "CollisionDetector.h"
#include "dynamics/Constraint.h"
#include "simulation/Particle.h"

class ConstraintGenerator {
 public:
  ConstraintGenerator(double collision_tolerance, double ghost_cutoff_distance);
  std::vector<Constraint> generateConstraints(const std::vector<Particle>& particles, int constraint_iterations);

 private:
  CollisionDetector collision_detector_;
  double ghost_cutoff_distance_;
};