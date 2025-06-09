#pragma once

#include <vector>

#include "CollisionDetector.h"
#include "dynamics/Constraint.h"
#include "simulation/Particle.h"

class ConstraintGenerator {
 public:
  ConstraintGenerator();
  std::vector<Constraint> generateConstraints(const std::vector<Particle>& particles);

 private:
  CollisionDetector collision_detector_;
  double ghost_cutoff_distance_;
};