#include "ConstraintGenerator.h"

#include <unordered_set>
#include <vector>

#include "CollisionDetector.h"
#include "dynamics/Constraint.h"
#include "simulation/Particle.h"
#include "util/ParticleMPI.h"

ConstraintGenerator::ConstraintGenerator(double collision_tolerance, double ghost_cutoff_distance)
    : collision_detector_(CollisionDetector(collision_tolerance)), ghost_cutoff_distance_(ghost_cutoff_distance) {
}

std::vector<Constraint> ConstraintGenerator::generateConstraints(
    const std::vector<Particle>& local_particles,
    const std::vector<Particle>& ghost_particles,
    const std::unordered_set<Constraint, ConstraintHash, ConstraintEqual>& existing_constraints,
    int constraint_iterations) {
  // 3. Detect collisions, now passing the set of existing constraints
  return collision_detector_.detectCollisions(local_particles, ghost_particles, existing_constraints, constraint_iterations);
}