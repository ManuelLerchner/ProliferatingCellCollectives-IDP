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
    const std::unordered_set<Constraint, ConstraintHash, ConstraintEqual>& existing_constraints,
    int constraint_iterations) {
  // 1. Gather all particles from all processes
  auto all_particles = collision_detector_.gatherAllParticles(local_particles);

  // 2. Filter for ghost particles
  auto ghost_particles = collision_detector_.filterGhostParticles(all_particles, local_particles, ghost_cutoff_distance_);

  collision_detector_.updateSpatialGrid(local_particles, ghost_particles);

  // 3. Detect collisions, now passing the set of existing constraints
  return collision_detector_.detectCollisions(local_particles, ghost_particles, existing_constraints, constraint_iterations);
}