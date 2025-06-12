#include "ConstraintGenerator.h"

#include "CollisionDetector.h"
#include "dynamics/Constraint.h"
#include "simulation/Particle.h"

ConstraintGenerator::ConstraintGenerator(double collision_tolerance, double ghost_cutoff_distance)
    : collision_detector_(CollisionDetector(collision_tolerance)), ghost_cutoff_distance_(ghost_cutoff_distance) {
}

std::vector<Constraint> ConstraintGenerator::generateConstraints(const std::vector<Particle>& local_particles, int constraint_iterations) {
  // Exchange ghost particles with neighboring ranks
  auto ghost_particles = CollisionDetector::gatherAllParticles(local_particles);
  auto filtered_ghosts = CollisionDetector::filterGhostParticles(ghost_particles, local_particles, ghost_cutoff_distance_);

  // Detect collisions between local and ghost particles
  return collision_detector_.detectCollisions(local_particles, filtered_ghosts, constraint_iterations);
}