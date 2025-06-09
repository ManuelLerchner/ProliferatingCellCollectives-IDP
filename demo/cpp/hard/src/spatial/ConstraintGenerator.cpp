#include "ConstraintGenerator.h"

#include "CollisionDetector.h"
#include "dynamics/Constraint.h"
#include "simulation/Particle.h"

ConstraintGenerator::ConstraintGenerator()
    : collision_detector_(CollisionDetector(1e-6)), ghost_cutoff_distance_(5.0) {
}

std::vector<Constraint> ConstraintGenerator::generateConstraints(const std::vector<Particle>& local_particles) {
  // Exchange ghost particles with neighboring ranks
  auto ghost_particles = CollisionDetector::gatherAllParticles(local_particles);
  auto filtered_ghosts = CollisionDetector::filterGhostParticles(ghost_particles, local_particles, ghost_cutoff_distance_);

  // Detect collisions between local and ghost particles
  return collision_detector_.detectCollisions(local_particles, filtered_ghosts);
}