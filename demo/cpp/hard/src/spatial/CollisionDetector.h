#pragma once

#include <array>
#include <vector>

#include "dynamics/Constraint.h"
#include "simulation/Particle.h"
#include "spatial/SpatialGrid.h"

class CollisionDetector {
 public:
  CollisionDetector(double collision_tolerance = 1e-6);

  std::vector<Constraint> detectCollisions(
      const std::vector<Particle>& local_particles,
      const std::vector<Particle>& ghost_particles);

  static std::vector<Particle> gatherAllParticles(const std::vector<Particle>& local_particles);
  static std::vector<Particle> filterGhostParticles(
      const std::vector<Particle>& all_particles,
      const std::vector<Particle>& local_particles,
      double cutoff_distance);

 private:
  double collision_tolerance_;
  SpatialGrid spatial_grid_;

  bool checkSpherocylinderCollision(
      const Particle& p1, const Particle& p2,
      double& overlap, std::array<double, 3>& contact_point,
      std::array<double, 3>& normal);

  std::array<double, 3> getParticleDirection(const Particle& p);
  std::pair<std::array<double, 3>, std::array<double, 3>> getParticleEndpoints(const Particle& p);
};