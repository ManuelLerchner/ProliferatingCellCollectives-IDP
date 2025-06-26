#pragma once

#include <array>
#include <optional>
#include <unordered_map>
#include <vector>

#include "dynamics/Constraint.h"
#include "simulation/Particle.h"
#include "spatial/SpatialGrid.h"

struct CollisionDetails {
  bool collision_detected;   // Actual penetration/overlap
  bool potential_collision;  // Early warning: separation < 0.3 * diameter
  double overlap;
  double min_distance;
  double sum_radii;
  double separation_ratio;  // min_distance / (0.3 * average_diameter)
  std::array<double, 3> contact_point;
  std::array<double, 3> normal;
  std::array<double, 3> closest_p1;
  std::array<double, 3> closest_p2;
};

class ParticleManager;

class CollisionDetector {
 public:
  CollisionDetector(double collision_tolerance, double cell_size);

  void updateBounds(const std::array<double, 3>& min_bounds, const std::array<double, 3>& max_bounds);

  std::vector<Constraint> detectCollisions(
      ParticleManager& particle_manager,
      int constraint_iterations);

  // Simplified helper methods
  void updateSpatialGrid(ParticleManager& particle_manager);

  SpatialGrid getSpatialGrid();

 private:
  double collision_tolerance_;
  double cell_size_;
  SpatialGrid spatial_grid_;

  CollisionDetails checkSpherocylinderCollision(
      const Particle& p1, const Particle& p2);

  std::array<double, 3> getParticleDirection(const Particle& p);
  std::pair<std::array<double, 3>, std::array<double, 3>> getParticleEndpoints(const Particle& p);

  void checkParticlePairs(
      ParticleManager& particle_manager,
      std::vector<Constraint>& constraints,
      int constraint_iterations);

  const Particle* getParticle(
      int global_id,
      ParticleManager& particle_manager);

  std::optional<Constraint> tryCreateConstraint(
      const Particle& p1, const Particle& p2,
      bool p1_local, bool p2_local, double tolerance,
      int constraint_iterations);
};