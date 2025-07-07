#pragma once

#include <array>
#include <optional>
#include <unordered_map>
#include <vector>

#include "dynamics/Constraint.h"
#include "simulation/Particle.h"
#include "spatial/SpatialGrid.h"

class ParticleManager;

class CollisionDetector {
 public:
  CollisionDetector(double collision_tolerance, double cell_size);

  void updateBounds(const std::array<double, 3>& min_bounds, const std::array<double, 3>& max_bounds);

  std::vector<Constraint> detectCollisions(
      ParticleManager& particle_manager,
      int constraint_iterations,
      std::function<bool(const Constraint&)> filter_constraints);

  // Simplified helper methods
  void updateSpatialGrid(ParticleManager& particle_manager);

  SpatialGrid getSpatialGrid();

 private:
  double collision_tolerance_;
  double cell_size_;
  SpatialGrid spatial_grid_;

  std::array<double, 3> getParticleDirection(const Particle& p);
  std::pair<std::array<double, 3>, std::array<double, 3>> getParticleEndpoints(const Particle& p);

  void checkParticlePairs(
      ParticleManager& particle_manager,
      std::vector<Constraint>& constraints,
      int constraint_iterations,
      std::function<bool(const Constraint&)> filter_constraints);

  Particle& getParticle(
      int global_id,
      ParticleManager& particle_manager);

  std::optional<Constraint> tryCreateConstraint(
      Particle& p1, Particle& p2,
      bool p1_local, bool p2_local,
      int constraint_iterations);
};