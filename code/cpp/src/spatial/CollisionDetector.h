#pragma once

#include <array>
#include <functional>  // for std::hash
#include <optional>
#include <unordered_map>
#include <utility>  // for std::pair
#include <vector>

#include "dynamics/Constraint.h"
#include "simulation/Particle.h"
#include "spatial/SpatialGrid.h"

class ParticleManager;

// Hash function for std::pair to use in unordered_map
struct pair_hash {
  template <class T1, class T2>
  std::size_t operator()(const std::pair<T1, T2>& p) const {
    auto h1 = std::hash<T1>{}(p.first);
    auto h2 = std::hash<T2>{}(p.second);
    return h1 ^ h2;
  }
};

class CollisionDetector {
 public:
  CollisionDetector(double collision_tolerance, double cell_size);

  void updateBounds(const std::array<double, 3>& min_bounds, const std::array<double, 3>& max_bounds);

  std::vector<Constraint> detectCollisions(
      ParticleManager& particle_manager,
      int constraint_iterations,
      double collision_tolerance);

  // Simplified helper methods
  void updateSpatialGrid(ParticleManager& particle_manager);

  SpatialGrid getSpatialGrid();

  // Reset the collision detector state (e.g., between solver runs)
  void reset();

  std::pair<std::array<double, 3>, std::array<double, 3>> getParticleEndpoints(const Particle& p);

 private:
  double collision_tolerance_;
  double cell_size_;
  SpatialGrid spatial_grid_;

  // Track existing contact points between particle pairs to avoid duplicates
  std::unordered_map<std::pair<int, int>, std::vector<std::array<double, 3>>, pair_hash> existing_contact_points_;
  static constexpr double CONTACT_POINT_TOLERANCE = 0.00;  // Distance below which contact points are considered the same

  std::array<double, 3> getParticleDirection(const Particle& p);

  void checkParticlePairs(
      ParticleManager& particle_manager,
      std::vector<Constraint>& constraints,
      int constraint_iterations,
      double collision_tolerance);

  Particle& getParticle(
      int global_id,
      ParticleManager& particle_manager);

  std::optional<Constraint> tryCreateConstraint(
      Particle& p1, Particle& p2,
      bool p1_local, bool p2_local,
      int constraint_iterations,
      double collision_tolerance,
      ParticleManager& particle_manager,
      const CollisionPair& pair);

  // Helper to check if a contact point is too close to existing ones
  bool isNewContactPoint(int gid1, int gid2, const std::array<double, 3>& contact_point);
};