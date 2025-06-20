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

class CollisionDetector {
 public:
  CollisionDetector(double collision_tolerance);

  std::vector<Constraint> detectCollisions(
      const std::vector<Particle>& local_particles,
      const std::vector<Particle>& ghost_particles,
      int constraint_iterations);

  static std::vector<Particle> gatherAllParticles(const std::vector<Particle>& local_particles);
  static std::vector<Particle> filterGhostParticles(
      const std::vector<Particle>& all_particles,
      const std::vector<Particle>& local_particles,
      double cutoff_distance);

 private:
  double collision_tolerance_;
  SpatialGrid spatial_grid_;

  CollisionDetails checkSpherocylinderCollision(
      const Particle& p1, const Particle& p2);

  std::array<double, 3> getParticleDirection(const Particle& p);
  std::pair<std::array<double, 3>, std::array<double, 3>> getParticleEndpoints(const Particle& p);

  // Simplified helper methods
  void updateSpatialGrid(
      const std::vector<Particle>& local_particles,
      const std::vector<Particle>& ghost_particles);

  void checkParticlePairsLocal(
      const std::vector<Particle>& particles1,
      const std::vector<Particle>& particles2,
      std::vector<Constraint>& constraints,
      int constraint_iterations);

  void checkParticlePairsCrossRank(
      const std::vector<Particle>& local_particles,
      const std::vector<Particle>& ghost_particles,
      std::vector<Constraint>& constraints,
      int constraint_iterations);

  const Particle* getParticle(
      int local_idx, int global_id,
      const std::vector<Particle>& local_particles,
      const std::unordered_map<int, const Particle*>& ghost_map);

  std::optional<Constraint> tryCreateConstraint(
      const Particle& p1, const Particle& p2,
      bool p1_local, bool p2_local, double tolerance, int constraint_iterations);
};