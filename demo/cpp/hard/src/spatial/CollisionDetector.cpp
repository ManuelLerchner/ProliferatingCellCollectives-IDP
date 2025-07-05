#include "CollisionDetector.h"

#include <mpi.h>
#include <petsc.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "simulation/Particle.h"
#include "simulation/ParticleManager.h"
#include "util/ArrayMath.h"
#include "util/Quaternion.h"
#include "util/SpherocylinderCell.h"

utils::geometry::DCPSegment3Segment3<double> distance_query;

// CollisionDetector Implementation
CollisionDetector::CollisionDetector(double collision_tolerance, double cell_size)
    : collision_tolerance_(collision_tolerance), cell_size_(cell_size), spatial_grid_(cell_size, {-1, -1, -1}, {1, 1, 1}) {
}

std::pair<std::array<double, 3>, std::array<double, 3>> CollisionDetector::getParticleEndpoints(const Particle& p) {
  using namespace utils::ArrayMath;
  const auto& pos = p.getPosition();
  auto dir = utils::Quaternion::getDirectionVector(p.getQuaternion());
  double diameter = p.getDiameter();
  double length = p.getLength();

  // Calculate endpoints of the rod using ArrayMath operations
  double half_core_length = length / 2 - diameter / 2;
  std::array<double, 3> half_length_vector = half_core_length * dir;
  std::array<double, 3> start_point = pos - half_length_vector;
  std::array<double, 3> end_point = pos + half_length_vector;

  return {start_point, end_point};
}

CollisionDetails CollisionDetector::checkSpherocylinderCollision(
    const Particle& p1, const Particle& p2, double future_colission_factor) {
  using namespace utils::ArrayMath;

  double sum_radii = (p1.getDiameter() + p2.getDiameter()) / 2.0;

  double sum_lengths = p1.getLength() + p2.getLength();
  double dist_centers_sq = magnitude_squared(p1.getPosition() - p2.getPosition());

  if (dist_centers_sq > sum_lengths * sum_lengths) {
    return CollisionDetails{false, false, 0.0, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
  }

  CollisionDetails details;

  // Get the line segments (cylindrical cores) of both particles
  auto [p1_start, p1_end] = getParticleEndpoints(p1);
  auto [p2_start, p2_end] = getParticleEndpoints(p2);

  // Use DCPQuery for accurate segment-segment distance calculation

  std::array<double, 3> closest_p1, closest_p2;
  auto result = distance_query(p1_start, p1_end, p2_start, p2_end);
  closest_p1 = result.closest[0];
  closest_p2 = result.closest[1];

  // Store closest points
  details.closest_p1 = closest_p1;
  details.closest_p2 = closest_p2;

  // Calculate radii and overlap
  details.overlap = sum_radii - result.distance;

  details.future_collision = -details.overlap < future_colission_factor * sum_radii;

  details.collision_detected = details.overlap > collision_tolerance_;

  if (details.collision_detected || details.future_collision) {
    // Calculate contact point and normal using ArrayMath
    details.contact_point = 0.5 * (closest_p1 + closest_p2);

    std::array<double, 3> direction = closest_p1 - closest_p2;
    details.normal = normalize(direction);
  }
  return details;
}

std::array<double, 3> CollisionDetector::getParticleDirection(const Particle& p) {
  return utils::Quaternion::getDirectionVector(p.getQuaternion());
}

void CollisionDetector::updateBounds(const std::array<double, 3>& min_bounds, const std::array<double, 3>& max_bounds) {
  double padding = 3.0;
  std::array<double, 3> padded_min = {min_bounds[0] - padding, min_bounds[1] - padding, min_bounds[2] - padding};
  std::array<double, 3> padded_max = {max_bounds[0] + padding, max_bounds[1] + padding, max_bounds[2] + padding};
  spatial_grid_ = SpatialGrid(cell_size_, padded_min, padded_max);
}

std::vector<Constraint> CollisionDetector::detectCollisions(
    ParticleManager& particle_manager,
    int constraint_iterations,
    double future_colission_factor) {
  std::vector<Constraint> constraints;

  // Simple approach: check all local-local pairs directly
  checkParticlePairs(particle_manager, constraints, constraint_iterations, future_colission_factor);

  // Get the number of locally owned constraints.
  int local_num_constraints = constraints.size();

  int first_gid;
  MPI_Scan(&local_num_constraints, &first_gid, 1, MPI_INT, MPI_SUM, PETSC_COMM_WORLD);
  first_gid -= local_num_constraints;

  // Assign contiguous global IDs to the local constraints.
  for (int i = 0; i < local_num_constraints; ++i) {
    constraints[i].gid = first_gid + i;
  }

  return constraints;
}

void CollisionDetector::updateSpatialGrid(ParticleManager& particle_manager) {
  if (particle_manager.local_particles.empty()) return;

  // Calculate bounds
  std::array<double, 3> min_pos = particle_manager.local_particles[0].getPosition();
  std::array<double, 3> max_pos = min_pos;

  auto updateBounds = [&](const Particle& p) {
    const auto& pos = p.getPosition();
    for (int i = 0; i < 3; ++i) {
      min_pos[i] = std::min(min_pos[i], pos[i] - p.getLength());
      max_pos[i] = std::max(max_pos[i], pos[i] + p.getLength());
    }
  };

  for (const auto& p : particle_manager.local_particles) updateBounds(p);
  for (const auto& p : particle_manager.ghost_particles) updateBounds(p);

  // Calculate cell size
  double max_particle_size = 0.0;
  for (const auto& p : particle_manager.local_particles) {
    max_particle_size = std::max(max_particle_size, p.getLength());
  }
  for (const auto& p : particle_manager.ghost_particles) {
    max_particle_size = std::max(max_particle_size, p.getLength());
  }

  double cell_size = std::max(1.0, max_particle_size);
  spatial_grid_ = SpatialGrid(cell_size, min_pos, max_pos);
}

void CollisionDetector::checkParticlePairs(
    ParticleManager& particle_manager,
    std::vector<Constraint>& constraints,
    int constraint_iterations,
    double future_colission_factor) {
  // Create ghost lookup map for efficiency
  std::unordered_map<PetscInt, const Particle*> ghost_map;
  for (const auto& ghost : particle_manager.ghost_particles) {
    ghost_map[ghost.getGID()] = &ghost;
  }

  auto collision_pairs = spatial_grid_.findPotentialCollisions(particle_manager.local_particles, particle_manager.ghost_particles);

  for (const auto& pair : collision_pairs) {
    const Particle& p1 = getParticle(pair.gidI, particle_manager);
    const Particle& p2 = getParticle(pair.gidJ, particle_manager);

    auto constraint = tryCreateConstraint(
        p1, p2,
        future_colission_factor,
        pair.is_localI,
        pair.is_localJ,
        collision_tolerance_,
        constraint_iterations);

    if (constraint.has_value()) {
      constraints.push_back(constraint.value());
    }
  }
}

const Particle& CollisionDetector::getParticle(
    int global_id,
    ParticleManager& particle_manager) {
  for (const auto& p : particle_manager.local_particles) {
    if (p.getGID() == global_id) {
      return p;
    }
  }
  for (const auto& p : particle_manager.ghost_particles) {
    if (p.getGID() == global_id) {
      return p;
    }
  }
  throw std::runtime_error("Particle not found");
}

std::optional<Constraint> CollisionDetector::tryCreateConstraint(
    const Particle& p1, const Particle& p2,
    double future_colission_factor,
    bool p1_local, bool p2_local, double tolerance,
    int constraint_iterations) {
  CollisionDetails details = checkSpherocylinderCollision(p1, p2, future_colission_factor);

  if (!details.collision_detected && !details.future_collision) {
    return std::nullopt;
  }

  using namespace utils::ArrayMath;

  const auto& pos1 = p1.getPosition();
  const auto& pos2 = p2.getPosition();
  std::array<double, 3> rPos1 = details.contact_point - pos1;
  std::array<double, 3> rPos2 = details.contact_point - pos2;

  auto orientation1 = utils::Quaternion::getDirectionVector(p1.getQuaternion());
  auto orientation2 = utils::Quaternion::getDirectionVector(p2.getQuaternion());

  auto stress1 = 0.5 * abs(dot(details.normal, orientation1));
  auto stress2 = 0.5 * abs(dot(details.normal, orientation2));

  bool owned_by_me;
  if (p1_local && p2_local) {
    owned_by_me = true;   // Owned if both particles are local
  } else if (p1_local) {  // p2 is a ghost
    owned_by_me = p1.getGID() < p2.getGID();
  } else if (p2_local) {  // p1 is a ghost
    owned_by_me = p2.getGID() < p1.getGID();
  } else {
    owned_by_me = false;  // Should not happen if collision detection is correct
    throw std::runtime_error("Both particles are ghosts");
  }

  if (!owned_by_me) {
    return std::nullopt;
  }

  auto constraint = Constraint(
      -details.overlap,
      details.overlap > tolerance,
      p1.getGID(), p2.getGID(),
      details.normal,
      rPos1, rPos2,
      details.contact_point,
      stress1, stress2,
      constraint_iterations,
      -1);

  return constraint;
}

SpatialGrid CollisionDetector::getSpatialGrid() {
  return spatial_grid_;
}