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

void CollisionDetector::reset() {
  existing_contact_points_.clear();
}

bool CollisionDetector::isNewContactPoint(int gid1, int gid2, const std::array<double, 3>& contact_point) {
  using namespace utils::ArrayMath;

  // Ensure gid1 < gid2 for consistent lookup
  if (gid1 > gid2) {
    std::swap(gid1, gid2);
  }

  auto pair = std::make_pair(gid1, gid2);
  auto it = existing_contact_points_.find(pair);

  if (it == existing_contact_points_.end()) {
    // No existing contact points for this pair
    existing_contact_points_[pair].push_back(contact_point);
    return true;
  }

  // Check if the new contact point is too close to any existing ones
  for (const auto& existing_point : it->second) {
    double dist = distance(contact_point, existing_point);
    if (dist < CONTACT_POINT_TOLERANCE) {
      return false;
    }
  }

  // No close points found, add this one
  it->second.push_back(contact_point);
  return true;
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

void CollisionDetector::updateBounds(const std::array<double, 3>& min_bounds, const std::array<double, 3>& max_bounds) {
  double padding = 3.0;
  std::array<double, 3> padded_min = {min_bounds[0] - padding, min_bounds[1] - padding, min_bounds[2] - padding};
  std::array<double, 3> padded_max = {max_bounds[0] + padding, max_bounds[1] + padding, max_bounds[2] + padding};
  spatial_grid_ = SpatialGrid(cell_size_, padded_min, padded_max);
}

std::vector<Constraint> CollisionDetector::detectCollisions(
    ParticleManager& particle_manager,
    int constraint_iterations,
    double collision_tolerance) {
  std::vector<Constraint> constraints;

  // Simple approach: check all local-local pairs directly
  checkParticlePairs(particle_manager, constraints, constraint_iterations, collision_tolerance);

  // Sort constraints by their contact point's x-position to group them spatially.
  std::sort(constraints.begin(), constraints.end(), [](const Constraint& a, const Constraint& b) {
    return a.contactPoint[0] < b.contactPoint[0];
  });

  // Get the number of locally owned constraints.
  int local_num_constraints = constraints.size();

  int first_gid = 0;
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
    double collision_tolerance) {
  // Create ghost lookup map for efficiency
  std::unordered_map<PetscInt, const Particle*> ghost_map;
  for (const auto& ghost : particle_manager.ghost_particles) {
    ghost_map[ghost.getGID()] = &ghost;
  }

  auto collision_pairs = spatial_grid_.findPotentialCollisions(particle_manager.local_particles, particle_manager.ghost_particles);

  for (const auto& pair : collision_pairs) {
    Particle& p1 = getParticle(pair.gidI, particle_manager);
    Particle& p2 = getParticle(pair.gidJ, particle_manager);

    auto constraint = tryCreateConstraint(
        p1, p2,
        pair.is_localI,
        pair.is_localJ,
        constraint_iterations,
        collision_tolerance,
        particle_manager,
        pair);

    if (constraint.has_value()) {
      constraints.push_back(constraint.value());
    }
  }
}

Particle& CollisionDetector::getParticle(
    int global_id,
    ParticleManager& particle_manager) {
  for (auto& p : particle_manager.local_particles) {
    if (p.getGID() == global_id) {
      return p;
    }
  }
  for (auto& p : particle_manager.ghost_particles) {
    if (p.getGID() == global_id) {
      return p;
    }
  }
  throw std::runtime_error("Particle not found");
}

std::optional<Constraint> CollisionDetector::tryCreateConstraint(
    Particle& p1, Particle& p2,
    bool p1_local, bool p2_local,
    int constraint_iterations,
    double collision_tolerance,
    ParticleManager& particle_manager,
    const CollisionPair& pair) {
  using namespace utils::ArrayMath;
  // Get the line segments (cylindrical cores) of both particles
  auto [p1_start, p1_end] = getParticleEndpoints(p1);
  auto [p2_start, p2_end] = getParticleEndpoints(p2);

  // Use DCPQuery for accurate segment-segment distance calculation
  std::array<double, 3> closest_p1, closest_p2;
  double distance;
  auto result = distance_query(p1_start, p1_end, p2_start, p2_end);
  closest_p1 = result.closest[0];
  closest_p2 = result.closest[1];
  distance = result.distance;

  // Calculate radii and overlap
  double r1 = p1.getDiameter();
  double r2 = p2.getDiameter();
  double sum_radii = (r1 + r2) / 2.0;  // Average of diameters = sum of radii

  double signed_distance = distance - sum_radii;

  if (signed_distance > collision_tolerance) {
    return std::nullopt;
  }

  // Calculate contact point at the midpoint of the overlap region
  std::array<double, 3> contact_point = 0.5 * (closest_p1 + closest_p2);

  // After first iteration, only accept new contact points
  if (constraint_iterations > 0) {
    if (!isNewContactPoint(p1.getGID(), p2.getGID(), contact_point)) {
      return std::nullopt;
    }
  }

  p1.incrementNumConstraints();
  p2.incrementNumConstraints();

  using namespace utils::ArrayMath;

  const auto& pos1 = p1.getPosition();
  const auto& pos2 = p2.getPosition();
  std::array<double, 3> rPos1 = contact_point - pos1;
  std::array<double, 3> rPos2 = contact_point - pos2;

  auto orientation1 = utils::Quaternion::getDirectionVector(p1.getQuaternion());
  auto orientation2 = utils::Quaternion::getDirectionVector(p2.getQuaternion());

  std::array<double, 3> contact_normal = normalize(closest_p1 - closest_p2);

  auto stress1 = 0.5 * abs(dot(contact_normal, orientation1));
  auto stress2 = 0.5 * abs(dot(contact_normal, orientation2));

  bool owned_by_me;
  if (p1_local && p2_local) {
    owned_by_me = true;
  } else if (p1_local || p2_local) {
    const auto gid1 = p1.getGID();
    const auto gid2 = p2.getGID();

    owned_by_me = (gid1 < gid2) ? p1_local : p2_local;
  } else {
    throw std::runtime_error("Both particles are ghosts");
  }

  if (!owned_by_me) {
    return std::nullopt;
  }

  // Get indices from the collision pair
  int localIdxI = pair.localIdxI;
  int localIdxJ = pair.localIdxJ;

  auto constraint = Constraint(
      signed_distance,  // Now penetration is positive for overlap
      p1.getGID(), p2.getGID(),
      contact_normal,
      rPos1, rPos2,
      contact_point,
      stress1, stress2,
      -1,
      constraint_iterations,
      p1_local, p2_local,
      localIdxI, localIdxJ);

  int rank, total_rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  MPI_Comm_size(PETSC_COMM_WORLD, &total_rank);

  // Calculate center penalty using quadratic falloff
  double rank_center = (total_rank - 1) / 2.0;                                // Center point (works for both even and odd numbers)
  double max_distance = std::max(rank_center, total_rank - 1 - rank_center);  // Distance to furthest edge
  double relative_distance;
  if (total_rank == 1) {
    relative_distance = 0.0;  // Single rank is always at the center
  } else {
    relative_distance = std::abs(rank - rank_center) / max_distance;  // Normalized distance [0,1]
  }
  double center_penalty = 1.0 - relative_distance * relative_distance;  // Quadratic falloff

  constraint.gamma = signed_distance >= 0.0 ? 0.0 : -signed_distance * 5e2;

  return constraint;
}

SpatialGrid CollisionDetector::getSpatialGrid() {
  return spatial_grid_;
}