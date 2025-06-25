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
#include "util/ParticleMPI.h"
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
    const Particle& p1, const Particle& p2) {
  using namespace utils::ArrayMath;

  CollisionDetails details;

  // Get the line segments (cylindrical cores) of both particles
  auto [p1_start, p1_end] = getParticleEndpoints(p1);
  auto [p2_start, p2_end] = getParticleEndpoints(p2);

  // Use DCPQuery for accurate segment-segment distance calculation

  std::array<double, 3> closest_p1, closest_p2;
  auto result = distance_query(p1_start, p1_end, p2_start, p2_end);
  details.min_distance = result.distance;
  closest_p1 = result.closest[0];
  closest_p2 = result.closest[1];

  // Store closest points
  details.closest_p1 = closest_p1;
  details.closest_p2 = closest_p2;

  // Calculate radii and overlap
  details.sum_radii = (p1.getDiameter() + p2.getDiameter()) / 2.0;
  details.overlap = details.sum_radii - details.min_distance;

  double separation = details.min_distance - details.sum_radii;

  details.potential_collision = separation < 0.3 * details.sum_radii;

  // Check for actual collision
  details.collision_detected = details.overlap > collision_tolerance_;

  // Calculate early warning threshold (separation < 0.3 * diameter)
  double average_diameter = (p1.getDiameter() + p2.getDiameter()) / 2.0;
  double warning_threshold = 0.3 * average_diameter;

  if (details.collision_detected || details.potential_collision) {
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

struct PairHash {
  template <class T1, class T2>
  std::size_t operator()(const std::pair<T1, T2>& p) const {
    auto h1 = std::hash<T1>{}(p.first);
    auto h2 = std::hash<T2>{}(p.second);
    return h1 ^ (h2 << 1);
  }
};

void CollisionDetector::updateBounds(const std::array<double, 3>& min_bounds, const std::array<double, 3>& max_bounds) {
  double padding = 3.0;
  std::array<double, 3> padded_min = {min_bounds[0] - padding, min_bounds[1] - padding, min_bounds[2] - padding};
  std::array<double, 3> padded_max = {max_bounds[0] + padding, max_bounds[1] + padding, max_bounds[2] + padding};
  spatial_grid_ = SpatialGrid(cell_size_, padded_min, padded_max);
}

std::vector<Constraint> CollisionDetector::detectCollisions(
    ParticleManager& particle_manager,
    const std::unordered_set<Constraint, ConstraintHash, ConstraintEqual>& existing_constraints,
    int constraint_iterations) {
  std::vector<Constraint> constraints;

  // Simple approach: check all local-local pairs directly
  checkParticlePairs(particle_manager, constraints, existing_constraints, constraint_iterations);

  // 1. Each process creates a list of its constraint pairs {min_gid, max_gid}
  std::vector<int> local_constraint_pairs;
  local_constraint_pairs.reserve(constraints.size() * 2);
  for (const auto& c : constraints) {
    local_constraint_pairs.push_back(std::min(c.gidI, c.gidJ));
    local_constraint_pairs.push_back(std::max(c.gidI, c.gidJ));
  }

  // 2. Gather the number of pairs from each process
  int local_num_pairs = local_constraint_pairs.size() / 2;
  int world_size;
  MPI_Comm_size(PETSC_COMM_WORLD, &world_size);
  std::vector<int> pair_counts(world_size);
  MPI_Allgather(&local_num_pairs, 1, MPI_INT, pair_counts.data(), 1, MPI_INT, PETSC_COMM_WORLD);

  // 3. Gather all pairs from all processes using Allgatherv
  std::vector<int> displacements(world_size, 0);
  int total_pairs = pair_counts[0];
  for (int i = 1; i < world_size; ++i) {
    displacements[i] = displacements[i - 1] + pair_counts[i - 1];
    total_pairs += pair_counts[i];
  }
  std::vector<int> all_constraint_pairs(total_pairs * 2);
  // Convert counts and displacements for MPI_Allgatherv (needs pairs -> ints)
  for (int i = 0; i < world_size; ++i) {
    pair_counts[i] *= 2;
    displacements[i] *= 2;
  }
  MPI_Allgatherv(local_constraint_pairs.data(), local_constraint_pairs.size(), MPI_INT,
                 all_constraint_pairs.data(), pair_counts.data(), displacements.data(), MPI_INT, PETSC_COMM_WORLD);

  // 4. Create a unique, sorted list of pairs to establish a global order
  std::vector<std::pair<int, int>> unique_pairs;
  unique_pairs.reserve(total_pairs);
  for (int i = 0; i < total_pairs; ++i) {
    unique_pairs.emplace_back(all_constraint_pairs[i * 2], all_constraint_pairs[i * 2 + 1]);
  }
  std::sort(unique_pairs.begin(), unique_pairs.end());
  unique_pairs.erase(std::unique(unique_pairs.begin(), unique_pairs.end()), unique_pairs.end());

  // 5. Create a map from the unique pair to a global ID
  std::unordered_map<std::pair<int, int>, int, PairHash> pair_to_gid;
  for (int i = 0; i < unique_pairs.size(); ++i) {
    pair_to_gid[unique_pairs[i]] = i;
  }

  // 6. Assign the correct global ID to each local constraint
  for (auto& c : constraints) {
    std::pair<int, int> p = {std::min(c.gidI, c.gidJ), std::max(c.gidI, c.gidJ)};
    c.gid = pair_to_gid[p];
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
    const std::unordered_set<Constraint, ConstraintHash, ConstraintEqual>& existing_constraints,
    int constraint_iterations) {
  // Create ghost lookup map for efficiency
  std::unordered_map<PetscInt, const Particle*> ghost_map;
  for (const auto& ghost : particle_manager.ghost_particles) {
    ghost_map[ghost.getGID()] = &ghost;
  }

  auto collision_pairs = spatial_grid_.findPotentialCollisions(particle_manager.local_particles, particle_manager.ghost_particles);

  for (const auto& pair : collision_pairs) {
    const Particle* p1 = getParticle(pair.gidI, particle_manager);
    const Particle* p2 = getParticle(pair.gidJ, particle_manager);

    if (!p1 || !p2) {
      // This can happen if a particle is not found, though it would be unexpected
      // in the current logic. Good to have a safeguard.
      continue;
    }

    auto constraint = tryCreateConstraint(
        *p1, *p2,
        pair.is_localI,
        pair.is_localJ,
        collision_tolerance_,
        existing_constraints,
        constraint_iterations);

    if (constraint.has_value()) {
      constraints.push_back(constraint.value());
    }
  }
}

const Particle* CollisionDetector::getParticle(
    int global_id,
    ParticleManager& particle_manager) {
  for (const auto& p : particle_manager.local_particles) {
    if (p.getGID() == global_id) {
      return &p;
    }
  }
  for (const auto& p : particle_manager.ghost_particles) {
    if (p.getGID() == global_id) {
      return &p;
    }
  }
  return nullptr;
}

std::optional<Constraint> CollisionDetector::tryCreateConstraint(
    const Particle& p1, const Particle& p2,
    bool p1_local, bool p2_local, double tolerance,
    const std::unordered_set<Constraint, ConstraintHash, ConstraintEqual>& existing_constraints,
    int constraint_iterations) {
  CollisionDetails details = checkSpherocylinderCollision(p1, p2);

  if (!details.collision_detected && !details.potential_collision) {
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

  auto local_id1 = p1_local ? p1.getLocalID() : -1;
  auto local_id2 = p2_local ? p2.getLocalID() : -1;

  bool owned_by_me;
  if (p1_local && p2_local) {
    owned_by_me = true;   // Owned if both particles are local
  } else if (p1_local) {  // p2 is a ghost
    owned_by_me = p1.setGID() < p2.setGID();
  } else if (p2_local) {  // p1 is a ghost
    owned_by_me = p2.setGID() < p1.setGID();
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
      p1.setGID(), p2.setGID(),
      local_id1, local_id2,
      details.normal,
      rPos1, rPos2,
      details.contact_point,
      stress1, stress2,
      constraint_iterations,
      p1.setGID());

  if (existing_constraints.contains(constraint)) {
    return std::nullopt;
  }

  return constraint;
}

SpatialGrid CollisionDetector::getSpatialGrid() {
  return spatial_grid_;
}