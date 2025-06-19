#include "CollisionDetector.h"

#include <mpi.h>
#include <petsc.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <unordered_set>

#include "util/ArrayMath.h"
#include "util/Quaternion.h"
#include "util/SpherocylinderCell.h"

// CollisionDetector Implementation
CollisionDetector::CollisionDetector(double collision_tolerance)
    : collision_tolerance_(collision_tolerance), spatial_grid_(1.0, {-10, -10, -10}, {10, 10, 10}) {
}

std::pair<std::array<double, 3>, std::array<double, 3>> CollisionDetector::getParticleEndpoints(const Particle& p) {
  using namespace utils::ArrayMath;
  const auto& pos = p.getPosition();
  auto dir = utils::Quaternion::getDirectionVector(p.getQuaternion());
  double diameter = 0.5;
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
  DCPQuery distance_query;
  std::array<double, 3> closest_p1, closest_p2;
  details.min_distance = distance_query(p1_start, p1_end, p2_start, p2_end, closest_p1, closest_p2);

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

std::vector<Constraint> CollisionDetector::detectCollisions(
    const std::vector<Particle>& local_particles,
    const std::vector<Particle>& ghost_particles,
    int constraint_iterations) {
  std::vector<Constraint> constraints;

  // Simple approach: check all local-local pairs directly
  checkParticlePairs(local_particles, local_particles, constraints, true, constraint_iterations);

  // Check local-ghost pairs using spatial grid
  if (!ghost_particles.empty()) {
    updateSpatialGrid(local_particles, ghost_particles);
    checkSpatialGridPairs(local_particles, ghost_particles, constraints, constraint_iterations);
  }

  return constraints;
}

void CollisionDetector::updateSpatialGrid(
    const std::vector<Particle>& local_particles,
    const std::vector<Particle>& ghost_particles) {
  if (local_particles.empty()) return;

  // Calculate bounds
  std::array<double, 3> min_pos = local_particles[0].getPosition();
  std::array<double, 3> max_pos = min_pos;

  auto updateBounds = [&](const Particle& p) {
    const auto& pos = p.getPosition();
    for (int i = 0; i < 3; ++i) {
      min_pos[i] = std::min(min_pos[i], pos[i] - p.getLength());
      max_pos[i] = std::max(max_pos[i], pos[i] + p.getLength());
    }
  };

  for (const auto& p : local_particles) updateBounds(p);
  for (const auto& p : ghost_particles) updateBounds(p);

  // Calculate cell size
  double max_particle_size = 0.0;
  for (const auto& p : local_particles) {
    max_particle_size = std::max(max_particle_size, p.getLength());
  }
  for (const auto& p : ghost_particles) {
    max_particle_size = std::max(max_particle_size, p.getLength());
  }

  double cell_size = std::max(1.0, max_particle_size);
  spatial_grid_ = SpatialGrid(cell_size, min_pos, max_pos);
}

void CollisionDetector::checkParticlePairs(
    const std::vector<Particle>& particles1,
    const std::vector<Particle>& particles2,
    std::vector<Constraint>& constraints,
    bool same_array,
    int constraint_iterations) {
  int start_j = same_array ? 1 : 0;

  for (int i = 0; i < particles1.size(); ++i) {
    int j_start = same_array ? i + start_j : 0;

    for (int j = j_start; j < particles2.size(); ++j) {
      auto constraint = tryCreateConstraint(
          particles1[i], particles2[j],
          i, same_array ? j : -1,  // local indices
          true, same_array,
          collision_tolerance_,
          constraint_iterations);  // locality flags

      if (constraint.has_value()) {
        constraints.push_back(constraint.value());
      }
    }
  }
}

void CollisionDetector::checkSpatialGridPairs(
    const std::vector<Particle>& local_particles,
    const std::vector<Particle>& ghost_particles,
    std::vector<Constraint>& constraints,
    int constraint_iterations) {
  // Create ghost lookup map for efficiency
  std::unordered_map<int, const Particle*> ghost_map;
  for (const auto& ghost : ghost_particles) {
    ghost_map[ghost.setGID()] = &ghost;
  }

  auto collision_pairs = spatial_grid_.findPotentialCollisions(local_particles, ghost_particles);

  for (const auto& pair : collision_pairs) {
    // Skip local-local pairs (already handled)
    if (pair.local_idx_i >= 0 && pair.local_idx_j >= 0) continue;

    // Get particles
    const Particle* p1 = getParticle(pair.local_idx_i, pair.global_id_i, local_particles, ghost_map);
    const Particle* p2 = getParticle(pair.local_idx_j, pair.global_id_j, local_particles, ghost_map);

    if (!p1 || !p2) continue;

    auto constraint = tryCreateConstraint(
        *p1, *p2,
        pair.local_idx_i, pair.local_idx_j,
        pair.local_idx_i >= 0, pair.local_idx_j >= 0,
        collision_tolerance_,
        constraint_iterations);

    if (constraint.has_value()) {
      constraints.push_back(constraint.value());
    }
  }
}

const Particle* CollisionDetector::getParticle(
    int local_idx, int global_id,
    const std::vector<Particle>& local_particles,
    const std::unordered_map<int, const Particle*>& ghost_map) {
  if (local_idx >= 0 && local_idx < local_particles.size()) {
    return &local_particles[local_idx];
  }

  auto it = ghost_map.find(global_id);
  return (it != ghost_map.end()) ? it->second : nullptr;
}

std::optional<Constraint> CollisionDetector::tryCreateConstraint(
    const Particle& p1, const Particle& p2,
    int local_i, int local_j, bool p1_local, bool p2_local, double tolerance, int constraint_iterations) {
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

  return Constraint(
      -details.overlap,
      details.overlap > tolerance,
      p1.setGID(), p2.setGID(),
      local_i, local_j,
      p1_local, p2_local,
      details.normal,
      rPos1, rPos2,
      details.contact_point,
      orientation1, orientation2,
      constraint_iterations);
}

std::vector<Particle> CollisionDetector::gatherAllParticles(const std::vector<Particle>& local_particles) {
  PetscMPIInt rank, size;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  MPI_Comm_size(PETSC_COMM_WORLD, &size);

  // Gather particle counts from all ranks
  int local_count = static_cast<int>(local_particles.size());
  std::vector<int> counts(size);
  MPI_Allgather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, PETSC_COMM_WORLD);

  // Calculate displacements
  std::vector<int> displacements(size);
  displacements[0] = 0;
  for (int i = 1; i < size; ++i) {
    displacements[i] = displacements[i - 1] + counts[i - 1];
  }

  int total_particles = displacements[size - 1] + counts[size - 1];

  // Serialize local particles
  std::vector<double> local_data;
  for (const auto& p : local_particles) {
    const auto& pos = p.getPosition();
    const auto& quat = p.getQuaternion();

    local_data.push_back(static_cast<double>(p.setGID()));
    local_data.insert(local_data.end(), pos.begin(), pos.end());
    local_data.insert(local_data.end(), quat.begin(), quat.end());
    local_data.push_back(p.getLength());
    local_data.push_back(p.getDiameter());
    local_data.push_back(p.getLength());  // Use current length as l0 since we don't store it
  }

  // Gather all particle data
  int data_per_particle = 11;  // id + 3 pos + 4 quat + length + diameter + l0
  std::vector<int> data_counts(size);
  std::vector<int> data_displacements(size);
  for (int i = 0; i < size; ++i) {
    data_counts[i] = counts[i] * data_per_particle;
    data_displacements[i] = displacements[i] * data_per_particle;
  }

  std::vector<double> all_data(total_particles * data_per_particle);
  MPI_Allgatherv(local_data.data(), local_data.size(), MPI_DOUBLE,
                 all_data.data(), data_counts.data(), data_displacements.data(), MPI_DOUBLE, PETSC_COMM_WORLD);

  // Deserialize particles
  std::vector<Particle> all_particles;
  for (int i = 0; i < total_particles; ++i) {
    int offset = i * data_per_particle;

    PetscInt id = static_cast<PetscInt>(all_data[offset]);
    std::array<double, 3> pos = {all_data[offset + 1], all_data[offset + 2], all_data[offset + 3]};
    std::array<double, 4> quat = {all_data[offset + 4], all_data[offset + 5], all_data[offset + 6], all_data[offset + 7]};
    double length = all_data[offset + 8];
    double diameter = all_data[offset + 9];
    double l0 = all_data[offset + 10];

    all_particles.emplace_back(id, pos, quat, length, l0, diameter);
  }

  return all_particles;
}

std::vector<Particle> CollisionDetector::filterGhostParticles(
    const std::vector<Particle>& all_particles,
    const std::vector<Particle>& local_particles,
    double cutoff_distance) {
  using namespace utils::ArrayMath;
  std::vector<Particle> ghost_particles;

  // Create set of local particle IDs for quick lookup
  std::unordered_set<PetscInt> local_ids;
  for (const auto& p : local_particles) {
    local_ids.insert(p.setGID());
  }

  // Find particles within cutoff distance of local particles
  for (const auto& candidate : all_particles) {
    // Skip if this is a local particle
    if (local_ids.find(candidate.setGID()) != local_ids.end()) {
      continue;
    }

    // Check distance to any local particle using ArrayMath
    bool within_cutoff = false;
    const auto& cand_pos = candidate.getPosition();

    for (const auto& local_p : local_particles) {
      const auto& local_pos = local_p.getPosition();

      // Use ArrayMath distance function for cleaner code
      double dist = distance(cand_pos, local_pos);

      // Include some margin for particle sizes
      double margin = (candidate.getLength() + local_p.getLength()) / 2.0;

      if (dist < cutoff_distance + margin) {
        within_cutoff = true;
        break;
      }
    }

    if (within_cutoff) {
      ghost_particles.push_back(candidate);
    }
  }

  return ghost_particles;
}