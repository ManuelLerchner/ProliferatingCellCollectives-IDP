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
#include "util/ArrayMath.h"
#include "util/ParticleMPI.h"
#include "util/Quaternion.h"
#include "util/SpherocylinderCell.h"

utils::geometry::DCPSegment3Segment3<double> distance_query;

// CollisionDetector Implementation
CollisionDetector::CollisionDetector(double collision_tolerance)
    : collision_tolerance_(collision_tolerance), spatial_grid_(1.0, {-10, -10, -10}, {10, 10, 10}) {
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

std::vector<Constraint> CollisionDetector::detectCollisions(
    const std::vector<Particle>& local_particles,
    const std::vector<Particle>& ghost_particles,
    const std::unordered_set<Constraint, ConstraintHash, ConstraintEqual>& existing_constraints,
    int constraint_iterations) {
  std::vector<Constraint> constraints;

  // Simple approach: check all local-local pairs directly
  checkParticlePairsLocal(local_particles, local_particles, constraints, existing_constraints, constraint_iterations);

  // Check local-ghost pairs using spatial grid
  if (!ghost_particles.empty()) {
    checkParticlePairsCrossRank(local_particles, ghost_particles, constraints, existing_constraints, constraint_iterations);
  }

  // --- Globally consistent GID assignment ---

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

void CollisionDetector::checkParticlePairsLocal(
    const std::vector<Particle>& particles1,
    const std::vector<Particle>& particles2,
    std::vector<Constraint>& constraints,
    const std::unordered_set<Constraint, ConstraintHash, ConstraintEqual>& existing_constraints,
    int constraint_iterations) {
  for (int i = 0; i < particles1.size(); ++i) {
    int j_start = i + 1;

    for (int j = j_start; j < particles2.size(); ++j) {
      auto constraint = tryCreateConstraint(
          particles1[i], particles2[j],
          true, true,
          collision_tolerance_,
          existing_constraints,
          constraint_iterations);

      if (constraint.has_value()) {
        constraints.push_back(constraint.value());
      }
    }
  }
}

void CollisionDetector::checkParticlePairsCrossRank(
    const std::vector<Particle>& local_particles,
    const std::vector<Particle>& ghost_particles,
    std::vector<Constraint>& constraints,
    const std::unordered_set<Constraint, ConstraintHash, ConstraintEqual>& existing_constraints,
    int constraint_iterations) {
  // Create ghost lookup map for efficiency
  std::unordered_map<PetscInt, const Particle*> ghost_map;
  for (const auto& ghost : ghost_particles) {
    ghost_map[ghost.setGID()] = &ghost;
  }

  auto collision_pairs = spatial_grid_.findPotentialCollisions(local_particles, ghost_particles);

  for (const auto& pair : collision_pairs) {
    // Skip local-local pairs (already handled by checkParticlePairsLocal)
    if (pair.is_localI && pair.is_localJ) {
      continue;
    }

    const Particle* p1 = nullptr;
    if (pair.is_localI) {
      // Find local particle by GID
      for (const auto& p : local_particles) {
        if (p.getGID() == pair.gidI) {
          p1 = &p;
          break;
        }
      }
    } else {
      auto it = ghost_map.find(pair.gidI);
      if (it != ghost_map.end()) {
        p1 = it->second;
      }
    }

    const Particle* p2 = nullptr;
    if (pair.is_localJ) {
      // Find local particle by GID
      for (const auto& p : local_particles) {
        if (p.getGID() == pair.gidJ) {
          p2 = &p;
          break;
        }
      }
    } else {
      auto it = ghost_map.find(pair.gidJ);
      if (it != ghost_map.end()) {
        p2 = it->second;
      }
    }

    if (!p1 || !p2) continue;

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
    bool p1_local, bool p2_local, double tolerance,
    const std::unordered_set<Constraint, ConstraintHash, ConstraintEqual>& existing_constraints,
    int constraint_iterations) {
  // For cross-rank constraints, ensure only one rank creates the constraint.
  // The rank owning the particle with the lower GID (p1) is responsible.
  // If p1 is not local to this rank, this rank should not create the constraint.
  if (!p1_local && p2_local) {
    return std::nullopt;
  }

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

  auto owned_by_me = (p1_local && p1.setGID() < p2.setGID()) || (!p1_local && p2_local);

  auto constraint = Constraint(
      -details.overlap,
      details.overlap > tolerance,
      p1.setGID(), p2.setGID(),
      local_id1, local_id2,
      owned_by_me,
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

std::vector<Particle> CollisionDetector::gatherAllParticles(const std::vector<Particle>& local_particles) {
  PetscMPIInt rank, size;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  MPI_Comm_size(PETSC_COMM_WORLD, &size);

  // Create custom MPI type for ParticleData
  MPI_Datatype particle_mpi_type;
  createParticleMPIType(&particle_mpi_type);

  // Gather particle counts from all ranks
  int local_count = static_cast<int>(local_particles.size());
  std::vector<int> counts(size);
  MPI_Allgather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, PETSC_COMM_WORLD);

  // Calculate displacements for MPI_Allgatherv
  std::vector<int> displacements(size);
  displacements[0] = 0;
  for (int i = 1; i < size; ++i) {
    displacements[i] = displacements[i - 1] + counts[i - 1];
  }
  int total_particles = displacements[size - 1] + counts[size - 1];

  // Convert local particles to ParticleData structs
  std::vector<ParticleData> local_particle_data;
  local_particle_data.reserve(local_count);
  for (const auto& p : local_particles) {
    local_particle_data.push_back(p.toStruct());
  }

  // Gather all particle data
  std::vector<ParticleData> all_particle_data(total_particles);
  MPI_Allgatherv(local_particle_data.data(), local_count, particle_mpi_type,
                 all_particle_data.data(), counts.data(), displacements.data(),
                 particle_mpi_type, PETSC_COMM_WORLD);

  // Convert ParticleData structs back to Particle objects
  std::vector<Particle> all_particles;
  all_particles.reserve(total_particles);
  for (const auto& pd : all_particle_data) {
    all_particles.emplace_back(pd);
  }

  // Free the custom MPI type
  MPI_Type_free(&particle_mpi_type);

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