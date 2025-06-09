#include "CollisionDetector.h"

#include <mpi.h>
#include <petsc.h>

#include <algorithm>
#include <cmath>
#include <limits>
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
  auto dir = getDirectionVector(p.getQuaternion());
  double diameter = 0.5;
  double length = p.getLength();

  // Calculate endpoints of the rod
  std::array<double, 3> start_point = pos - (length / 2 - diameter / 2) * dir;
  std::array<double, 3> end_point = pos + (length / 2 - diameter / 2) * dir;

  return {start_point, end_point};
}

bool CollisionDetector::checkSpherocylinderCollision(
    const Particle& p1, const Particle& p2,
    double& overlap, std::array<double, 3>& contact_point,
    std::array<double, 3>& normal) {
  // Get the line segments (cylindrical cores) of both particles
  auto [p1_start, p1_end] = getParticleEndpoints(p1);
  auto [p2_start, p2_end] = getParticleEndpoints(p2);

  // Use DCPQuery for accurate segment-segment distance calculation
  DCPQuery distance_query;
  std::array<double, 3> closest_p1, closest_p2;
  double min_distance = distance_query(p1_start, p1_end, p2_start, p2_end, closest_p1, closest_p2);

  // Check for collision
  double sum_radii = (p1.getDiameter() + p2.getDiameter()) / 2.0;
  overlap = sum_radii - min_distance;

  if (overlap > collision_tolerance_) {
    // Collision detected

    // Calculate contact point (midpoint between closest points)
    contact_point = {
        (closest_p1[0] + closest_p2[0]) / 2.0,
        (closest_p1[1] + closest_p2[1]) / 2.0,
        (closest_p1[2] + closest_p2[2]) / 2.0};

    // Calculate normal vector (from p2 to p1)
    double dx = closest_p1[0] - closest_p2[0];
    double dy = closest_p1[1] - closest_p2[1];
    double dz = closest_p1[2] - closest_p2[2];

    double norm_length = std::sqrt(dx * dx + dy * dy + dz * dz);
    if (norm_length > std::numeric_limits<double>::epsilon()) {
      normal = {dx / norm_length, dy / norm_length, dz / norm_length};
    } else {
      // Particles are exactly on top of each other, use arbitrary normal
      normal = {1.0, 0.0, 0.0};
    }

    return true;
  }

  return false;
}

std::vector<Constraint> CollisionDetector::detectCollisions(
    const std::vector<Particle>& local_particles,
    const std::vector<Particle>& ghost_particles) {
  std::vector<Constraint> constraints;

  // Update spatial grid bounds based on particle positions
  if (!local_particles.empty()) {
    std::array<double, 3> min_pos = local_particles[0].getPosition();
    std::array<double, 3> max_pos = min_pos;

    for (const auto& p : local_particles) {
      const auto& pos = p.getPosition();
      for (int i = 0; i < 3; ++i) {
        min_pos[i] = std::min(min_pos[i], pos[i] - p.getLength());
        max_pos[i] = std::max(max_pos[i], pos[i] + p.getLength());
      }
    }

    for (const auto& p : ghost_particles) {
      const auto& pos = p.getPosition();
      for (int i = 0; i < 3; ++i) {
        min_pos[i] = std::min(min_pos[i], pos[i] - p.getLength());
        max_pos[i] = std::max(max_pos[i], pos[i] + p.getLength());
      }
    }

    // Recreate spatial grid with updated bounds
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

  // Check collisions between local particles
  for (int i = 0; i < local_particles.size(); ++i) {
    for (int j = i + 1; j < local_particles.size(); ++j) {
      const Particle& p1 = local_particles[i];
      const Particle& p2 = local_particles[j];

      double overlap;
      std::array<double, 3> contact_point, normal;

      if (checkSpherocylinderCollision(p1, p2, overlap, contact_point, normal)) {
        // Calculate relative positions on particles
        const auto& pos1 = p1.getPosition();
        const auto& pos2 = p2.getPosition();

        std::array<double, 3> rPos1 = {
            contact_point[0] - pos1[0],
            contact_point[1] - pos1[1],
            contact_point[2] - pos1[2]};

        std::array<double, 3> rPos2 = {
            contact_point[0] - pos2[0],
            contact_point[1] - pos2[1],
            contact_point[2] - pos2[2]};

        // Create constraint
        constraints.emplace_back(Constraint(
            -overlap,       // negative overlap indicates penetration
            p1.getId(),     // particle I ID
            p2.getId(),     // particle J ID
            normal,         // surface normal from I to J
            rPos1,          // relative position on particle I
            rPos2,          // relative position on particle J
            contact_point,  // lab frame location on particle I
            contact_point   // lab frame location on particle J
            ));
      }
    }
  }

  // Find potential collision pairs using spatial grid for local-ghost interactions
  auto collision_pairs = spatial_grid_.findPotentialCollisions(local_particles, ghost_particles);

  // Check each potential pair for actual collision (local-ghost interactions ONLY)
  for (const auto& pair : collision_pairs) {
    const Particle* p1 = nullptr;
    const Particle* p2 = nullptr;
    bool is_local_local_pair = false;

    // Get particle references and check if this is a local-local pair (already handled above)
    if (pair.local_idx_i >= 0) {
      p1 = &local_particles[pair.local_idx_i];
    } else {
      // Find ghost particle by ID
      for (const auto& ghost : ghost_particles) {
        if (ghost.getId() == pair.global_id_i) {
          p1 = &ghost;
          break;
        }
      }
    }

    if (pair.local_idx_j >= 0) {
      p2 = &local_particles[pair.local_idx_j];
      // If both particles are local, this is a local-local pair that was already handled
      if (pair.local_idx_i >= 0) {
        is_local_local_pair = true;
      }
    } else {
      // Find ghost particle by ID
      for (const auto& ghost : ghost_particles) {
        if (ghost.getId() == pair.global_id_j) {
          p2 = &ghost;
          break;
        }
      }
    }

    // Skip local-local pairs as they were already handled in the first loop
    if (is_local_local_pair) {
      continue;
    }

    if (p1 && p2) {
      double overlap;
      std::array<double, 3> contact_point, normal;

      if (checkSpherocylinderCollision(*p1, *p2, overlap, contact_point, normal)) {
        // Calculate relative positions on particles
        const auto& pos1 = p1->getPosition();
        const auto& pos2 = p2->getPosition();

        std::array<double, 3> rPos1 = {
            contact_point[0] - pos1[0],
            contact_point[1] - pos1[1],
            contact_point[2] - pos1[2]};

        std::array<double, 3> rPos2 = {
            contact_point[0] - pos2[0],
            contact_point[1] - pos2[1],
            contact_point[2] - pos2[2]};

        // Create constraint
        constraints.emplace_back(Constraint(
            -overlap,          // negative overlap indicates penetration
            pair.global_id_i,  // particle I ID
            pair.global_id_j,  // particle J ID
            normal,            // surface normal from I to J
            rPos1,             // relative position on particle I
            rPos2,             // relative position on particle J
            contact_point,     // lab frame location on particle I
            contact_point      // lab frame location on particle J
            ));
      }
    }
  }

  return constraints;
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

    local_data.push_back(static_cast<double>(p.getId()));
    local_data.insert(local_data.end(), pos.begin(), pos.end());
    local_data.insert(local_data.end(), quat.begin(), quat.end());
    local_data.push_back(p.getLength());
    local_data.push_back(p.getDiameter());
  }

  // Gather all particle data
  int data_per_particle = 10;  // id + 3 pos + 4 quat + length + diameter
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

    all_particles.emplace_back(id, pos, quat, length, diameter);
  }

  return all_particles;
}

std::vector<Particle> CollisionDetector::filterGhostParticles(
    const std::vector<Particle>& all_particles,
    const std::vector<Particle>& local_particles,
    double cutoff_distance) {
  std::vector<Particle> ghost_particles;

  // Create set of local particle IDs for quick lookup
  std::unordered_set<PetscInt> local_ids;
  for (const auto& p : local_particles) {
    local_ids.insert(p.getId());
  }

  // Find particles within cutoff distance of local particles
  for (const auto& candidate : all_particles) {
    // Skip if this is a local particle
    if (local_ids.find(candidate.getId()) != local_ids.end()) {
      continue;
    }

    // Check distance to any local particle
    bool within_cutoff = false;
    const auto& cand_pos = candidate.getPosition();

    for (const auto& local_p : local_particles) {
      const auto& local_pos = local_p.getPosition();

      double dx = cand_pos[0] - local_pos[0];
      double dy = cand_pos[1] - local_pos[1];
      double dz = cand_pos[2] - local_pos[2];
      double distance = std::sqrt(dx * dx + dy * dy + dz * dz);

      // Include some margin for particle sizes
      double margin = (candidate.getLength() + local_p.getLength()) / 2.0;

      if (distance < cutoff_distance + margin) {
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