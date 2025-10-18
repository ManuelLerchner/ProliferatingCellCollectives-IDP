#include "SpatialGrid.h"

#include <mpi.h>
#include <petsc.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <unordered_set>

#include "simulation/Particle.h"

// SpatialGrid Implementation
SpatialGrid::SpatialGrid(double cell_size, const std::array<double, 3>& domain_min, const std::array<double, 3>& domain_max)
    : cell_size_(cell_size), domain_min_(domain_min), domain_max_(domain_max) {
  // No need to pre-allocate cells - hashmap will grow as needed
}

void SpatialGrid::clear() {
  grid_cells_.clear();
}

void SpatialGrid::insertParticle(int particle_idx, const std::array<double, 3>& position, double length, double diameter, bool is_local, int local_idx) {
  auto cell_coords = getCellCoords(position);
  grid_cells_[cell_coords].push_back({particle_idx, is_local, local_idx});
}

std::array<int, 3> SpatialGrid::getCellCoords(const std::array<double, 3>& position) const {
  std::array<int, 3> coords;
  for (int i = 0; i < 3; ++i) {
    // Use floor to handle negative coordinates properly
    coords[i] = static_cast<int>(std::floor((position[i] - domain_min_[i]) / cell_size_));
  }
  return coords;
}

std::vector<std::array<int, 3>> SpatialGrid::getNeighborCells(const std::array<int, 3>& cell_coords) const {
  std::vector<std::array<int, 3>> neighbors;
  neighbors.reserve(27);  // 3x3x3 cube

  // Check all 27 neighboring cells (including self)
  for (int dz = -1; dz <= 1; ++dz) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        std::array<int, 3> neighbor = {
            cell_coords[0] + dx,
            cell_coords[1] + dy,
            cell_coords[2] + dz};
        neighbors.push_back(neighbor);
      }
    }
  }

  return neighbors;
}

std::vector<CollisionPair> SpatialGrid::findPotentialCollisions(const std::vector<Particle>& local_particles, const std::vector<Particle>& ghost_particles) {
  std::vector<CollisionPair> pairs;

  // Insert all particles into grid
  clear();

  for (size_t i = 0; i < local_particles.size(); i++) {
    const auto& p = local_particles[i];
    insertParticle(p.getGID(), p.getPosition(), p.getLength(), p.getDiameter(), true, i);
  }

  for (size_t i = 0; i < ghost_particles.size(); i++) {
    const auto& p = ghost_particles[i];
    insertParticle(p.getGID(), p.getPosition(), p.getLength(), p.getDiameter(), false, i);
  }

  // Track processed cell pairs to avoid duplicates
  std::set<std::pair<std::array<int, 3>, std::array<int, 3>>> processed_pairs;

  // Iterate through all occupied cells
  for (const auto& [cell_coords, cell_particles] : grid_cells_) {
    auto neighbor_cells = getNeighborCells(cell_coords);

    for (const auto& neighbor_coords : neighbor_cells) {
      // Skip if this cell pair was already processed
      auto cell_pair = std::minmax(cell_coords, neighbor_coords);
      if (!processed_pairs.insert(cell_pair).second) {
        continue;
      }

      // Check if neighbor cell exists in hashmap
      auto neighbor_it = grid_cells_.find(neighbor_coords);
      if (neighbor_it == grid_cells_.end()) {
        continue;  // No particles in this neighbor cell
      }

      const auto& cell1 = cell_particles;
      const auto& cell2 = neighbor_it->second;
      bool same_cell = (cell_coords == neighbor_coords);

      for (size_t i = 0; i < cell1.size(); ++i) {
        size_t j_start = same_cell ? (i + 1) : 0;
        for (size_t j = j_start; j < cell2.size(); ++j) {
          const auto& p1_info = cell1[i];
          const auto& p2_info = cell2[j];

          // Don't check ghost-ghost collisions
          if (!p1_info.is_local && !p2_info.is_local) {
            continue;
          }

          pairs.push_back({.gidI = p1_info.gid,
                           .gidJ = p2_info.gid,
                           .is_localI = p1_info.is_local,
                           .is_localJ = p2_info.is_local,
                           .localIdxI = p1_info.local_idx,
                           .localIdxJ = p2_info.local_idx});
        }
      }
    }
  }

  return pairs;
}

double SpatialGrid::getCellSize() const {
  return cell_size_;
}

std::array<double, 3> SpatialGrid::getDomainMin() const {
  return domain_min_;
}

std::array<double, 3> SpatialGrid::getDomainMax() const {
  return domain_max_;
}