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
  for (int i = 0; i < 3; ++i) {
    grid_dims_[i] = static_cast<int>(std::ceil((domain_max[i] - domain_min[i]) / cell_size));
  }

  int total_cells = grid_dims_[0] * grid_dims_[1] * grid_dims_[2];
  grid_cells_.resize(total_cells);
}

void SpatialGrid::clear() {
  for (auto& cell : grid_cells_) {
    cell.clear();
  }
}

void SpatialGrid::insertParticle(int particle_idx, const std::array<double, 3>& position, double length, double diameter) {
  int cell_idx = getCellIndex(position);
  if (cell_idx >= 0 && cell_idx < grid_cells_.size()) {
    grid_cells_[cell_idx].push_back(particle_idx);
  }
}

int SpatialGrid::getCellIndex(const std::array<double, 3>& position) const {
  auto coords = getCellCoords(position);

  // Check bounds
  for (int i = 0; i < 3; ++i) {
    if (coords[i] < 0 || coords[i] >= grid_dims_[i]) {
      return -1;  // Out of bounds
    }
  }

  return coords[0] + coords[1] * grid_dims_[0] + coords[2] * grid_dims_[0] * grid_dims_[1];
}

std::array<int, 3> SpatialGrid::getCellCoords(const std::array<double, 3>& position) const {
  std::array<int, 3> coords;
  for (int i = 0; i < 3; ++i) {
    coords[i] = static_cast<int>((position[i] - domain_min_[i]) / cell_size_);
  }
  return coords;
}

std::vector<int> SpatialGrid::getNeighborCells(int cell_idx) const {
  std::vector<int> neighbors;

  if (cell_idx < 0 || cell_idx >= grid_cells_.size()) {
    return neighbors;
  }

  // Convert linear index to 3D coordinates
  int z = cell_idx / (grid_dims_[0] * grid_dims_[1]);
  int y = (cell_idx % (grid_dims_[0] * grid_dims_[1])) / grid_dims_[0];
  int x = cell_idx % grid_dims_[0];

  // Check all 27 neighboring cells (including self)
  for (int dz = -1; dz <= 1; ++dz) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        int nx = x + dx;
        int ny = y + dy;
        int nz = z + dz;

        if (nx >= 0 && nx < grid_dims_[0] &&
            ny >= 0 && ny < grid_dims_[1] &&
            nz >= 0 && nz < grid_dims_[2]) {
          int neighbor_idx = nx + ny * grid_dims_[0] + nz * grid_dims_[0] * grid_dims_[1];
          neighbors.push_back(neighbor_idx);
        }
      }
    }
  }

  return neighbors;
}

std::vector<CollisionPair> SpatialGrid::findPotentialCollisions(const std::vector<Particle>& local_particles, const std::vector<Particle>& ghost_particles) {
  std::vector<CollisionPair> pairs;

  // Insert all particles into grid
  clear();

  for (int i = 0; i < local_particles.size(); ++i) {
    const auto& pos = local_particles[i].getPosition();
    insertParticle(i, pos, local_particles[i].getLength(), local_particles[i].getDiameter());
  }

  // Insert ghost particles with negative indices to distinguish them
  for (int i = 0; i < ghost_particles.size(); ++i) {
    const auto& pos = ghost_particles[i].getPosition();
    insertParticle(-(i + 1), pos, ghost_particles[i].getLength(), ghost_particles[i].getDiameter());
  }

  // Find potential collision pairs
  std::set<std::pair<int, int>> processed_pairs;

  for (int cell_idx = 0; cell_idx < grid_cells_.size(); ++cell_idx) {
    const auto& cell = grid_cells_[cell_idx];
    auto neighbor_cells = getNeighborCells(cell_idx);

    // Check within cell
    for (int i = 0; i < cell.size(); ++i) {
      for (int j = i + 1; j < cell.size(); ++j) {
        int idx1 = cell[i];
        int idx2 = cell[j];

        // Only consider pairs involving at least one local particle
        if (idx1 >= 0 || idx2 >= 0) {
          auto pair_key = std::make_pair(std::min(idx1, idx2), std::max(idx1, idx2));
          if (processed_pairs.find(pair_key) == processed_pairs.end()) {
            processed_pairs.insert(pair_key);

            CollisionPair pair;
            pair.local_idx_i = idx1 >= 0 ? idx1 : -1;
            pair.local_idx_j = idx2 >= 0 ? idx2 : -1;
            pair.global_id_i = idx1 >= 0 ? local_particles[idx1].getId() : ghost_particles[-(idx1 + 1)].getId();
            pair.global_id_j = idx2 >= 0 ? local_particles[idx2].getId() : ghost_particles[-(idx2 + 1)].getId();
            pair.is_cross_rank = (idx1 < 0) || (idx2 < 0);

            pairs.push_back(pair);
          }
        }
      }
    }

    // Check with neighbor cells
    for (int neighbor_idx : neighbor_cells) {
      if (neighbor_idx <= cell_idx) continue;  // Avoid duplicate checks

      const auto& neighbor_cell = grid_cells_[neighbor_idx];

      for (int p1_idx : cell) {
        for (int p2_idx : neighbor_cell) {
          // Only consider pairs involving at least one local particle
          if (p1_idx >= 0 || p2_idx >= 0) {
            auto pair_key = std::make_pair(std::min(p1_idx, p2_idx), std::max(p1_idx, p2_idx));
            if (processed_pairs.find(pair_key) == processed_pairs.end()) {
              processed_pairs.insert(pair_key);

              CollisionPair pair;
              pair.local_idx_i = p1_idx >= 0 ? p1_idx : -1;
              pair.local_idx_j = p2_idx >= 0 ? p2_idx : -1;
              pair.global_id_i = p1_idx >= 0 ? local_particles[p1_idx].getId() : ghost_particles[-(p1_idx + 1)].getId();
              pair.global_id_j = p2_idx >= 0 ? local_particles[p2_idx].getId() : ghost_particles[-(p2_idx + 1)].getId();
              pair.is_cross_rank = (p1_idx < 0) || (p2_idx < 0);

              pairs.push_back(pair);
            }
          }
        }
      }
    }
  }

  return pairs;
}