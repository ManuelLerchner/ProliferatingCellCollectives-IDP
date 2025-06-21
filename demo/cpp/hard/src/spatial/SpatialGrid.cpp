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

void SpatialGrid::insertParticle(int particle_idx, const std::array<double, 3>& position, double length, double diameter, bool is_local) {
  int cell_idx = getCellIndex(position);
  if (cell_idx >= 0 && cell_idx < grid_cells_.size()) {
    grid_cells_[cell_idx].push_back({particle_idx, is_local});
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

  for (const auto& p : local_particles) {
    insertParticle(p.getGID(), p.getPosition(), p.getLength(), p.getDiameter(), true);
  }

  for (const auto& p : ghost_particles) {
    insertParticle(p.getGID(), p.getPosition(), p.getLength(), p.getDiameter(), false);
  }

  for (int cell_idx = 0; cell_idx < grid_cells_.size(); ++cell_idx) {
    auto neighbor_indices = getNeighborCells(cell_idx);
    for (int neighbor_idx : neighbor_indices) {
      if (neighbor_idx < cell_idx) {
        continue;
      }

      const auto& cell1 = grid_cells_[cell_idx];
      const auto& cell2 = grid_cells_[neighbor_idx];

      for (size_t i = 0; i < cell1.size(); ++i) {
        for (size_t j = 0; j < cell2.size(); ++j) {
          if (cell_idx == neighbor_idx && i >= j) {
            continue;
          }

          const auto& p1_info = cell1[i];
          const auto& p2_info = cell2[j];

          // Don't check ghost-ghost collisions
          if (!p1_info.second && !p2_info.second) {
            continue;
          }

          pairs.push_back({p1_info.first, p2_info.first, p1_info.second, p2_info.second});
        }
      }
    }
  }

  return pairs;
}