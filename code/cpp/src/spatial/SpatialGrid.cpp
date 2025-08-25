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
    if (domain_max[i] < domain_min[i]) {
      grid_dims_[i] = 1;
    } else {
      grid_dims_[i] = static_cast<size_t>(std::ceil((domain_max[i] - domain_min[i]) / cell_size));
    }
  }

  size_t total_cells = grid_dims_[0] * grid_dims_[1] * grid_dims_[2];
  grid_cells_.resize(total_cells);
}

void SpatialGrid::clear() {
  for (auto& cell : grid_cells_) {
    cell.clear();
  }
}

void SpatialGrid::insertParticle(int particle_idx, const std::array<double, 3>& position, double length, double diameter, bool is_local, int local_idx) {
  size_t cell_idx = getCellIndex(position);
  if (cell_idx >= 0 && cell_idx < grid_cells_.size()) {
    grid_cells_[cell_idx].push_back({particle_idx, is_local, local_idx});
  }
}

size_t SpatialGrid::getCellIndex(const std::array<double, 3>& position) const {
  auto coords = getCellCoords(position);

  // Check bounds
  for (int i = 0; i < 3; ++i) {
    if (coords[i] >= grid_dims_[i]) {  // coords are unsigned, so only check upper bound
      throw std::runtime_error("Particle out of bounds");
    }
  }

  return coords[0] + coords[1] * grid_dims_[0] + coords[2] * grid_dims_[0] * grid_dims_[1];
}

std::array<size_t, 3> SpatialGrid::getCellCoords(const std::array<double, 3>& position) const {
  std::array<size_t, 3> coords;
  for (int i = 0; i < 3; ++i) {
    double pos = position[i] - domain_min_[i];
    if (pos < 0) {
      // This case should be handled by the bounds check in getCellIndex,
      // but as a safeguard, we can treat it as out of bounds.
      // Returning a large value will fail the bounds check.
      coords[i] = std::numeric_limits<size_t>::max();
    } else {
      coords[i] = static_cast<size_t>(pos / cell_size_);
    }
  }
  return coords;
}

std::vector<size_t> SpatialGrid::getNeighborCells(size_t cell_idx) const {
  std::vector<size_t> neighbors;

  if (cell_idx >= grid_cells_.size()) {
    return neighbors;
  }

  // Convert linear index to 3D coordinates
  size_t z = cell_idx / (grid_dims_[0] * grid_dims_[1]);
  size_t y = (cell_idx % (grid_dims_[0] * grid_dims_[1])) / grid_dims_[0];
  size_t x = cell_idx % grid_dims_[0];

  // Check all 27 neighboring cells (including self)
  for (int dz = -1; dz <= 1; ++dz) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        long long nx = static_cast<long long>(x) + dx;
        long long ny = static_cast<long long>(y) + dy;
        long long nz = static_cast<long long>(z) + dz;

        if (nx >= 0 && nx < grid_dims_[0] &&
            ny >= 0 && ny < grid_dims_[1] &&
            nz >= 0 && nz < grid_dims_[2]) {
          size_t neighbor_idx = nx + ny * grid_dims_[0] + nz * grid_dims_[0] * grid_dims_[1];
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

  for (size_t i = 0; i < local_particles.size(); i++) {
    const auto& p = local_particles[i];
    insertParticle(p.getGID(), p.getPosition(), p.getLength(), p.getDiameter(), true, i);
  }

  for (size_t i = 0; i < ghost_particles.size(); i++) {
    const auto& p = ghost_particles[i];
    insertParticle(p.getGID(), p.getPosition(), p.getLength(), p.getDiameter(), false, i);
  }

  for (size_t cell_idx = 0; cell_idx < grid_cells_.size(); ++cell_idx) {
    auto neighbor_indices = getNeighborCells(cell_idx);
    for (size_t neighbor_idx : neighbor_indices) {
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

std::array<size_t, 3> SpatialGrid::getGridDims() const {
  return grid_dims_;
}
