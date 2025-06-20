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
    insertParticle(local_particles[i].setGID(), pos, local_particles[i].getLength(), local_particles[i].getDiameter());
  }

  // Insert ghost particles with negative indices to distinguish them
  for (int i = 0; i < ghost_particles.size(); ++i) {
    const auto& pos = ghost_particles[i].getPosition();
    insertParticle(ghost_particles[i].setGID(), pos, ghost_particles[i].getLength(), ghost_particles[i].getDiameter());
  }

  for (int cell_idx = 0; cell_idx < grid_cells_.size(); ++cell_idx) {
    const auto& cell = grid_cells_[cell_idx];
    auto neighbor_cells = getNeighborCells(cell_idx);

    // Check within cell
    for (int i = 0; i < cell.size(); ++i) {
      for (int j = i + 1; j < cell.size(); ++j) {
        int gidI = cell[i];
        int gidJ = cell[j];

        CollisionPair pair = {gidI, gidJ, true, true};

        pairs.push_back(pair);
      }

      // Check with neighbor cells
      for (int neighbor_idx : neighbor_cells) {
        const auto& neighbor_cell = grid_cells_[neighbor_idx];

        for (int p2_idx : neighbor_cell) {
          int gidI = cell[i];
          int gidJ = p2_idx;

          CollisionPair pair = {gidI, gidJ, true, false};

          pairs.push_back(pair);
        }
      }
    }
  }

  return pairs;
}