#pragma once

#include <mpi.h>
#include <petsc.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <set>
#include <unordered_set>
#include <vector>

// Forward declarations
class Particle;

struct CollisionPair {
  int local_idx_i, local_idx_j;
  int global_id_i, global_id_j;
  bool is_cross_rank;
};

class SpatialGrid {
 public:
  SpatialGrid(double cell_size, const std::array<double, 3>& domain_min, const std::array<double, 3>& domain_max);

  void clear();
  void insertParticle(int particle_idx, const std::array<double, 3>& position, double length, double diameter);
  std::vector<CollisionPair> findPotentialCollisions(const std::vector<Particle>& local_particles, const std::vector<Particle>& ghost_particles);

 private:
  double cell_size_;
  std::array<double, 3> domain_min_, domain_max_;
  std::array<int, 3> grid_dims_;
  std::vector<std::vector<int>> grid_cells_;

  int getCellIndex(const std::array<double, 3>& position) const;
  std::array<int, 3> getCellCoords(const std::array<double, 3>& position) const;
  std::vector<int> getNeighborCells(int cell_idx) const;
};
