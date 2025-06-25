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
  int gidI;
  int gidJ;
  bool is_localI;
  bool is_localJ;
};

class SpatialGrid {
 public:
  SpatialGrid(double cell_size, const std::array<double, 3>& domain_min, const std::array<double, 3>& domain_max);

  void clear();
  void insertParticle(int particle_idx, const std::array<double, 3>& position, double length, double diameter, bool is_local);
  std::vector<CollisionPair> findPotentialCollisions(const std::vector<Particle>& local_particles, const std::vector<Particle>& ghost_particles);

  double getCellSize() const;
  std::array<double, 3> getDomainMin() const;
  std::array<double, 3> getDomainMax() const;
  std::array<size_t, 3> getGridDims() const;

 private:
  double cell_size_;
  std::array<double, 3> domain_min_, domain_max_;
  std::array<size_t, 3> grid_dims_;
  std::vector<std::vector<std::pair<int, bool>>> grid_cells_;

  size_t getCellIndex(const std::array<double, 3>& position) const;
  std::array<size_t, 3> getCellCoords(const std::array<double, 3>& position) const;
  std::vector<size_t> getNeighborCells(size_t cell_idx) const;
};
