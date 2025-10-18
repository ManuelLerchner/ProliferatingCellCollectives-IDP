#pragma once

#include <mpi.h>
#include <petsc.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "simulation/Particle.h"

struct CollisionPair {
  int gidI;
  int gidJ;
  bool is_localI;
  bool is_localJ;
  int localIdxI;  // Index in local_particles or ghost_particles array
  int localIdxJ;  // Index in local_particles or ghost_particles array
};

// Hash function for 3D cell coordinates
struct CellHash {
  std::size_t operator()(const std::array<int, 3>& cell) const {
    // Simple hash combination
    std::size_t h1 = std::hash<int>{}(cell[0]);
    std::size_t h2 = std::hash<int>{}(cell[1]);
    std::size_t h3 = std::hash<int>{}(cell[2]);
    return h1 ^ (h2 << 1) ^ (h3 << 2);
  }
};

class SpatialGrid {
 public:
  SpatialGrid(double cell_size, const std::array<double, 3>& domain_min, const std::array<double, 3>& domain_max);

  void clear();
  void insertParticle(int particle_idx, const std::array<double, 3>& position, double length, double diameter, bool is_local, int local_idx);
  std::vector<CollisionPair> findPotentialCollisions(const std::vector<Particle>& local_particles, const std::vector<Particle>& ghost_particles);

  double getCellSize() const;
  std::array<double, 3> getDomainMin() const;
  std::array<double, 3> getDomainMax() const;

 private:
  double cell_size_;
  std::array<double, 3> domain_min_, domain_max_;

  struct ParticleInfo {
    int gid;
    bool is_local;
    int local_idx;
  };

  // Use hashmap instead of vector - maps cell coordinates to particles
  std::unordered_map<std::array<int, 3>, std::vector<ParticleInfo>, CellHash> grid_cells_;

  std::array<int, 3> getCellCoords(const std::array<double, 3>& position) const;
  std::vector<std::array<int, 3>> getNeighborCells(const std::array<int, 3>& cell_coords) const;
};