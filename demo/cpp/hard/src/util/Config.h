#pragma once

#include <cmath>

struct Vec3d {
  double x, y, z;
};

struct SimulationConfig {
  double dt;
  double end_time;
  double log_frequency_seconds = 0;
  Vec3d min_box_size;
  int domain_resize_frequency;

  // Adaptive timestepping parameters
  bool enable_adaptive_dt;
  int target_bbpgd_iterations;
  int dt_adjust_frequency;
  double dt_adjust_factor;  // 10% adjustment up or down
  double min_dt;
  double max_dt;
};

struct PhysicsConfig {
  double xi;   // Drag coefficient
  double TAU;  // Growth time constant
  double l0;   // Initial length
  double LAMBDA;
  double temperature;
  bool monolayer;
  Vec3d gravity;

  double getLambdaDimensionless() const {
    return (TAU / (xi * l0 * l0)) * LAMBDA;
  }
};

struct SolverConfig {
  double tolerance;
  long long max_bbpgd_iterations;
  int max_recursive_iterations;
  double linked_cell_size;
  int min_preallocation_size;
  double growth_factor;
  int max_constraints_per_pair;

  int getMinPreallocationSize(int n) const {
    int total_ranks;
    MPI_Comm_size(PETSC_COMM_WORLD, &total_ranks);
    return std::max(min_preallocation_size, n * 12) / total_ranks;
  }
};