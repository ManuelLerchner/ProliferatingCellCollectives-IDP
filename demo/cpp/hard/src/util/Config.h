#pragma once

#include <cmath>

struct Vec3d {
  double x, y, z;
};

struct SimulationConfig {
  double dt_s;
  double end_time;
  double log_frequency_seconds = 0;
  Vec3d min_box_size;

  // Adaptive timestepping parameters
  bool enable_adaptive_dt;
  int target_bbpgd_iterations;
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

  double getLambdaDimensionless() const {
    return (TAU / (xi * l0 * l0)) * LAMBDA;
  }
};

struct SolverConfig {
  double tolerance;
  double allowed_overlap;
  long long max_bbpgd_iterations;
  int max_recursive_iterations;
  double linked_cell_size;
  double growth_factor;
  double particle_preallocation_factor;

  int getMinPreallocationSize(int n) const {
    return std::max(1, static_cast<int>(n * particle_preallocation_factor));
  }
};