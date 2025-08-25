#pragma once

#include <cmath>

struct Vec3d {
  double x, y, z;
};

struct SimulationConfig {
  double dt_s;
  double end_radius;
  double log_frequency_seconds = 0;
  Vec3d min_box_size;
};

struct PhysicsConfig {
  // General parameters
  double xi;           // Drag coefficient
  double TAU;          // Growth time constant
  double l0;           // Initial length
  double LAMBDA;       // Stress sensitivity
  double temperature;  // Brownian motion temperature

  // Soft potential
  double kcc;    // Collision constant
  double alpha;  // Overlap parameter

  PetscBool monolayer;

  double getLambdaDimensionless() const {
    return LAMBDA;
  }
};

struct SolverConfig {
  double tolerance;
  int max_bbpgd_iterations;
  int max_recursive_iterations;
  double linked_cell_size;
  double growth_factor;
  double particle_preallocation_factor;

  int getMinPreallocationSize(int n) const {
    return std::max(1, static_cast<int>(n * particle_preallocation_factor));
  }
};

struct SimulationParameters {
  SimulationConfig sim_config;
  PhysicsConfig physics_config;
  SolverConfig solver_config;
  std::string starter_vtk;
  std::string mode;
};
