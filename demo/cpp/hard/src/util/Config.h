#pragma once

struct SimulationConfig {
  double dt;
  double end_time;
  double log_frequency_seconds = 0;
};

struct PhysicsConfig {
  double xi;   // Drag coefficient
  double TAU;  // Growth time constant
  double l0;   // Initial length
  double LAMBDA;

  double getLambdaDimensionless() const {
    return (TAU / (xi * l0 * l0)) * LAMBDA;
  }
};

struct SolverConfig {
  double tolerance;
  long long max_bbpgd_iterations;
  int max_recursive_iterations;
};