#pragma once

struct PhysicsConfig {
  double xi;
  double TAU;
  double l0;
  double LAMBDA;

  double getLambdaDimensionless() const {
    return (TAU / (xi * l0 * l0)) * LAMBDA;
  }
};

struct SolverConfig {
  double dt;
  double tolerance;
  int max_iterations;
};