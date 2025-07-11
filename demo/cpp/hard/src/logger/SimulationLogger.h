#pragma once

#include "VTK.h"

namespace vtk {

struct SimulationStep {
  double simulation_time_s;
  double step_duration_s;
  size_t step;

  // Particle metrics
  int num_particles;
  size_t num_constraints;
  double colony_radius;

  // Solver metrics
  int recursive_iterations;
  long long bbpgd_iterations;
  double max_overlap;
  double residual;
  double dt_s;

  // New metrics
  double memory_usage_mb;  // Current memory usage in MB
  double peak_memory_mb;   // Peak memory usage in MB
  double cpu_time_s;       // CPU time in seconds
  double mpi_comm_time_s;  // Time spent in MPI communication in seconds
  double load_imbalance;   // Load imbalance ratio (max/avg particles per rank)
};

class SimulationLogger {
 public:
  SimulationLogger(const std::string& outputDirectory, const std::string& baseFilename, bool preserve_existing = false, size_t step = 0);
  void log(const SimulationStep& step);

 private:
  VTKDataLogger<SimulationStep> logger_;
};

}  // namespace vtk