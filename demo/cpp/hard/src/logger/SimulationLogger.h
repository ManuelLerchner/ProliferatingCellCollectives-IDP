#pragma once

#include "VTK.h"

namespace vtk {

struct SimulationStep {
  double simulation_time_s;
  double step_duration_s;
  double run_time_s;
  int num_particles;
  int num_constraints;
  int recursive_iterations;
  int bbpgd_iterations;
  double max_overlap;
  double residual;
  double dt_s;
};

class SimulationLogger {
 public:
  SimulationLogger(const std::string& outputDirectory, const std::string& baseFilename);
  void log(const SimulationStep& step);

 private:
  VTKDataLogger<SimulationStep> logger_;
};

}  // namespace vtk