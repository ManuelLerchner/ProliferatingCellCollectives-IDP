#pragma once

#include <vector>

#include "VTK.h"

namespace vtk {

struct BBPGDStep {
  size_t step;
  double residual;
  double step_size;
  double linear;
  double quadratic;
  double growth;
  double total;
  size_t recursive_iteration;
  size_t total_constraints;
};

class BBPGDLogger {
 public:
  BBPGDLogger(const std::string& outputDirectory,
              const std::string& baseFilename,
              size_t step = 0)
      : logger_(outputDirectory, baseFilename, true, false, step) {}

  void collect(BBPGDStep step);

  void log();

  void set_recursive_iteration(size_t iteration) {
    recursive_iteration = iteration;
  }

 private:
  VTKDataLogger<BBPGDStep> logger_;
  std::vector<BBPGDStep> steps_;
  size_t recursive_iteration = 0;
};

}  // namespace vtk
