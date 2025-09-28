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
};

class BBPGDLogger {
 public:
  BBPGDLogger(const std::string& outputDirectory,
              const std::string& baseFilename,
              size_t step = 0)
      : logger_(outputDirectory, baseFilename, true, false, step) {}

  void collect(const BBPGDStep& step);

  void log();

 private:
  VTKDataLogger<BBPGDStep> logger_;
  std::vector<BBPGDStep> steps_;
};

}  // namespace vtk
