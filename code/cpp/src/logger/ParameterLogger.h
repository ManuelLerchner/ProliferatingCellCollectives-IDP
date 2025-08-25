#pragma once

#include "VTK.h"
#include "util/Config.h"

namespace vtk {

class ParameterLogger {
 public:
  ParameterLogger(const std::string& outputDirectory, const std::string& baseFilename, bool preserve_existing = false, size_t step = 0);
  void log(const SimulationParameters& step);

 private:
  VTKDataLogger<SimulationParameters> logger_;
};

}  // namespace vtk