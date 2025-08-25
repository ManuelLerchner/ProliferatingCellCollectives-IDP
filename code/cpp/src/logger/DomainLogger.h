#pragma once

#include <set>

#include "VTK.h"
#include "dynamics/Constraint.h"
#include "simulation/Particle.h"

namespace vtk {

class DomainLogger {
 public:
  DomainLogger(const std::string& outputDirectory, const std::string& baseFilename, bool preserve_existing = false, size_t step = 0);
  void log(const std::pair<std::array<double, 3>, std::array<double, 3>>& domain);

 private:
  VTKDataLogger<std::pair<std::array<double, 3>, std::array<double, 3>>> logger_;
};

}  // namespace vtk