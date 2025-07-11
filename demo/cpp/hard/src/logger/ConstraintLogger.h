#pragma once

#include <set>

#include "VTK.h"
#include "dynamics/Constraint.h"
#include "simulation/Particle.h"

namespace vtk {

class ConstraintLogger {
 public:
  ConstraintLogger(const std::string& outputDirectory, const std::string& baseFilename, bool preserve_existing = false, size_t step = 0);
  void log(const std::vector<Constraint>& constraints);

 private:
  VTKDataLogger<std::vector<Constraint>> logger_;
};

}  // namespace vtk