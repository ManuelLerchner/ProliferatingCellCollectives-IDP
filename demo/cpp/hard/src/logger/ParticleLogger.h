#pragma once

#include <set>

#include "VTK.h"
#include "dynamics/Constraint.h"
#include "simulation/Particle.h"

namespace vtk {

class ParticleLogger {
 public:
  ParticleLogger(const std::string& outputDirectory, const std::string& baseFilename, bool preserve_existing = false, size_t step = 0);
  void log(const std::vector<Particle>& particles);

 private:
  VTKDataLogger<std::vector<Particle>> logger_;
};

}  // namespace vtk