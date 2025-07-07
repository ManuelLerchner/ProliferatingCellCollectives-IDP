#pragma once

#include "VTK.h"
#include "dynamics/Constraint.h"
#include "simulation/Particle.h"

namespace vtk {

class ParticleLogger {
 public:
  ParticleLogger(const std::string& outputDirectory, const std::string& baseFilename);
  void log(const std::vector<Particle>& particles);

 private:
  VTKDataLogger<std::vector<Particle>> logger_;
};

class ConstraintLogger {
 public:
  ConstraintLogger(const std::string& outputDirectory, const std::string& baseFilename);
  void log(const std::vector<Constraint>& constraints);

 private:
  VTKDataLogger<std::vector<Constraint>> logger_;
};

}  // namespace vtk