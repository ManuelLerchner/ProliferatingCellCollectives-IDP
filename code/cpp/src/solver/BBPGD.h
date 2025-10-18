#pragma once

#include <math.h>
#include <petsc.h>

#include <functional>
#include <optional>

#include "logger/BBPGDLogger.h"
#include "util/Config.h"
#include "util/PetscRaii.h"

struct BBPGDResult {
  size_t bbpgd_iterations;
  double residual;
};

// Gradient interface
class Gradient {
 public:
  virtual VecWrapper& gradient(const VecWrapper& gamma_curr) = 0;
  virtual double residual(const VecWrapper& gradient_val, const VecWrapper& gamma) = 0;
  virtual std::tuple<double, double, double, double> energy(const VecWrapper& gamma) = 0;
};

BBPGDResult BBPGD(
    Gradient& gradient,
    VecWrapper& gamma,  // in-out parameter
    double allowed_residual,
    size_t max_bbpgd_iterations,
    std::optional<std::shared_ptr<vtk::BBPGDLogger>> bbpgd_logger);