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
  virtual void gradient(const VecWrapper& gamma_curr, VecWrapper& phi_next_out) = 0;
  virtual double residual(const VecWrapper& gradient_val, const VecWrapper& gamma) = 0;
};

BBPGDResult BBPGD(
    Gradient& gradient,
    VecWrapper& gamma,  // in-out parameter
    double allowed_residual,
    size_t max_bbpgd_iterations,
    size_t iter);