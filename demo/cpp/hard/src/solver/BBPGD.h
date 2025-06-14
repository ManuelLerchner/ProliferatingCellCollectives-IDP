#pragma once

#include <petsc.h>

#include <functional>

#include "util/Config.h"
#include "util/PetscRaii.h"

struct BBPGDResult {
  VecWrapper gamma;
  long long bbpgd_iterations;
  double residual;
};

BBPGDResult BBPGD(
    std::function<void(const VecWrapper& input, VecWrapper& output)>,
    std::function<double(const VecWrapper&, const VecWrapper&)> residual,
    const VecWrapper& gamma0,
    const SolverConfig& solver_config);