#pragma once

#include <petsc.h>

#include <functional>

#include "dynamics/Config.h"
#include "util/PetscRaii.h"

VecWrapper BBPGD(
    std::function<VecWrapper(const VecWrapper&)> gradient,
    std::function<double(const VecWrapper&, const VecWrapper&)> residual,
    const VecWrapper& gamma0,
    const SolverConfig& solver_config);