#pragma once

#include <petsc.h>

#include <functional>

#include "util/PetscRaii.h"

VecWrapper BBPGD(
    std::function<VecWrapper(const VecWrapper&)> gradient,
    std::function<double(const VecWrapper&, const VecWrapper&)> residual,
    const VecWrapper& gamma0,
    double eps,
    int max_iterations);