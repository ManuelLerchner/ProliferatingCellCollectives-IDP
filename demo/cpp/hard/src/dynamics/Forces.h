
#pragma once

#include <petsc.h>

#include <memory>
#include <vector>

#include "Constraint.h"

std::unique_ptr<Mat> calculate_Jacobian(std::vector<Constraint> constraints, int num_bodies);