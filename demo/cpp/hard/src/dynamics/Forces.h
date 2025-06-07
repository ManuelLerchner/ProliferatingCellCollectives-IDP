
#pragma once

#include <petsc.h>

#include <memory>
#include <vector>

#include "Constraint.h"
#include "util/PetscRaii.h"

MatWrapper calculate_Jacobian(const std::vector<Constraint>& local_contacts, PetscInt local_num_bodies, PetscInt global_num_bodies);