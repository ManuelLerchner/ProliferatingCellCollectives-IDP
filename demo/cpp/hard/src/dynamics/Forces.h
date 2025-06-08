
#pragma once

#include <petsc.h>

#include <memory>
#include <vector>

#include "Constraint.h"
#include "util/PetscRaii.h"

MatWrapper calculate_Jacobian(
    const std::vector<Constraint>& local_contacts,
    PetscInt local_num_bodies,
    PetscInt global_num_bodies,
    ISLocalToGlobalMapping body_dof_map_6N,
    ISLocalToGlobalMapping constraint_map_N);

VecWrapper create_phi_vector(const std::vector<Constraint>& local_contacts,
                             ISLocalToGlobalMapping constraint_map_N);