#include "Forces.h"

#include <omp.h>

#include <iostream>
#include <memory>
#include <vector>

#include "Constraint.h"
#include "petscmat.h"
#include "util/ArrayMath.h"

std::unique_ptr<Mat> calculate_Jacobian(std::vector<Constraint> contacts, int num_bodies) {
  int num_constraints = contacts.size();
  int matrix_rows = 6 * num_bodies;
  int matrix_cols = num_constraints;

  std::unique_ptr<Mat> D = std::make_unique<Mat>();
  MatCreate(PETSC_COMM_WORLD, D.get());
  MatSetSizes(*D, PETSC_DECIDE, PETSC_DECIDE, matrix_rows, matrix_cols);
  MatSetType(*D, MATAIJ);
  MatSetFromOptions(*D);

  // --- Step 1: Correct Parallel Preallocation ---
  // For this specific problem, each row of the Jacobian will have at most `num_constraints` non-zeros.
  // Since we only have one constraint, we know each affected row has exactly one non-zero.
  // We can provide this as a hint. A dense preallocation is also an option for simplicity if the number of constraints is small.
  MatMPIAIJSetPreallocation(*D, 1, NULL, 1, NULL);
  MatSeqAIJSetPreallocation(*D, 1, NULL);

  // --- Step 2: Fill The Matrix (Parallel Safe) ---
  PetscInt rstart, rend;
  MatGetOwnershipRange(*D, &rstart, &rend);

  for (int c_idx = 0; c_idx < num_constraints; c_idx++) {
    const auto& constraint = contacts[c_idx];

    const std::array<double, 3> n_i = constraint.normI;
    const std::array<double, 3> r_i = constraint.rPosI;
    std::array<double, 3> torque_i = cross_product(r_i, n_i);
    std::array<double, 6> forces_i = {n_i[0], n_i[1], n_i[2], torque_i[0], torque_i[1], torque_i[2]};

    const std::array<double, 3> n_j = -n_i;
    const std::array<double, 3> r_j = constraint.rPosJ;
    std::array<double, 3> torque_j = cross_product(r_j, n_j);
    std::array<double, 6> forces_j = {n_j[0], n_j[1], n_j[2], torque_j[0], torque_j[1], torque_j[2]};

    // Each process sets values only for the rows it owns.
    for (int k = 0; k < 6; ++k) {
      PetscInt row_i = constraint.gidI * 6 + k;
      if (row_i >= rstart && row_i < rend) {
        MatSetValue(*D, row_i, c_idx, forces_i[k], INSERT_VALUES);
      }
      PetscInt row_j = constraint.gidJ * 6 + k;
      if (row_j >= rstart && row_j < rend) {
        MatSetValue(*D, row_j, c_idx, forces_j[k], INSERT_VALUES);
      }
    }
  }

  MatAssemblyBegin(*D, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(*D, MAT_FINAL_ASSEMBLY);

  return D;
}
