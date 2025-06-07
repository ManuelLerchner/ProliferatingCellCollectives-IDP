#include "Forces.h"

#include "Particle.h"
#include "util/ArrayMath.h"

MatWrapper calculate_Jacobian(
    const std::vector<Constraint>& local_contacts,
    PetscInt local_num_bodies,
    PetscInt global_num_bodies,
    ISLocalToGlobalMapping body_dof_map_6N,
    ISLocalToGlobalMapping constraint_map_N) {
  PetscInt local_num_constraints = local_contacts.size();
  PetscInt global_num_constraints;
  MPI_Allreduce(&local_num_constraints, &global_num_constraints, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD);

  PetscInt local_row_count = 6 * local_num_bodies;

  PetscInt matrix_rows = 6 * global_num_bodies;
  PetscInt matrix_cols = global_num_constraints;

  // D is a (6 * global_num_bodies, global_num_constraints) matrix
  MatWrapper D;
  MatCreate(PETSC_COMM_WORLD, D.get_ref());
  MatSetSizes(D, local_row_count, 1, matrix_rows, matrix_cols);
  MatSetType(D, MATAIJ);
  MatSetFromOptions(D);

  MatMPIAIJSetPreallocation(D, 1, NULL, 0, NULL);
  MatSeqAIJSetPreallocation(D, 1, NULL);

  MatSetLocalToGlobalMapping(D, body_dof_map_6N, constraint_map_N);

  PetscInt col_start_offset;
  MPI_Scan(&local_num_constraints, &col_start_offset, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD);
  col_start_offset -= local_num_constraints;

  PetscInt rstart, rend;
  MatGetOwnershipRange(D, &rstart, &rend);

  for (int c_local_idx = 0; c_local_idx < local_contacts.size(); ++c_local_idx) {
    PetscInt c_global_idx = col_start_offset + c_local_idx;
    const auto& constraint = local_contacts[c_local_idx];

    // normal vectors
    const auto n_i = constraint.normI;
    const auto n_j = -n_i;

    // contact points
    const auto r_i = constraint.rPosI;
    const auto r_j = constraint.rPosJ;

    // torques
    const auto torque_i = cross_product(r_i, n_i);
    const auto torque_j = cross_product(r_j, n_j);

    // force vectors (6, 1)
    double F_i[6] = {n_i[0], n_i[1], n_i[2], torque_i[0], torque_i[1], torque_i[2]};
    double F_j[6] = {n_j[0], n_j[1], n_j[2], torque_j[0], torque_j[1], torque_j[2]};

    // Indices in the global matrix
    int gi_global = constraint.gidI * 6;
    int gj_global = constraint.gidJ * 6;
    PetscInt ids_i[6] = {gi_global + 0, gi_global + 1, gi_global + 2, gi_global + 3, gi_global + 4, gi_global + 5};
    PetscInt ids_j[6] = {gj_global + 0, gj_global + 1, gj_global + 2, gj_global + 3, gj_global + 4, gj_global + 5};

    // DEBUG
    //  double F_i[6] = {gi_global, gi_global, gi_global, gi_global, gi_global, gi_global};
    //  double F_j[6] = {gj_global, gj_global, gj_global, gj_global, gj_global, gj_global};

    if (gi_global >= rstart && gi_global < rend) {
      MatSetValues(D, 6, ids_i, 1, &c_global_idx, F_i, INSERT_VALUES);
    }

    if (gj_global >= rstart && gj_global < rend) {
      MatSetValues(D, 6, ids_j, 1, &c_global_idx, F_j, INSERT_VALUES);
    }
  }

  MatAssemblyBegin(D, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(D, MAT_FINAL_ASSEMBLY);

  return D;
}