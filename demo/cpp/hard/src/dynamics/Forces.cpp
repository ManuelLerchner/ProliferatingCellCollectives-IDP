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
  MatSetSizes(D, local_row_count, local_num_constraints, matrix_rows, matrix_cols);
  MatSetType(D, MATAIJ);
  MatSetFromOptions(D);

  MatMPIAIJSetPreallocation(D, 1, NULL, 0, NULL);
  MatSeqAIJSetPreallocation(D, 1, NULL);

  MatSetLocalToGlobalMapping(D, body_dof_map_6N, constraint_map_N);

  for (int c_local_idx = 0; c_local_idx < local_contacts.size(); ++c_local_idx) {
    const auto& constraint = local_contacts[c_local_idx];

    // Normal vectors
    const auto n_i = constraint.normI;
    const auto n_j = -n_i;

    // Contact points
    const auto r_i = constraint.rPosI;
    const auto r_j = constraint.rPosJ;

    // Torques
    const auto torque_i = cross_product(r_i, n_i);
    const auto torque_j = cross_product(r_j, n_j);

    // Force vectors (6, 1)
    double F_i[6] = {n_i[0], n_i[1], n_i[2], torque_i[0], torque_i[1], torque_i[2]};
    double F_j[6] = {n_j[0], n_j[1], n_j[2], torque_j[0], torque_j[1], torque_j[2]};

    // Local body indices (assuming bodies are locally indexed 0, 1, 2, ...)
    // You'll need to convert global body IDs to local indices
    int gi_local = constraint.gidI;  // This should be the local index of body I
    int gj_local = constraint.gidJ;  // This should be the local index of body J

    PetscInt local_ids_i[6] = {gi_local * 6 + 0, gi_local * 6 + 1, gi_local * 6 + 2,
                               gi_local * 6 + 3, gi_local * 6 + 4, gi_local * 6 + 5};
    PetscInt local_ids_j[6] = {gj_local * 6 + 0, gj_local * 6 + 1, gj_local * 6 + 2,
                               gj_local * 6 + 3, gj_local * 6 + 4, gj_local * 6 + 5};

    // Use local constraint index
    PetscInt c_local = c_local_idx;

    // Set values using local indices - PETSc will handle global mapping
    // Check if body I is owned by this process
    if (gi_local >= 0 && gi_local < local_num_bodies) {
      MatSetValuesLocal(D, 6, local_ids_i, 1, &c_local, F_i, INSERT_VALUES);
    }

    // Check if body J is owned by this process
    if (gj_local >= 0 && gj_local < local_num_bodies) {
      MatSetValuesLocal(D, 6, local_ids_j, 1, &c_local, F_j, INSERT_VALUES);
    }
  }

  MatAssemblyBegin(D, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(D, MAT_FINAL_ASSEMBLY);

  return D;
}

VecWrapper create_phi_vector(const std::vector<Constraint>& local_contacts,
                             ISLocalToGlobalMapping constraint_map_N) {
  PetscInt local_num_constraints = local_contacts.size();
  PetscInt global_num_constraints;
  MPI_Allreduce(&local_num_constraints, &global_num_constraints, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD);

  // phi is a (global_num_constraints, 1) vector
  VecWrapper phi;
  VecCreate(PETSC_COMM_WORLD, phi.get_ref());
  VecSetSizes(phi, local_num_constraints, global_num_constraints);  // Fixed: local size first
  VecSetFromOptions(phi);

  // Set the local-to-global mapping (consistent with Jacobian)
  VecSetLocalToGlobalMapping(phi, constraint_map_N);

  // Set values using local indices (let PETSc handle the mapping)
  for (int i = 0; i < local_num_constraints; ++i) {
    VecSetValueLocal(phi, i, local_contacts[i].delta0, INSERT_VALUES);  // Using local indexing
  }

  VecAssemblyBegin(phi);
  VecAssemblyEnd(phi);

  return phi;
}