#include "Physics.h"

#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>

#include "Constraint.h"
#include "petscmat.h"
#include "petscsys.h"
#include "util/ArrayMath.h"
#include "util/PetscRaii.h"

void calculate_jacobian_local(
    MatWrapper& D,
    const std::vector<Constraint>& local_constraints,
    PetscInt offset) {
  using namespace utils::ArrayMath;

  for (int c_local_idx = 0; c_local_idx < local_constraints.size(); ++c_local_idx) {
    const auto& constraint = local_constraints[c_local_idx];

    const auto n_i = constraint.normI;
    const auto n_j = -n_i;
    const auto r_i = constraint.rPosI;
    const auto r_j = constraint.rPosJ;
    const auto torque_i = cross_product(r_i, n_i);
    const auto torque_j = cross_product(r_j, n_j);

    double F_i[6] = {n_i[0], n_i[1], n_i[2], torque_i[0], torque_i[1], torque_i[2]};
    double F_j[6] = {n_j[0], n_j[1], n_j[2], torque_j[0], torque_j[1], torque_j[2]};

    PetscInt c_global_idx = offset + c_local_idx;

    PetscInt rows_i[6] = {constraint.gidI * 6 + 0, constraint.gidI * 6 + 1, constraint.gidI * 6 + 2,
                          constraint.gidI * 6 + 3, constraint.gidI * 6 + 4, constraint.gidI * 6 + 5};
    PetscCallAbort(PETSC_COMM_WORLD, MatSetValues(D, 6, rows_i, 1, &c_global_idx, F_i, INSERT_VALUES));

    PetscInt rows_j[6] = {constraint.gidJ * 6 + 0, constraint.gidJ * 6 + 1, constraint.gidJ * 6 + 2,
                          constraint.gidJ * 6 + 3, constraint.gidJ * 6 + 4, constraint.gidJ * 6 + 5};
    PetscCallAbort(PETSC_COMM_WORLD, MatSetValues(D, 6, rows_j, 1, &c_global_idx, F_j, INSERT_VALUES));
  }
}

void create_phi_vector_local(VecWrapper& phi, const std::vector<Constraint>& local_constraints, PetscInt col_offset) {
  PetscInt local_num_constraints = local_constraints.size();
  if (local_num_constraints == 0) return;

  for (int i = 0; i < local_num_constraints; ++i) {
    PetscInt c_global_idx = col_offset + i;
    VecSetValue(phi, c_global_idx, local_constraints[i].delta0, INSERT_VALUES);
  }
}

void calculate_stress_matrix_local(MatWrapper& S, const std::vector<Constraint>& local_constraints, PetscInt offset) {
  using namespace utils::ArrayMath;

  PetscInt local_num_constraints = local_constraints.size();
  if (local_num_constraints == 0) return;

  for (PetscInt c_idx = 0; c_idx < local_constraints.size(); ++c_idx) {
    const auto& constraint = local_constraints[c_idx];
    PetscInt c_global_idx = offset + c_idx;
    PetscCallAbort(PETSC_COMM_WORLD, MatSetValue(S, constraint.gidI, c_global_idx, constraint.stressI, INSERT_VALUES));
    PetscCallAbort(PETSC_COMM_WORLD, MatSetValue(S, constraint.gidJ, c_global_idx, constraint.stressJ, INSERT_VALUES));
  }
}

MatWrapper calculate_MobilityMatrix(const std::vector<Particle>& local_particles, double xi) {
  PetscInt local_num_particles = local_particles.size();
  PetscInt local_dims = 6 * local_num_particles;

  MatWrapper M = MatWrapper::CreateAIJ(local_dims, local_dims);

  MatMPIAIJSetPreallocation(M, 1, NULL, 0, NULL);
  MatSeqAIJSetPreallocation(M, 1, NULL);

  for (PetscInt p_idx = 0; p_idx < local_num_particles; ++p_idx) {
    const auto& particle = local_particles[p_idx];
    double inv_len = 1.0 / (particle.getLength() * xi);
    double inv_len3_x_12 = 12.0 / (particle.getLength() * particle.getLength() * particle.getLength() * xi);

    for (int i = 0; i < 3; ++i) {
      PetscCallAbort(PETSC_COMM_WORLD, MatSetValue(M, particle.getGID() * 6 + i, particle.getGID() * 6 + i, inv_len, INSERT_VALUES));
    }
    for (int i = 3; i < 6; ++i) {
      PetscCallAbort(PETSC_COMM_WORLD, MatSetValue(M, particle.getGID() * 6 + i, particle.getGID() * 6 + i, inv_len3_x_12, INSERT_VALUES));
    }
  }

  MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);

  return M;
}

MatWrapper calculate_QuaternionMap(const std::vector<Particle>& local_particles) {
  PetscInt local_num_particles = local_particles.size();
  PetscInt local_rows = 7 * local_num_particles;
  PetscInt local_cols = 6 * local_num_particles;

  MatWrapper G = MatWrapper::CreateAIJ(local_rows, local_cols);

  std::vector<PetscInt> d_nnz(local_rows);
  std::vector<PetscInt> o_nnz(local_rows, 0);
  for (PetscInt i = 0; i < local_num_particles; ++i) {
    d_nnz[i * 7 + 0] = 1;
    d_nnz[i * 7 + 1] = 1;
    d_nnz[i * 7 + 2] = 1;
    d_nnz[i * 7 + 3] = 3;
    d_nnz[i * 7 + 4] = 3;
    d_nnz[i * 7 + 5] = 3;
    d_nnz[i * 7 + 6] = 3;
  }

  MatMPIAIJSetPreallocation(G, 0, d_nnz.data(), 0, o_nnz.data());
  MatSeqAIJSetPreallocation(G, 0, d_nnz.data());

  for (PetscInt p_idx = 0; p_idx < local_num_particles; ++p_idx) {
    const auto& particle = local_particles[p_idx];
    auto [s, wx, wy, wz] = particle.getQuaternion();

    for (int i = 0; i < 3; ++i) {
      PetscCallAbort(PETSC_COMM_WORLD, MatSetValue(G, particle.getGID() * 7 + i, particle.getGID() * 6 + i, 1, INSERT_VALUES));
    }

    PetscScalar xi_vals[4][3] = {
        {-0.5 * wx, -0.5 * wy, -0.5 * wz},
        {0.5 * s, -0.5 * wz, 0.5 * wy},
        {0.5 * wz, 0.5 * s, -0.5 * wx},
        {-0.5 * wy, 0.5 * wx, 0.5 * s}};

    PetscInt lrows[4] = {particle.getGID() * 7 + 3, particle.getGID() * 7 + 4, particle.getGID() * 7 + 5, particle.getGID() * 7 + 6};
    PetscInt lcols[3] = {particle.getGID() * 6 + 3, particle.getGID() * 6 + 4, particle.getGID() * 6 + 5};

    PetscCallAbort(PETSC_COMM_WORLD, MatSetValues(G, 4, lrows, 3, lcols, &xi_vals[0][0], INSERT_VALUES));
  }

  MatAssemblyBegin(G, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(G, MAT_FINAL_ASSEMBLY);

  return G;
}