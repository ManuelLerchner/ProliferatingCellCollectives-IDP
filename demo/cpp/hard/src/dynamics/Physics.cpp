#include "Physics.h"

#include <omp.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>

#include "Constraint.h"
#include "petscmat.h"
#include "simulation/ParticleManager.h"
#include "util/ArrayMath.h"
#include "util/PetscRaii.h"

#define PREALLOCATION_NNZ 12

MatWrapper calculate_Jacobian(
    const std::vector<Constraint>& local_constraints,
    const std::vector<Particle>& local_particles) {
  using namespace utils::ArrayMath;

  // D is a (6 * global_num_bodies, global_num_constraints) matrix
  MatWrapper D = MatWrapper::CreateAIJ(6 * local_particles.size(), local_constraints.size(), PETSC_DETERMINE, PETSC_DETERMINE);

  MatMPIAIJSetPreallocation(D, PREALLOCATION_NNZ, NULL, PREALLOCATION_NNZ, NULL);
  MatSeqAIJSetPreallocation(D, PREALLOCATION_NNZ, NULL);
  MatSetOption(D, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);

  for (int c_local_idx = 0; c_local_idx < local_constraints.size(); ++c_local_idx) {
    const auto& constraint = local_constraints[c_local_idx];
    PetscInt global_col_idx = constraint.gid;

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

    // Check if body I is local to this process
    PetscInt local_ids_i[6] = {constraint.gidI * 6 + 0, constraint.gidI * 6 + 1, constraint.gidI * 6 + 2,
                               constraint.gidI * 6 + 3, constraint.gidI * 6 + 4, constraint.gidI * 6 + 5};
    PetscCallAbort(PETSC_COMM_WORLD, MatSetValues(D, 6, local_ids_i, 1, &global_col_idx, F_i, INSERT_VALUES));

    PetscInt local_ids_j[6] = {constraint.gidJ * 6 + 0, constraint.gidJ * 6 + 1, constraint.gidJ * 6 + 2,
                               constraint.gidJ * 6 + 3, constraint.gidJ * 6 + 4, constraint.gidJ * 6 + 5};
    PetscCallAbort(PETSC_COMM_WORLD, MatSetValues(D, 6, local_ids_j, 1, &global_col_idx, F_j, INSERT_VALUES));
  }

  MatAssemblyBegin(D, MAT_FINAL_ASSEMBLY);

  return D;
}

VecWrapper create_phi_vector(const std::vector<Constraint>& local_constraints) {
  // phi is a (global_num_constraints, 1) vector
  VecWrapper phi = VecWrapper::Create(local_constraints.size());

  // Set values using local indices (let PETSc handle the mapping)
  for (int i = 0; i < local_constraints.size(); ++i) {
    VecSetValue(phi, local_constraints[i].gid, local_constraints[i].delta0, INSERT_VALUES);  // Using local indexing
  }

  VecAssemblyBegin(phi);

  return phi;
}

MatWrapper calculate_MobilityMatrix(const std::vector<Particle>& local_particles, double xi) {
  PetscInt local_num_particles = local_particles.size();
  PetscInt local_dims = 6 * local_num_particles;

  // M is a (6 * global_num_particles, 6 * global_num_particles) matrix
  MatWrapper M = MatWrapper::CreateAIJ(local_dims, local_dims, PETSC_DETERMINE, PETSC_DETERMINE);

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

  // --- Phase 4: Assembly ---
  MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);

  return M;
}

MatWrapper calculate_QuaternionMap(const std::vector<Particle>& local_particles) {
  PetscInt local_num_particles = local_particles.size();
  PetscInt local_rows = 7 * local_num_particles;
  PetscInt local_cols = 6 * local_num_particles;

  // G is a (7 * global_num_particles, 6 * global_num_particles) matrix
  MatWrapper G = MatWrapper::CreateAIJ(local_rows, local_cols, PETSC_DETERMINE, PETSC_DETERMINE);

  std::vector<PetscInt> d_nnz(local_rows);
  std::vector<PetscInt> o_nnz(local_rows, 0);  // All blocks are local
  for (PetscInt i = 0; i < local_num_particles; ++i) {
    // Rows for the Identity block
    d_nnz[i * 7 + 0] = 1;
    d_nnz[i * 7 + 1] = 1;
    d_nnz[i * 7 + 2] = 1;
    // Rows for the Xi block
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

    // Xi block
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

MatWrapper calculate_stress_matrix(const std::vector<Constraint>& local_constraints, const std::vector<Particle>& local_particles) {
  using namespace utils::ArrayMath;

  PetscInt local_num_particles = local_particles.size();

  // S is a (num_particles, num_constraints) matrix
  MatWrapper S = MatWrapper::CreateAIJ(local_num_particles, local_constraints.size(), PETSC_DETERMINE, PETSC_DETERMINE);

  MatMPIAIJSetPreallocation(S, PREALLOCATION_NNZ, NULL, PREALLOCATION_NNZ, NULL);
  MatSeqAIJSetPreallocation(S, PREALLOCATION_NNZ, NULL);
  MatSetOption(S, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);

  for (PetscInt c_idx = 0; c_idx < local_constraints.size(); ++c_idx) {
    const auto& constraint = local_constraints[c_idx];

    PetscCallAbort(PETSC_COMM_WORLD, MatSetValue(S, constraint.gidI, constraint.gid, constraint.stressI, INSERT_VALUES));

    PetscCallAbort(PETSC_COMM_WORLD, MatSetValue(S, constraint.gidJ, constraint.gid, constraint.stressJ, INSERT_VALUES));
  }

  MatAssemblyBegin(S, MAT_FINAL_ASSEMBLY);

  return S;
}
