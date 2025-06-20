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

MatWrapper calculate_Jacobian(
    const std::vector<Constraint>& local_constraints,
    const std::vector<Particle>& local_particles,
    ISLocalToGlobalMapping velocityL2GMap,
    ISLocalToGlobalMapping constraintL2GMap_N) {
  using namespace utils::ArrayMath;

  PetscInt local_num_bodies = local_particles.size();
  PetscInt local_row_count = 6 * local_num_bodies;

  // D is a (6 * global_num_bodies, global_num_constraints) matrix
  MatWrapper D;
  MatCreate(PETSC_COMM_WORLD, D.get_ref());
  MatSetSizes(D, local_row_count, local_constraints.size(), PETSC_DETERMINE, PETSC_DETERMINE);
  MatSetType(D, MATAIJ);
  MatSetFromOptions(D);

  // Precise preallocation: count constraints per particle
  std::vector<PetscInt> d_nnz(local_row_count, 0);
  std::vector<PetscInt> o_nnz(local_row_count, 0);

  // Count how many constraints each local particle is involved in

  MatMPIAIJSetPreallocation(D, 0, NULL, 0, NULL);
  MatSeqAIJSetPreallocation(D, 0, NULL);

  MatSetLocalToGlobalMapping(D, velocityL2GMap, constraintL2GMap_N);
  MatSetOption(D, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);

  for (int c_local_idx = 0; c_local_idx < local_constraints.size(); ++c_local_idx) {
    const auto& constraint = local_constraints[c_local_idx];

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

    // Check if body I is owned by this process
    if (constraint.particleI_isLocal) {
      PetscInt local_ids_i[6] = {constraint.localI * 6 + 0, constraint.localI * 6 + 1, constraint.localI * 6 + 2,
                                 constraint.localI * 6 + 3, constraint.localI * 6 + 4, constraint.localI * 6 + 5};
      MatSetValuesLocal(D, 6, local_ids_i, 1, &c_local_idx, F_i, INSERT_VALUES);
    }

    // Check if body J is owned by this process

    if (constraint.particleJ_isLocal) {
      PetscInt local_ids_j[6] = {constraint.localJ * 6 + 0, constraint.localJ * 6 + 1, constraint.localJ * 6 + 2,
                                 constraint.localJ * 6 + 3, constraint.localJ * 6 + 4, constraint.localJ * 6 + 5};
      MatSetValuesLocal(D, 6, local_ids_j, 1, &c_local_idx, F_j, INSERT_VALUES);
    }
  }

  MatAssemblyBegin(D, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(D, MAT_FINAL_ASSEMBLY);

  return D;
}

VecWrapper create_phi_vector(const std::vector<Constraint>& local_constraints, ISLocalToGlobalMapping constraintL2GMap_N) {
  // phi is a (global_num_constraints, 1) vector
  VecWrapper phi;
  VecCreate(PETSC_COMM_WORLD, phi.get_ref());
  VecSetSizes(phi, local_constraints.size(), PETSC_DETERMINE);  // Fixed: local size first
  VecSetFromOptions(phi);

  // Set the local-to-global mapping (consistent with Jacobian)
  VecSetLocalToGlobalMapping(phi, constraintL2GMap_N);

  // Set values using local indices (let PETSc handle the mapping)
  for (int i = 0; i < local_constraints.size(); ++i) {
    VecSetValueLocal(phi, i, local_constraints[i].delta0, INSERT_VALUES);  // Using local indexing
  }

  VecAssemblyBegin(phi);
  VecAssemblyEnd(phi);

  return phi;
}

MatWrapper calculate_MobilityMatrix(const std::vector<Particle>& local_particles, PetscInt global_num_particles, double xi, ISLocalToGlobalMappingWrapper& velocityL2GMap) {
  PetscInt local_num_particles = local_particles.size();
  PetscInt local_dims = 6 * local_num_particles;

  // M is a (6 * global_num_particles, 6 * global_num_particles) matrix
  MatWrapper M;
  MatCreate(PETSC_COMM_WORLD, M.get_ref());
  MatSetType(M, MATMPIAIJ);
  MatSetSizes(M, local_dims, local_dims, PETSC_DETERMINE, PETSC_DETERMINE);

  MatMPIAIJSetPreallocation(M, 1, NULL, 0, NULL);
  MatSeqAIJSetPreallocation(M, 1, NULL);

  MatSetLocalToGlobalMapping(M, velocityL2GMap, velocityL2GMap);

  for (PetscInt p_idx = 0; p_idx < local_num_particles; ++p_idx) {
    const auto& particle = local_particles[p_idx];
    double inv_len = 1.0 / particle.getLength();
    double inv_len3_x_12 = 12.0 / (particle.getLength() * particle.getLength() * particle.getLength());

    // DEBUG
    // inv_len = particle.id;
    // inv_len3_x_12 = particle.id;

    for (int i = 0; i < 3; ++i) {
      MatSetValueLocal(M, p_idx * 6 + i, p_idx * 6 + i, inv_len, INSERT_VALUES);
    }
    for (int i = 3; i < 6; ++i) {
      MatSetValueLocal(M, p_idx * 6 + i, p_idx * 6 + i, inv_len3_x_12, INSERT_VALUES);
    }
  }

  // --- Phase 4: Assembly ---
  MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);

  // multiply by xi
  MatScale(M, 1 / xi);

  return M;
}

MatWrapper calculate_QuaternionMap(const std::vector<Particle>& local_particles, ISLocalToGlobalMappingWrapper& configL2GMap, ISLocalToGlobalMappingWrapper& velocityL2GMap) {
  PetscInt local_num_particles = local_particles.size();
  PetscInt local_rows = 7 * local_num_particles;
  PetscInt local_cols = 6 * local_num_particles;

  // G is a (7 * global_num_particles, 6 * global_num_particles) matrix
  MatWrapper G;
  MatCreate(PETSC_COMM_WORLD, G.get_ref());
  MatSetType(G, MATMPIAIJ);
  MatSetSizes(G, local_rows, local_cols, PETSC_DETERMINE, PETSC_DETERMINE);

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

  MatSetLocalToGlobalMapping(G, configL2GMap, velocityL2GMap);

  for (PetscInt p_idx = 0; p_idx < local_num_particles; ++p_idx) {
    const auto& particle = local_particles[p_idx];
    auto [s, wx, wy, wz] = particle.getQuaternion();

    // Identity block (here filled with 2 as we divide by 0.5 later)
    for (int i = 0; i < 3; ++i) {
      PetscInt local_row = p_idx * 7 + i;
      PetscInt local_col = p_idx * 6 + i;
      MatSetValueLocal(G, local_row, local_col, 2, INSERT_VALUES);
    }

    // Xi block
    PetscScalar xi_vals[4][3] = {
        {-wx, -wy, -wz},
        {s, -wz, wy},
        {wz, s, -wx},
        {-wy, wx, s}};

    PetscInt lrows[4] = {p_idx * 7 + 3, p_idx * 7 + 4, p_idx * 7 + 5, p_idx * 7 + 6};
    PetscInt lcols[3] = {p_idx * 6 + 3, p_idx * 6 + 4, p_idx * 6 + 5};

    MatSetValuesLocal(G, 4, lrows, 3, lcols, &xi_vals[0][0], INSERT_VALUES);
  }

  MatAssemblyBegin(G, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(G, MAT_FINAL_ASSEMBLY);

  // multiply by 0.5
  MatScale(G, 0.5);

  return G;
}

MatWrapper calculate_stress_matrix(const std::vector<Constraint>& local_constraints, const std::vector<Particle>& local_particles, ISLocalToGlobalMappingWrapper& length_map, ISLocalToGlobalMappingWrapper& constraintL2GMap) {
  using namespace utils::ArrayMath;

  PetscInt local_num_particles = local_particles.size();
  PetscInt local_num_constraints = local_constraints.size();

  // S is a (num_particles, num_constraints) matrix
  MatWrapper S;
  MatCreate(PETSC_COMM_WORLD, S.get_ref());
  MatSetType(S, MATMPIAIJ);
  MatSetSizes(S, local_num_particles, local_num_constraints, PETSC_DETERMINE, PETSC_DETERMINE);

  std::vector<PetscInt> d_nnz(local_num_particles);
  std::vector<PetscInt> o_nnz(local_num_particles, 0);
  for (PetscInt i = 0; i < local_num_particles; ++i) {
    d_nnz[i] = 1;
  }
  MatMPIAIJSetPreallocation(S, 0, NULL, 0, NULL);
  MatSeqAIJSetPreallocation(S, 0, NULL);
  MatSetOption(S, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);

  MatSetLocalToGlobalMapping(S, length_map, constraintL2GMap);

  for (PetscInt c_idx = 0; c_idx < local_num_constraints; ++c_idx) {
    const auto& constraint = local_constraints[c_idx];

    if (constraint.particleI_isLocal) {
      MatSetValueLocal(S, constraint.localI, c_idx, constraint.stressI, INSERT_VALUES);
    }
    if (constraint.particleJ_isLocal) {
      MatSetValueLocal(S, constraint.localJ, c_idx, constraint.stressJ, INSERT_VALUES);
    }
  }

  MatAssemblyBegin(S, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(S, MAT_FINAL_ASSEMBLY);

  return S;
}
