#include "Physics.h"

#include <omp.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>

#include "Constraint.h"
#include "ParticleData.h"
#include "ParticleManager.h"
#include "petscmat.h"
#include "util/ArrayMath.h"
#include "util/PetscRaii.h"

MatWrapper calculate_MobilityMatrix(std::vector<Particle>& local_particles, double xi, PetscInt global_num_particles, ISLocalToGlobalMappingWrapper& ltog_map) {
  PetscInt local_num_particles = local_particles.size();
  PetscInt dims = 6 * global_num_particles;
  PetscInt local_dims = 6 * local_num_particles;

  // M is a (6 * global_num_particles, 6 * global_num_particles) matrix
  MatWrapper M;
  MatCreate(PETSC_COMM_WORLD, M.get_ref());
  MatSetType(M, MATMPIAIJ);
  MatSetSizes(M, PETSC_DECIDE, PETSC_DECIDE, dims, dims);
  MatSetFromOptions(M);

  MatMPIAIJSetPreallocation(M, 1, NULL, 0, NULL);
  MatSeqAIJSetPreallocation(M, 1, NULL);

  MatSetLocalToGlobalMapping(M, ltog_map, ltog_map);

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

MatWrapper calculate_QuaternionMap(std::vector<Particle>& local_particles, PetscInt global_num_particles, ISLocalToGlobalMappingWrapper& row_map_7d, ISLocalToGlobalMappingWrapper& col_map_6d) {
  PetscInt local_num_particles = local_particles.size();
  PetscInt global_rows = 7 * global_num_particles;
  PetscInt global_cols = 6 * global_num_particles;
  PetscInt local_rows = 7 * local_num_particles;

  // G is a (7 * global_num_particles, 6 * global_num_particles) matrix
  MatWrapper G;
  MatCreate(PETSC_COMM_WORLD, G.get_ref());
  MatSetType(G, MATMPIAIJ);
  MatSetSizes(G, PETSC_DECIDE, PETSC_DECIDE, global_rows, global_cols);
  MatSetFromOptions(G);

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

  MatSetLocalToGlobalMapping(G, row_map_7d, col_map_6d);

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