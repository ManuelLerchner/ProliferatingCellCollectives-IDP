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

MatWrapper calculate_MobilityMatrix(std::vector<Particle>& local_particles, PetscInt global_num_particles, ISLocalToGlobalMappingWrapper& ltog_map) {
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
    double inv_len = 1.0 / particle.length;
    double inv_len3_x_12 = 12.0 / (particle.length * particle.length * particle.length);

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

  return M;
}