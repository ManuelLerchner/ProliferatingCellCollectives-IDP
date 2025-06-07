#include "Physics.h"

#include <omp.h>

#include <iostream>
#include <memory>
#include <vector>

#include "Constraint.h"
#include "ParticleData.h"
#include "ParticleManager.h"
#include "petscmat.h"
#include "util/ArrayMath.h"
#include "util/PetscRaii.h"

std::unique_ptr<Mat> calculate_MobilityMatrix(Vec configuration, int number_of_particles) {
  std::unique_ptr<Mat> M = std::make_unique<Mat>();

  int dims = 6 * number_of_particles;

  MatCreate(PETSC_COMM_WORLD, M.get());
  MatSetType(*M, MATMPIAIJ);
  MatSetSizes(*M, PETSC_DECIDE, PETSC_DECIDE, dims, dims);
  MatMPIAIJSetPreallocation(*M, 36, NULL, 36, NULL);
  MatSetFromOptions(*M);

  PetscInt local_start, local_end;
  VecGetOwnershipRange(configuration, &local_start, &local_end);
  PetscInt local_particle_count = (local_end - local_start) / COMPONENTS_PER_PARTICLE;

  const double* configuration_array;
  VecGetArrayRead(configuration, &configuration_array);

  std::cout << "local_start: " << local_start << std::endl;
  std::cout << "local_end: " << local_end << std::endl;
  std::cout << "local_particle_count: " << local_particle_count << std::endl;

  for (int i = 0; i < local_particle_count; i++) {
    int global_particle_idx = local_start / COMPONENTS_PER_PARTICLE + i;

    auto length = get_length(configuration_array, i);

    double mLinear = 1.0 / length;
    double mAngular = 12.0 / (length * length * length);

    MatSetValue(*M, global_particle_idx * 6, global_particle_idx * 6, mLinear, INSERT_VALUES);
    MatSetValue(*M, global_particle_idx * 6 + 1, global_particle_idx * 6 + 1, mLinear, INSERT_VALUES);
    MatSetValue(*M, global_particle_idx * 6 + 2, global_particle_idx * 6 + 2, mLinear, INSERT_VALUES);
    MatSetValue(*M, global_particle_idx * 6 + 3, global_particle_idx * 6 + 3, mAngular, INSERT_VALUES);
    MatSetValue(*M, global_particle_idx * 6 + 4, global_particle_idx * 6 + 4, mAngular, INSERT_VALUES);
    MatSetValue(*M, global_particle_idx * 6 + 5, global_particle_idx * 6 + 5, mAngular, INSERT_VALUES);
  }

  VecRestoreArrayRead(configuration, &configuration_array);

  MatAssemblyBegin(*M, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(*M, MAT_FINAL_ASSEMBLY);

  return M;
}
