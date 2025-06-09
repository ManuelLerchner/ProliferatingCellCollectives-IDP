#include "MappingManager.h"

#include <petsc.h>

#include <algorithm>
#include <vector>

#include "Constraint.h"
#include "Particle.h"
#include "util/ArrayMath.h"
#include "util/PetscRaii.h"

ISLocalToGlobalMappingWrapper createLocalToGlobalMapping(const std::vector<Particle>& local_particles, int components_per_particle) {
  //   std::sort(local_particles.begin(), local_particles.end(),
  //             [](const Particle& a, const Particle& b) { return a.getId() < b.getId(); });

  // assumes particles are sorted by id

  PetscInt local_dims = components_per_particle * local_particles.size();

  std::vector<PetscInt> ownership_map(local_dims);
  for (PetscInt i = 0; i < local_particles.size(); ++i) {
    for (int j = 0; j < components_per_particle; ++j) {
      ownership_map[i * components_per_particle + j] = local_particles[i].getId() * components_per_particle + j;
    }
  }
  IS is_local_rows;
  ISCreateGeneral(PETSC_COMM_SELF, local_dims, ownership_map.data(), PETSC_COPY_VALUES, &is_local_rows);
  ISLocalToGlobalMappingWrapper ltog_map;
  ISLocalToGlobalMappingCreateIS(is_local_rows, ltog_map.get_ref());
  ISDestroy(&is_local_rows);

  return ltog_map;
}

ISLocalToGlobalMappingWrapper create_constraint_map(int local_num_constraints) {
  // This logic creates a contiguous global numbering for the constraints.
  PetscInt col_start_offset;
  MPI_Scan(&local_num_constraints, &col_start_offset, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD);
  col_start_offset -= local_num_constraints;

  // Build the ownership map using this contiguous numbering.
  std::vector<PetscInt> col_ownership_map(local_num_constraints);
  for (PetscInt i = 0; i < local_num_constraints; ++i) {
    col_ownership_map[i] = col_start_offset + i;
  }

  // Create the PETSc mapping object.
  IS is_local_cols;
  ISCreateGeneral(PETSC_COMM_SELF, local_num_constraints, col_ownership_map.data(), PETSC_COPY_VALUES, &is_local_cols);
  ISLocalToGlobalMappingWrapper ltog_map;
  ISLocalToGlobalMappingCreateIS(is_local_cols, ltog_map.get_ref());
  ISDestroy(&is_local_cols);

  return ltog_map;
}

MappingManager::Mappings MappingManager::createMappings(const std::vector<Particle>& local_particles, const std::vector<Constraint>& local_constraints) {
  auto col_map_6d = createLocalToGlobalMapping(local_particles, 6);
  auto row_map_7d = createLocalToGlobalMapping(local_particles, 7);
  auto constraint_map = create_constraint_map(local_constraints.size());

  return {.col_map_6d = std::move(col_map_6d), .row_map_7d = std::move(row_map_7d), .constraint_map = std::move(constraint_map)};
}