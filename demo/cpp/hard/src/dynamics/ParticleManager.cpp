#include "ParticleManager.h"

#include <petsc.h>
#include <solver/LCP.h>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <unordered_map>
#include <vector>

#include "Constraint.h"
#include "Forces.h"
#include "ParticleData.h"
#include "Physics.h"

ParticleManager::ParticleManager() {
}

void ParticleManager::queueNewParticle(Particle p) {
  new_particle_buffer.push_back(p);
}

void ParticleManager::commitNewParticles() {
  PetscInt num_to_add_local = new_particle_buffer.size();

  PetscInt first_id_for_this_rank;
  MPI_Scan(&num_to_add_local, &first_id_for_this_rank, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD);
  first_id_for_this_rank -= num_to_add_local;

  first_id_for_this_rank += this->global_particle_count;

  std::vector<PetscInt> new_ids(num_to_add_local);
  std::iota(new_ids.begin(), new_ids.end(), first_id_for_this_rank);
  for (PetscInt i = 0; i < num_to_add_local; ++i) {
    new_particle_buffer[i].id = new_ids[i];
  }

  local_particles.insert(
      local_particles.end(),
      std::make_move_iterator(new_particle_buffer.begin()),
      std::make_move_iterator(new_particle_buffer.end()));
  new_particle_buffer.clear();

  PetscInt total_added_this_step;
  MPI_Allreduce(&num_to_add_local, &total_added_this_step, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD);

  this->global_particle_count += total_added_this_step;
}

ISLocalToGlobalMappingWrapper ParticleManager::createLocalToGlobalMapping(int local_num_particles, int components_per_particle) {
  std::sort(local_particles.begin(), local_particles.end(),
            [](const Particle& a, const Particle& b) { return a.id < b.id; });

  PetscInt local_dims = components_per_particle * local_num_particles;

  std::vector<PetscInt> ownership_map(local_dims);
  for (PetscInt i = 0; i < local_num_particles; ++i) {
    for (int j = 0; j < components_per_particle; ++j) {
      ownership_map[i * components_per_particle + j] = local_particles[i].id * components_per_particle + j;
    }
  }
  IS is_local_rows;
  ISCreateGeneral(PETSC_COMM_SELF, local_dims, ownership_map.data(), PETSC_COPY_VALUES, &is_local_rows);
  ISLocalToGlobalMappingWrapper ltog_map;
  ISLocalToGlobalMappingCreateIS(is_local_rows, ltog_map.get_ref());
  ISDestroy(&is_local_rows);

  return ltog_map;
}

ISLocalToGlobalMappingWrapper ParticleManager::create_constraint_map(int local_num_constraints) {
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

void ParticleManager::timeStep() {
  std::vector<Constraint> local_constraints;

  for (int i = 0; i < local_particles.size(); i += 2) {
    if (i + 1 < local_particles.size()) {
      int id_i = local_particles[i].id;
      int id_j = local_particles[i + 1].id;
      local_constraints.emplace_back(Constraint(0.5, id_i, id_j, {1, 0, 0}, {1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}));
    }
  }

  ISLocalToGlobalMappingWrapper col_map_6d = createLocalToGlobalMapping(local_particles.size(), 6);
  ISLocalToGlobalMappingWrapper row_map_7d = createLocalToGlobalMapping(local_particles.size(), 7);

  ISLocalToGlobalMappingWrapper constraint_map = create_constraint_map(local_constraints.size());

  MatWrapper D = calculate_Jacobian(local_constraints, local_particles.size(), global_particle_count, row_map_7d, constraint_map);
  // PetscPrintf(PETSC_COMM_WORLD, "D Matrix:\n");
  // MatView(D.get(), PETSC_VIEWER_STDOUT_WORLD);

  MatWrapper M = calculate_MobilityMatrix(local_particles, global_particle_count, col_map_6d);
  // PetscPrintf(PETSC_COMM_WORLD, "Mobility Matrix M:\n");
  // MatView(M.get(), PETSC_VIEWER_STDOUT_WORLD);

  MatWrapper G = calculate_QuaternionMap(local_particles, global_particle_count, row_map_7d, col_map_6d);
  // PetscPrintf(PETSC_COMM_WORLD, "Quaternion Map G:\n");
  // MatView(G.get(), PETSC_VIEWER_STDOUT_WORLD);

  // create gamma matrix of shape (constraint_count, 1)
  VecWrapper gamma;
  VecCreate(PETSC_COMM_WORLD, gamma.get_ref());
  VecSetSizes(gamma, local_constraints.size(), PETSC_DETERMINE);
  VecSetFromOptions(gamma);

  VecSet(gamma, 1.0);

  // print gamma shape
  PetscInt gamma_rows;
  VecGetSize(gamma.get(), &gamma_rows);
  PetscPrintf(PETSC_COMM_WORLD, "gamma shape: %d\n", gamma_rows);

  // print D shape
  PetscInt rows, cols;
  MatGetSize(D.get(), &rows, &cols);
  PetscPrintf(PETSC_COMM_WORLD, "D shape: %d x %d\n", rows, cols);

  // cX = G @ M @ D^T @ gamma
  VecWrapper t1;
  MatCreateVecs(D.get(), NULL, t1.get_ref());
  MatMult(D, gamma.get(), t1.get());

  VecWrapper t2;
  MatCreateVecs(M.get(), NULL, t2.get_ref());
  MatMult(M, t1.get(), t2.get());

  VecWrapper t3;
  MatCreateVecs(G.get(), NULL, t3.get_ref());
  MatMult(G, t2.get(), t3.get());

  VecView(t3.get(), PETSC_VIEWER_STDOUT_WORLD);
}

void ParticleManager::run() {
  for (int i = 0; i < 1; i++) {
    commitNewParticles();

    timeStep();
  }
}
