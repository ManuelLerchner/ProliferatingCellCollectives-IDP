#include "ParticleManager.h"

#include <petsc.h>
#include <solver/LCP.h>

#include <iostream>
#include <numeric>
#include <unordered_map>
#include <vector>

#include "Constraint.h"
#include "Forces.h"
#include "ParticleData.h"
#include "Physics.h"
#include "util/PetscRaii.h"

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

void ParticleManager::timeStep() {
  std::vector<Constraint> local_constraints;

  for (int i = 0; i < local_particles.size(); i += 2) {
    if (i + 1 < local_particles.size()) {
      int id_i = local_particles[i].id;
      int id_j = local_particles[i + 1].id;
      local_constraints.emplace_back(Constraint(0.5, id_i, id_j, {1, 0, 0}, {1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}));
    }
  }

  MatWrapper D = calculate_Jacobian(local_constraints, local_particles.size(), global_particle_count);

  // std::unique_ptr<Mat> M = calculate_MobilityMatrix(configuration, global_particle_count_);

  PetscPrintf(PETSC_COMM_WORLD, "D Matrix:\n");
  MatView(D.get(), PETSC_VIEWER_STDOUT_WORLD);
  // PetscPrintf(PETSC_COMM_WORLD, "Mobility Matrix M:\n");
  // MatView(*M, PETSC_VIEWER_STDOUT_WORLD);
}

void ParticleManager::run() {
  for (int i = 0; i < 1; i++) {
    commitNewParticles();

    timeStep();
  }
}
