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

ParticleManager::ParticleManager(double dt) : dt(dt) {
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
    new_particle_buffer[i].setId(new_ids[i]);
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
            [](const Particle& a, const Particle& b) { return a.getId() < b.getId(); });

  PetscInt local_dims = components_per_particle * local_num_particles;

  std::vector<PetscInt> ownership_map(local_dims);
  for (PetscInt i = 0; i < local_num_particles; ++i) {
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

VecWrapper ParticleManager::estimate_phi_next(const VecWrapper& phi, const MatWrapper& D, const MatWrapper& M, const VecWrapper& gamma, double dt) {
  // phi_next = phi + dt * (D^T @ G @ M @ D @ gamma)

  // t1 = D @ gamma
  VecWrapper t1;
  MatCreateVecs(D.get(), NULL, t1.get_ref());
  MatMult(D, gamma.get(), t1.get());

  // t2 = M @ t1
  VecWrapper t2;
  MatCreateVecs(M.get(), NULL, t2.get_ref());
  MatMult(M, t1.get(), t2.get());

  // t3 = D^T @ t2

  VecWrapper t3;
  MatCreateVecs(D.get(), t3.get_ref(), NULL);
  MatMultTranspose(D, t2.get(), t3.get());

  // phi_next = phi.copy()
  VecWrapper phi_next;
  VecDuplicate(phi.get(), phi_next.get_ref());
  VecCopy(phi.get(), phi_next.get());

  // phi_next = phi_next + dt * t3
  VecAXPY(phi_next.get(), dt, t3.get());

  return std::move(phi_next);
}

void ParticleManager::updateLocalParticlesFromVector(const VecWrapper& dC) {
  // Get local portion of dC vector to update particles
  const PetscScalar* dC_array;
  VecGetArrayRead(dC.get(), &dC_array);

  PetscInt local_size;
  VecGetLocalSize(dC.get(), &local_size);

  PetscPrintf(PETSC_COMM_WORLD, "Local dC vector size: %d, num particles: %d\n", local_size, (int)local_particles.size());

  // Update local particles using helper functions
  int expected_size = local_particles.size() * Particle::getStateSize();
  if (local_size < expected_size) {
    PetscPrintf(PETSC_COMM_WORLD, "Warning: vector size mismatch. Expected %d, got %d\n", expected_size, local_size);
  }

  for (int i = 0; i < local_particles.size(); i++) {
    int base_offset = i * Particle::getStateSize();
    if (base_offset + Particle::getStateSize() <= local_size) {
      // Update particle state using helper function
      local_particles[i].updateState(dC_array, i, dt);

      // Normalize quaternion to maintain unit length
      local_particles[i].normalizeQuaternion();
    }
  }

  VecRestoreArrayRead(dC.get(), &dC_array);
}

void ParticleManager::timeStep() {
  std::vector<Constraint> local_constraints;

  for (int i = 0; i < local_particles.size(); i += 2) {
    if (i + 1 < local_particles.size()) {
      int id_i = local_particles[i].getId();
      int id_j = local_particles[i + 1].getId();
      local_constraints.emplace_back(Constraint(
          -0.000000009999999999999999,
          id_i,
          id_j,
          {1, 0, 0},              // normI
          {-0.0000000025, 0, 0},  // posI
          {0.0000000025, 0, 0},   // posJ
          {0.00000000245, 0, 0},  // labI
          {-0.00000000245, 0, 0}  // labJ
          ));
    }
  }

  ISLocalToGlobalMappingWrapper col_map_6d = createLocalToGlobalMapping(local_particles.size(), 6);
  ISLocalToGlobalMappingWrapper row_map_7d = createLocalToGlobalMapping(local_particles.size(), 7);
  ISLocalToGlobalMappingWrapper constraint_map = create_constraint_map(local_constraints.size());

  double xi = 200 * 3600;
  double TAU = (54 * 60);

  double l0 = 1.0;

  // tolerance agains growth
  //  LAMBDA_ecoli = 2.44e-1
  //  LAMBDA = 0 -> pressure has no effect on growth -> exponential growth

  double LAMBDA = 2.44e-1;

  double LAMBDA_DIMENSIONLESS = (TAU / (xi * l0 * l0)) * LAMBDA;

  MatWrapper D = calculate_Jacobian(local_constraints, local_particles.size(), global_particle_count, row_map_7d, constraint_map);
  // PetscPrintf(PETSC_COMM_WORLD, "D Matrix:\n");
  // MatView(D.get(), PETSC_VIEWER_STDOUT_WORLD);

  MatWrapper M = calculate_MobilityMatrix(local_particles, xi, global_particle_count, col_map_6d);
  // PetscPrintf(PETSC_COMM_WORLD, "Mobility Matrix M:\n");
  // MatView(M.get(), PETSC_VIEWER_STDOUT_WORLD);

  MatWrapper G = calculate_QuaternionMap(local_particles, global_particle_count, row_map_7d, col_map_6d);
  // PetscPrintf(PETSC_COMM_WORLD, "Quaternion Map G:\n");
  // MatView(G.get(), PETSC_VIEWER_STDOUT_WORLD);

  VecWrapper phi = create_phi_vector(local_constraints, constraint_map);
  // PetscPrintf(PETSC_COMM_WORLD, "Phi Vector:\n");
  // VecView(phi.get(), PETSC_VIEWER_STDOUT_WORLD);

  // make function to calculate the gradient of phi_next
  auto gradient = [&](const VecWrapper& gamma) -> VecWrapper {
    return estimate_phi_next(phi, D, M, gamma, dt);
  };

  auto residual = [&](const VecWrapper& gradient_val, const VecWrapper& gamma) -> double {
    const PetscScalar *gamma_array, *grad_array;

    VecGetArrayRead(gamma.get(), &gamma_array);
    VecGetArrayRead(gradient_val.get(), &grad_array);

    VecWrapper projected;
    VecDuplicate(gradient_val.get(), projected.get_ref());

    PetscScalar* proj_array;
    VecGetArray(projected.get(), &proj_array);

    PetscInt n;
    VecGetLocalSize(gamma.get(), &n);
    for (PetscInt i = 0; i < n; i++) {
      proj_array[i] = (PetscRealPart(gamma_array[i]) > 0)
                          ? grad_array[i]
                          : std::min(0.0, PetscRealPart(grad_array[i]));
    }
    VecRestoreArray(projected.get(), &proj_array);

    VecRestoreArrayRead(gradient_val.get(), &grad_array);
    VecRestoreArrayRead(gamma.get(), &gamma_array);

    double norm;
    VecNorm(projected.get(), NORM_INFINITY, &norm);
    return norm;
  };

  VecWrapper gamma;
  VecCreate(PETSC_COMM_WORLD, gamma.get_ref());
  VecSetSizes(gamma, local_constraints.size(), PETSC_DETERMINE);
  VecSetType(gamma, VECSTANDARD);
  VecZeroEntries(gamma.get());

  VecWrapper gamma_next = BBPGD(gradient, residual, gamma, 1e-8, 1000);

  VecWrapper df;
  MatCreateVecs(D.get(), NULL, df.get_ref());
  MatMult(D, gamma_next.get(), df.get());

  VecWrapper dU;
  MatCreateVecs(M.get(), NULL, dU.get_ref());
  MatMult(M, df.get(), dU.get());

  VecWrapper dC;
  MatCreateVecs(G.get(), NULL, dC.get_ref());
  MatMult(G, dU.get(), dC.get());

  updateLocalParticlesFromVector(dC);

  for (int i = 0; i < local_particles.size(); i++) {
    local_particles[i].printState();
  }
}

void ParticleManager::run() {
  for (int i = 0; i < 1; i++) {
    commitNewParticles();

    timeStep();
  }
}
