
#include "Soft.h"

#include "dynamics/Physics.h"
#include "util/ArrayMath.h"

ParticleManager::SolverSolution solveSoftPotential(ParticleManager& particle_manager, CollisionDetector& collision_detector, SimulationParameters params, double dt, int iter, std::function<void()> exchangeGhostParticles, vtk::ParticleLogger& particle_logger, vtk::ConstraintLogger& constraint_logger) {
  using namespace utils::ArrayMath;

  exchangeGhostParticles();

  MatWrapper M = calculate_MobilityMatrix(particle_manager.local_particles, params.physics_config.xi);
  MatWrapper G = calculate_QuaternionMap(particle_manager.local_particles);

  // Reset collision detector state
  collision_detector.reset();

  // Use a larger tolerance for initial collision detection
  double tolerance = 0.1;

  auto new_constraints = collision_detector.detectCollisions(particle_manager, 0, tolerance);

  double max_overlap = globalReduce(std::accumulate(new_constraints.begin(), new_constraints.end(), 0.0,
                                                    [](double acc, const Constraint& c) {
                                                      return std::max(acc, c.signed_distance < 0 ? -c.signed_distance : 0);
                                                    }),
                                    MPI_MAX);

  VecWrapper F = VecWrapper::FromMat(M);
  VecZeroEntries(F);

  VecWrapper gamma = VecWrapper::Create(new_constraints.size());
  VecZeroEntries(gamma);

  for (const auto& constraint : new_constraints) {
    double overlap = -constraint.signed_distance;
    if (overlap <= 0.00) continue;

    const auto& p1 = constraint.localI ? particle_manager.local_particles[constraint.localIdxI] : particle_manager.ghost_particles[constraint.localIdxI];
    const auto& p2 = constraint.localJ ? particle_manager.local_particles[constraint.localIdxJ] : particle_manager.ghost_particles[constraint.localIdxJ];

    // Calculate effective radius (CellsMD3D approach)
    double R1 = p1.getDiameter() / 2.0;
    double R2 = p2.getDiameter() / 2.0;
    double R_eff = std::sqrt(R1 * R2 / (R1 + R2));

    // Calculate overlap and force
    double F_elastic = R_eff * params.physics_config.kcc * std::pow(overlap, 1.5);

    VecSetValue(gamma, constraint.gid, F_elastic, INSERT_VALUES);

    const auto f_i = constraint.normI * F_elastic;
    const auto f_j = -f_i;
    const auto r_i = constraint.rPosI;
    const auto r_j = constraint.rPosJ;
    const auto torque_i = cross_product(r_i, f_i);
    const auto torque_j = cross_product(r_j, f_j);

    double F_i[6] = {f_i[0], f_i[1], f_i[2], torque_i[0], torque_i[1], torque_i[2]};
    double F_j[6] = {f_j[0], f_j[1], f_j[2], torque_j[0], torque_j[1], torque_j[2]};

    PetscInt rows_i[6] = {constraint.gidI * 6 + 0, constraint.gidI * 6 + 1, constraint.gidI * 6 + 2,
                          constraint.gidI * 6 + 3, constraint.gidI * 6 + 4, constraint.gidI * 6 + 5};
    PetscCallAbort(PETSC_COMM_WORLD, VecSetValues(F, 6, rows_i, F_i, ADD_VALUES));

    PetscInt rows_j[6] = {constraint.gidJ * 6 + 0, constraint.gidJ * 6 + 1, constraint.gidJ * 6 + 2,
                          constraint.gidJ * 6 + 3, constraint.gidJ * 6 + 4, constraint.gidJ * 6 + 5};
    PetscCallAbort(PETSC_COMM_WORLD, VecSetValues(F, 6, rows_j, F_j, ADD_VALUES));
  }

  VecAssemblyBegin(F);
  VecAssemblyEnd(F);

  VecAssemblyBegin(gamma);
  VecAssemblyEnd(gamma);

  // Move particles
  VecWrapper F_ext = VecWrapper::FromMat(M);
  VecWrapper U_ext = VecWrapper::FromMat(M);
  calculate_external_velocities(U_ext, F_ext, particle_manager.local_particles, M, dt, 0, params.physics_config);

  VecWrapper U = VecWrapper::FromMat(M);
  VecWrapper deltaC = VecWrapper::FromMat(G);

  // U = M @ f
  PetscCallAbort(PETSC_COMM_WORLD, MatMult(M, F, U));

  // U = U + U_ext
  PetscCallAbort(PETSC_COMM_WORLD, VecAXPY(U, 1.0, U_ext));

  // deltaC = G @ U
  PetscCallAbort(PETSC_COMM_WORLD, MatMult(G, U, deltaC));

  particle_manager.moveLocalParticlesFromSolution({.dC = deltaC, .f = F, .u = U}, dt);

  // Grow particles
  VecWrapper length = getLengthVector(particle_manager.local_particles);
  VecWrapper ldot = VecWrapper::Like(length);
  VecWrapper impedance = VecWrapper::Like(length);
  VecWrapper stress = VecWrapper::Like(length);

  // Calculate growth rates using elastic forces
  MatWrapper L = MatWrapper::CreateAIJ(new_constraints.size(), particle_manager.local_particles.size());

  PetscInt ownership_start, ownership_end;
  PetscCallAbort(PETSC_COMM_WORLD, MatGetOwnershipRange(L, &ownership_start, &ownership_end));

  calculate_stress_matrix_local(L, new_constraints, ownership_start);
  MatAssemblyBegin(L, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(L, MAT_FINAL_ASSEMBLY);

  calculate_ldot_inplace(L, length, gamma, params.physics_config.getLambdaDimensionless(), params.physics_config.TAU, ldot, stress, impedance);

  // Grow particles
  particle_manager.setGrowParamsFromSolution({.dL = ldot, .impedance = impedance, .stress = stress});
  particle_manager.grow(dt);

  return {.constraints = new_constraints, .constraint_iterations = 1, .bbpgd_iterations = 0, .residual = 0, .max_overlap = max_overlap};
}
