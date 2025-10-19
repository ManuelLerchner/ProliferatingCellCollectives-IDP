
#include "Soft.h"

#include "dynamics/Physics.h"
#include "dynamics/models/hard/HardModelGradient.h"
#include "util/ArrayMath.h"

void calculate_forces(VecWrapper& F, const MatWrapper& M, const MatWrapper& G, const VecWrapper& U_ext, VecWrapper& U_total, VecWrapper& deltaC) {
  VecWrapper U_c = VecWrapper::FromMat(M);

  // U = M @ f
  PetscCallAbort(PETSC_COMM_WORLD, MatMult(M, F, U_c));

  // U = U + U_ext
  PetscCallAbort(PETSC_COMM_WORLD, VecWAXPY(U_total, 1.0, U_ext, U_c));

  // deltaC = G @ U
  PetscCallAbort(PETSC_COMM_WORLD, MatMult(G, U_total, deltaC));
}

ParticleManager::SolverSolution solveSoftPotential(ParticleManager& particle_manager, CollisionDetector& collision_detector, SimulationParameters& params, double dt, int iter, std::function<void()> exchangeGhostParticles, vtk::ParticleLogger& particle_logger, vtk::ConstraintLogger& constraint_logger) {
  using namespace utils::ArrayMath;

  exchangeGhostParticles();

  MatWrapper M = calculate_MobilityMatrix(particle_manager.local_particles, params.physics_config.xi);
  MatWrapper G = calculate_QuaternionMap(particle_manager.local_particles);

  // Reset collision detector state
  collision_detector.reset();

  // Use a larger tolerance for initial collision detection
  double tolerance = 0.1;

  auto new_constraints = collision_detector.detectCollisions(particle_manager, 0, 0);

  double max_overlap = globalReduce(std::accumulate(new_constraints.begin(), new_constraints.end(), 0.0,
                                                    [](double acc, const Constraint& c) {
                                                      return std::max(acc, c.signed_distance < 0 ? -c.signed_distance : 0);
                                                    }),
                                    MPI_MAX);

  VecWrapper F = VecWrapper::Create(6 * particle_manager.local_particles.size());
  VecWrapper stress = VecWrapper::Create(particle_manager.local_particles.size());

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

    // Accumulate force
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

    // Accumulate stress
    PetscCallAbort(PETSC_COMM_WORLD, VecSetValue(stress, constraint.gidI, constraint.stressI * F_elastic, ADD_VALUES));
    PetscCallAbort(PETSC_COMM_WORLD, VecSetValue(stress, constraint.gidJ, constraint.stressJ * F_elastic, ADD_VALUES));
  }

  VecAssemblyBegin(F);
  VecAssemblyEnd(F);
  VecAssemblyBegin(stress);
  VecAssemblyEnd(stress);

  // Move particles
  VecWrapper U_ext = VecWrapper::Create(6 * particle_manager.local_particles.size());
  VecWrapper U = VecWrapper::Like(U_ext);

  calculate_external_velocities(U_ext, particle_manager.local_particles, M, dt, 0, params.physics_config);

  VecWrapper deltaC = VecWrapper::FromMat(G);
  calculate_forces(F, M, G, U_ext, U, deltaC);

  particle_manager.moveLocalParticlesFromSolution({.dC = deltaC, .f = F, .u = U}, dt);

  // Grow particles
  VecWrapper length = getLengthVector(particle_manager.local_particles);
  VecWrapper ldot = VecWrapper::Like(length);
  VecWrapper impedance = VecWrapper::Like(length);
  VecCopy(stress, impedance);

  calculate_growth_rate_vector(length, impedance, params.physics_config.getLambdaDimensionless(), params.physics_config.TAU, ldot);

  // Grow particles
  particle_manager.setGrowParamsFromSolution({.dL = ldot, .impedance = impedance, .stress = stress});
  particle_manager.grow(dt);

  return {.constraints = new_constraints, .constraint_iterations = 0, .bbpgd_iterations = 0, .residual = max_overlap, .max_overlap = max_overlap};
}
