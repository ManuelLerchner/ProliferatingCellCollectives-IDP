#include <mpi.h>
#include <petsc.h>

#include <random>

#include "simulation/ParticleManager.h"
#include "util/Config.h"

int main(int argc, char** argv) {
  PetscInitialize(&argc, &argv, nullptr, nullptr);

  int total_ranks;
  MPI_Comm_size(PETSC_COMM_WORLD, &total_ranks);

  PetscPrintf(PETSC_COMM_WORLD, "Running simulation with %d ranks\n", total_ranks);

  double DT = 60;  // seconds

  PhysicsConfig physic_config = {
      .xi = 200 * 3600,
      .TAU = 54 * 60,
      .l0 = 1.0,
      .LAMBDA = 2.44e-1,
  };
  SolverConfig solver_config = {.dt = DT, .tolerance = physic_config.l0 / 1e3, .max_bbpgd_iterations = 50000, .max_recursive_iterations = 3};

  ParticleManager system(physic_config, solver_config);

  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  int number_of_particles = 150.0 / (2 * total_ranks);

  for (int i = 0; i < 1; i++) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, 1);
    double angle = d(gen) * 2 * M_PI;

    double x = d(gen) * 5;
    double y = d(gen) * 5;
    double z = 0;

    Particle p1 = Particle(0, {x, y, z}, {cos(angle / 2), 0, 0, sin(angle / 2)}, physic_config.l0, physic_config.l0, physic_config.l0 / 2);
    Particle p2 = Particle(1, {-x, y, z}, {cos(angle / 2), 0, 0, sin(angle / 2)}, physic_config.l0, physic_config.l0, physic_config.l0 / 2);

    system.queueNewParticle(p1);
    // system.queueNewParticle(p2);
  }

  double END_TIME = 5 * 60 * 60;  // 5 hours

  int num_steps = END_TIME / solver_config.dt;
  system.run(num_steps);

  PetscFinalize();
  return EXIT_SUCCESS;
}