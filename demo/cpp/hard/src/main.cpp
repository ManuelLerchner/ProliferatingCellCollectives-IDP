#include <mpi.h>
#include <petsc.h>

#include <random>

#include "spatial/Domain.h"
#include "util/Config.h"

int main(int argc, char** argv) {
  PetscInitialize(&argc, &argv, nullptr, nullptr);

  int total_ranks;
  MPI_Comm_size(PETSC_COMM_WORLD, &total_ranks);

  PetscPrintf(PETSC_COMM_WORLD, "Running simulation with %d ranks\n", total_ranks);

  double DT = 30;
  double END_TIME = 7 * 60 * 60;
  double LOG_FREQUENCY = 120;

  SimulationConfig sim_config = {
      .dt = DT,
      .end_time = END_TIME,
      .log_frequency_seconds = LOG_FREQUENCY,
      .min_box_size = {5.0, 5.0, 0},
      .domain_resize_frequency = 1};

  PhysicsConfig physic_config = {
      .xi = 200 * 3600,
      .TAU = 54 * 60,
      .l0 = 1,
      .LAMBDA = 2.44e-1,
      .temperature = 0.01,
      .monolayer = true,
      .gravity = {0.0, 0.0, 0.0},
  };

  SolverConfig solver_config = {
      .tolerance = physic_config.l0 / 1e3,
      .max_bbpgd_iterations = 100000,
      .max_recursive_iterations = 50};

  Domain domain(sim_config, physic_config, solver_config);

  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  if (rank == 0) {
    double angle = 0;
    Particle p1 = Particle(0, {0, rank * 1.5, 0}, {cos(angle / 2), 0, 0, sin(angle / 2)}, physic_config.l0, physic_config.l0, physic_config.l0 / 2);

    domain.queueNewParticles({p1});
  }

  domain.run();

  PetscFinalize();
  return EXIT_SUCCESS;
}