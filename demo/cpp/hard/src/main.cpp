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

  double DT = 30;
  double END_TIME = 4 * 60 * 60;
  double LOG_FREQUENCY = 60;

  SimulationConfig sim_config = {
      .dt = DT,
      .end_time = END_TIME,
      .log_frequency_seconds = LOG_FREQUENCY};

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

  ParticleManager system(sim_config, physic_config, solver_config);

  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  int number_of_particles = 150.0 / (2 * total_ranks);

  std::random_device rd;
  std::mt19937 gen(55);

  for (int i = 0; i < 1; i++) {
    std::normal_distribution<> d(0, 1);

    double angle = d(gen) * 2 * M_PI;

    Particle p1 = Particle(0, {rank * 1.5, 0, 0}, {cos(angle / 2), 0, 0, sin(angle / 2)}, physic_config.l0, physic_config.l0, physic_config.l0 / 2);

    system.queueNewParticle(p1);
  }


  // todo: rebalancing
  // todo: maybe use smaller matrices size. only for owned constraints

  int num_steps = sim_config.end_time / sim_config.dt;
  system.run(600);

  PetscFinalize();
  return EXIT_SUCCESS;
}