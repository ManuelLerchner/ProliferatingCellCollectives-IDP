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
  double LOG_FREQUENCY = 30;

  SimulationConfig sim_config = {
      .dt = DT,
      .end_time = END_TIME,
      .log_frequency_seconds = LOG_FREQUENCY};

  PhysicsConfig physic_config = {
      .xi = 200 * 3600,
      .TAU = 54 * 60,
      .l0 = 1,
      .LAMBDA = 2.44e-1,
      .temperature = 0,
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
  std::mt19937 gen;
  gen.seed(42);

  for (int i = 0; i < 1; i++) {
    std::normal_distribution<> d(0, 1);

    if (total_ranks == 1) {
      double angle1 = 0;
      double angle2 = 0.1;
      Particle p1 = Particle(0, {0, 0, 0}, {cos(angle1 / 2), 0, 0, sin(angle1 / 2)}, physic_config.l0, physic_config.l0, physic_config.l0 / 2);
      Particle p2 = Particle(1, {2, 0, 0}, {cos(angle2 / 2), 0, 0, sin(angle2 / 2)}, physic_config.l0, physic_config.l0, physic_config.l0 / 2);
      system.queueNewParticle(p1);
      system.queueNewParticle(p2);
    }

    if (total_ranks == 2) {
      double x = 2 * rank;
      double angle = 0.1 * rank;
      Particle p1 = Particle(1, {x, 0, 0}, {cos(angle / 2), 0, 0, sin(angle / 2)}, physic_config.l0, physic_config.l0, physic_config.l0 / 2);
      system.queueNewParticle(p1);
    }
  }

  int num_steps = sim_config.end_time / sim_config.dt;
  system.run(600);

  PetscFinalize();
  return EXIT_SUCCESS;
}