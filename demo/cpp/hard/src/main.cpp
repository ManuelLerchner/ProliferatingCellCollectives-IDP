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

  double DT = 30;                 // seconds
  double END_TIME = 5 * 60 * 60;  // 5 hours
  double LOG_FREQUENCY = 1;  // 5 minutes

  SimulationConfig sim_config = {
      .dt = DT,
      .end_time = END_TIME,
      .log_frequency_seconds = LOG_FREQUENCY};

  PhysicsConfig physic_config = {
      .xi = 1,
      .TAU = 54 * 60,
      .l0 = 1,
      .LAMBDA = 2.44e-1,
  };

  SolverConfig solver_config = {
      .tolerance = physic_config.l0 / 1e3,
      .max_bbpgd_iterations = 50000,
      .max_recursive_iterations = 5};

  ParticleManager system(sim_config, physic_config, solver_config);

  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  int number_of_particles = 150.0 / (2 * total_ranks);

  for (int i = 0; i < 1; i++) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, 1);
    double angle = d(gen) * 2 * M_PI;

    double x = 5 + rank;
    double y = 5 + rank;
    double z = 0;

    Particle p1 = Particle(0, {x, y, z}, {cos(angle / 2), 0, 0, sin(angle / 2)}, physic_config.l0, physic_config.l0, physic_config.l0 / 2);
    Particle p2 = Particle(1, {-x, y, z}, {cos(angle / 2), 0, 0, sin(angle / 2)}, physic_config.l0, physic_config.l0, physic_config.l0 / 2);

    system.queueNewParticle(p1);
    // system.queueNewParticle(p2);
  }

  int num_steps = sim_config.end_time / sim_config.dt;
  system.run(num_steps);

  PetscFinalize();
  return EXIT_SUCCESS;
}