#include <mpi.h>
#include <petsc.h>
#include <petscconf.h>

#include <random>

#include "spatial/Domain.h"
#include "util/Config.h"

int main(int argc, char** argv) {
  PetscInitialize(&argc, &argv, 0, 0);
  {
    PetscLogDefaultBegin();

    int total_ranks;
    MPI_Comm_size(PETSC_COMM_WORLD, &total_ranks);

    PetscPrintf(PETSC_COMM_WORLD, "Running simulation with %d ranks\n", total_ranks);

    double DT = 30;
    double END_TIME = 300 * 60;
    double LOG_FREQUENCY = 1 * 60;

    PhysicsConfig physic_config = {
        .xi = 200 * 3600,
        .TAU = 54 * 60,
        .l0 = 1,
        .LAMBDA = 2.44e-3,
        .temperature = 0.01,
        .monolayer = true,
        .gravity = {0.0, 0.0, 0.0},
    };

    SimulationConfig sim_config = {
        .dt = DT,
        .end_time = END_TIME,
        .log_frequency_seconds = LOG_FREQUENCY,
        .min_box_size = {physic_config.l0 + 5, physic_config.l0 + 5, 0},
        .domain_resize_frequency = 50,

        .enable_adaptive_dt = true,
        .target_bbpgd_iterations = 100,
        .dt_adjust_frequency = 10,
        .dt_adjust_factor = 0.1,
        .min_dt = 1e-6,
        .max_dt = 30};

    SolverConfig solver_config = {
        .tolerance = physic_config.l0 / 1e3,
        .max_bbpgd_iterations = 10000,
        .max_recursive_iterations = 500,
        .linked_cell_size = physic_config.l0 * 2.2,
        .min_preallocation_size = 0,
        .growth_factor = 2,
        .max_constraints_per_pair = 6,
    };

    Domain domain(sim_config, physic_config, solver_config);

    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    if (rank == 0) {
      double angle = 0;
      Particle p1 = Particle(0, {rank * 3.0, 0, 0}, {cos(angle / 2), 0, 0, sin(angle / 2)}, physic_config.l0, physic_config.l0, physic_config.l0 / 2);

      domain.queueNewParticles({p1});
    }

    domain.run();

    PetscLogView(PETSC_VIEWER_STDOUT_WORLD);
  }
  PetscFinalize();
  return EXIT_SUCCESS;
}