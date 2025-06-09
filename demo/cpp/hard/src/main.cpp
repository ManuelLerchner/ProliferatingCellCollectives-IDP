#include <mpi.h>
#include <petsc.h>

#include "simulation/ParticleManager.h"
#include "util/Config.h"

int main(int argc, char** argv) {
  PetscInitialize(&argc, &argv, nullptr, nullptr);

  PhysicsConfig physic_config = {
      .xi = 200 * 3600,
      .TAU = 54 * 60,
      .l0 = 1.0,
      .LAMBDA = 2.44e-1,
  };
  SolverConfig solver_config = {.dt = 180, .tolerance = 1e-8, .max_iterations = 1000};

  ParticleManager system(physic_config, solver_config);

  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  if (rank == 0) {
    double angle = 0;
    Particle p1 = Particle(0, {physic_config.l0 / 2 * 0.99, 0.0, 0.0}, {cos(angle / 2), 0, 0, sin(angle / 2)}, physic_config.l0, physic_config.l0 / 2);
    Particle p2 = Particle(1, {-physic_config.l0 / 2 * 0.99, 0.0, 0.0}, {cos(angle / 2), 0, 0, sin(angle / 2)}, physic_config.l0, physic_config.l0 / 2);
    system.queueNewParticle(p1);
    system.queueNewParticle(p2);
  }

  system.run();

  PetscFinalize();
  return EXIT_SUCCESS;
}