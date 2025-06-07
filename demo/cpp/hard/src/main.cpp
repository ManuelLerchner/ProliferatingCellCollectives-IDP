#include <mpi.h>
#include <petsc.h>

#include "dynamics/ParticleManager.h"

int main(int argc, char** argv) {
  PetscInitialize(&argc, &argv, nullptr, nullptr);

  ParticleManager system;

  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  // if (rank == 0) {
    double angle = 0;
    double l0 = 1.0;
    Particle p1 = {.id = 0, .position = {0.5, 0.1, 0.0}, .quaternion = {cos(angle / 2), 0, 0, sin(angle / 2)}, .length = l0, .diameter = l0 / 2};
    Particle p2 = {.id = 1, .position = {1.0, -0.1, 0.0}, .quaternion = {cos(angle / 2), 0, 0, sin(angle / 2)}, .length = l0, .diameter = l0 / 2};
    system.queueNewParticle(p1);
    system.queueNewParticle(p2);
  // }

  system.run();

  PetscFinalize();
  return EXIT_SUCCESS;
}