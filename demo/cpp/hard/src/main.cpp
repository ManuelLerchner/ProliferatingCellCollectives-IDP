#include <mpi.h>
#include <petsc.h>

#include "dynamics/ParticleManager.h"

int main(int argc, char** argv) {
  PetscInitialize(&argc, &argv, nullptr, nullptr);

  double dt = 180;

  ParticleManager system(dt);

  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  // if (rank == 0) {
  double angle = 0;
  double l0 = 1;
  Particle p1 = Particle(0, {l0 / 2 * 0.99, 0.0, 0.0}, {cos(angle / 2), 0, 0, sin(angle / 2)}, l0, l0 / 2);
  Particle p2 = Particle(1, {-l0 / 2 * 0.99, 0.0, 0.0}, {cos(angle / 2), 0, 0, sin(angle / 2)}, l0, l0 / 2);
  system.queueNewParticle(p1);
  system.queueNewParticle(p2);
  // }


  system.run();

  PetscFinalize();
  return EXIT_SUCCESS;
}