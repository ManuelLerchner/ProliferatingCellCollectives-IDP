#include <mpi.h>
#include <petsc.h>

#include "dynamics/BacterialSystem.h"

int main(int argc, char** argv) {
  PetscInitialize(&argc, &argv, nullptr, nullptr);

  BacterialSystem system(argc, argv);
  system.run();

  // PetscFinalize();
  // return EXIT_SUCCESS;
}