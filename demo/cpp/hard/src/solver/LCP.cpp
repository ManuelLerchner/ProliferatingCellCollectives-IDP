#include "LCP.h"

Vec solveLCP(Mat A, Vec b) {
  Vec gamma;
  VecCreate(PETSC_COMM_WORLD, &gamma);

  //  fill gamam iwth zeros for now

  VecSetSizes(gamma, PETSC_DECIDE, 1);
  VecSetFromOptions(gamma);

  VecSet(gamma, 0.0);

  return gamma;
}