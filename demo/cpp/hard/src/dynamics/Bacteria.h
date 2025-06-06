#pragma once
#include <petsc.h>

#include <array>

struct Bacterium {
  PetscInt id;  // Global unique ID for the bacterium
  std::array<double, 3> position;
  std::array<double, 4> quaternion;
  double length;
  double diameter;
};