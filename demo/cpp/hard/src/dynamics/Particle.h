#pragma once
#include <petsc.h>

#include <array>

struct Particle {
  PetscInt id;  // Global unique ID for the particle
  std::array<double, 3> position;
  std::array<double, 4> quaternion;
  double length;
  double diameter;
};