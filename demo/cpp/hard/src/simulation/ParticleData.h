#pragma once

#include <petsc.h>

#include <array>

// A plain-old-data (POD) struct for serializing Particle objects for MPI communication.
struct ParticleData {
  PetscInt gID;
  std::array<double, 3> position;
  std::array<double, 4> quaternion;
  std::array<double, 3> force;
  std::array<double, 3> torque;
  std::array<double, 3> velocityLinear;
  std::array<double, 3> velocityAngular;
  double impedance;
  double length;
  double l0;
  double diameter;
};