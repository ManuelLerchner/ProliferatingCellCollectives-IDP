#pragma once
#include <petsc.h>

#include <array>

// Constants for vector operations
static constexpr int POSITION_SIZE = 3;
static constexpr int QUATERNION_SIZE = 4;
static constexpr int STATE_SIZE = POSITION_SIZE + QUATERNION_SIZE;

class Particle {
 public:
  Particle(PetscInt id, const std::array<double, POSITION_SIZE>& position, const std::array<double, QUATERNION_SIZE>& quaternion, double length, double diameter);

  void updatePosition(const PetscScalar* data, int offset, double dt);

  void updateQuaternion(const PetscScalar* data, int offset, double dt);

  void updateState(const PetscScalar* data, int particle_index, double dt);

  void normalizeQuaternion();

  void printState() const;

  const std::array<double, POSITION_SIZE>& getPosition() const;
  const std::array<double, QUATERNION_SIZE>& getQuaternion() const;

  double getLength() const;

  double getDiameter() const;

  PetscInt getId() const;

  void setId(PetscInt id);

  static constexpr int getStateSize() {
    return STATE_SIZE;
  }

 private:
  PetscInt id;
  std::array<double, POSITION_SIZE> position;
  std::array<double, QUATERNION_SIZE> quaternion;
  double length;
  double diameter;
};