#pragma once
#include <petsc.h>

#include <array>
#include <cmath>
#include <iostream>

// Constants for vector operations
static constexpr int POSITION_SIZE = 3;
static constexpr int QUATERNION_SIZE = 4;
static constexpr int STATE_SIZE = POSITION_SIZE + QUATERNION_SIZE;  // 7 total

class Particle {
  PetscInt id;  // Global unique ID for the particle
  std::array<double, POSITION_SIZE> position;
  std::array<double, QUATERNION_SIZE> quaternion;
  double length;
  double diameter;

 public:
  Particle(PetscInt id, const std::array<double, POSITION_SIZE>& position, const std::array<double, QUATERNION_SIZE>& quaternion, double length, double diameter) : id(id), position(position), quaternion(quaternion), length(length), diameter(diameter) {}

  // Update position from vector components
  void updatePosition(const PetscScalar* data, int offset, double dt) {
    position[0] += dt * PetscRealPart(data[offset + 0]);
    position[1] += dt * PetscRealPart(data[offset + 1]);
    position[2] += dt * PetscRealPart(data[offset + 2]);
  }

  // Update quaternion from vector components
  void updateQuaternion(const PetscScalar* data, int offset, double dt) {
    quaternion[0] += dt * PetscRealPart(data[offset + 0]);
    quaternion[1] += dt * PetscRealPart(data[offset + 1]);
    quaternion[2] += dt * PetscRealPart(data[offset + 2]);
    quaternion[3] += dt * PetscRealPart(data[offset + 3]);
  }

  // Update both position and quaternion from vector
  void updateState(const PetscScalar* data, int particle_index, double dt) {
    int base_offset = particle_index * STATE_SIZE;
    updatePosition(data, base_offset, dt);
    updateQuaternion(data, base_offset + POSITION_SIZE, dt);
  }

  // Normalize quaternion to maintain unit length
  void normalizeQuaternion() {
    double norm = std::sqrt(quaternion[0] * quaternion[0] +
                            quaternion[1] * quaternion[1] +
                            quaternion[2] * quaternion[2] +
                            quaternion[3] * quaternion[3]);

    if (norm > 1e-12) {
      quaternion[0] /= norm;
      quaternion[1] /= norm;
      quaternion[2] /= norm;
      quaternion[3] /= norm;
    }
  }

  void printState() const {
    fprintf(stdout, "Particle %d: pos=[%f, %f, %f], quat=[%f, %f, %f, %f]\n",
            id,
            position[0], position[1], position[2],
            quaternion[0], quaternion[1],
            quaternion[2], quaternion[3]);
  }

  // Get state size for this particle type
  static constexpr int getStateSize() {
    return STATE_SIZE;
  }

  PetscInt getId() const {
    return id;
  }

  void setId(PetscInt id) {
    this->id = id;
  }

  const std::array<double, POSITION_SIZE>& getPosition() const {
    return position;
  }

  double getLength() const {
    return length;
  }

  double getDiameter() const {
    return diameter;
  }

  const std::array<double, QUATERNION_SIZE>& getQuaternion() const {
    return quaternion;
  }
};