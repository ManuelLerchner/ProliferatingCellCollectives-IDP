#include "Particle.h"

#include <petsc.h>

#include <cmath>
#include <iostream>

Particle::Particle(PetscInt gID, const std::array<double, POSITION_SIZE>& position, const std::array<double, QUATERNION_SIZE>& quaternion, double length, double diameter) : gID(gID), position(position), quaternion(quaternion), length(length), diameter(diameter) {}

void Particle::updatePosition(const PetscScalar* dC, int offset, double dt) {
  position[0] += dt * PetscRealPart(dC[offset + 0]);
  position[1] += dt * PetscRealPart(dC[offset + 1]);
  position[2] += dt * PetscRealPart(dC[offset + 2]);
}

void Particle::updateQuaternion(const PetscScalar* dC, int offset, double dt) {
  quaternion[0] += dt * PetscRealPart(dC[offset + 0]);
  quaternion[1] += dt * PetscRealPart(dC[offset + 1]);
  quaternion[2] += dt * PetscRealPart(dC[offset + 2]);
  quaternion[3] += dt * PetscRealPart(dC[offset + 3]);
  normalizeQuaternion();
}

void Particle::addForce(const PetscScalar* df, int offset) {
  force[0] += PetscRealPart(df[offset + 0]);
  force[1] += PetscRealPart(df[offset + 1]);
  force[2] += PetscRealPart(df[offset + 2]);
}

void Particle::addTorque(const PetscScalar* df, int offset) {
  torque[0] += PetscRealPart(df[offset + 0]);
  torque[1] += PetscRealPart(df[offset + 1]);
  torque[2] += PetscRealPart(df[offset + 2]);
}

void Particle::eulerStep(const PetscScalar* dC, int particle_index, double dt) {
  int base_offset = particle_index * STATE_SIZE;
  updatePosition(dC, base_offset, dt);
  updateQuaternion(dC, base_offset + POSITION_SIZE, dt);

  // Final validation after complete state update
  validateAndWarn();
}

void Particle::clearForceAndTorque() {
  force[0] = 0.0;
  force[1] = 0.0;
  force[2] = 0.0;
  torque[0] = 0.0;
  torque[1] = 0.0;
  torque[2] = 0.0;
}

void Particle::addForceAndTorque(const PetscScalar* f, const PetscScalar* U, int particle_index) {
  int base_offset = particle_index * 6;
  addForce(f, base_offset);
  addTorque(f, base_offset + 3);

  // Final validation after complete state update
  validateAndWarn();
}

void Particle::normalizeQuaternion() {
  double norm = std::sqrt(quaternion[0] * quaternion[0] +
                          quaternion[1] * quaternion[1] +
                          quaternion[2] * quaternion[2] +
                          quaternion[3] * quaternion[3]);

  if (norm > 1e-12) {
    quaternion[0] /= norm;
    quaternion[1] /= norm;
    quaternion[2] /= norm;
    quaternion[3] /= norm;
  } else {
    // Fallback to unit quaternion if norm is too small
    PetscPrintf(PETSC_COMM_WORLD, "WARNING: Particle %d quaternion norm too small (%g), resetting to unit quaternion\n", gID, norm);
    quaternion[0] = 1.0;
    quaternion[1] = 0.0;
    quaternion[2] = 0.0;
    quaternion[3] = 0.0;
  }

  // Validate after normalization
  validateAndWarn();
}

bool Particle::isValid() const {
  // Check position
  for (int i = 0; i < POSITION_SIZE; ++i) {
    if (!std::isfinite(position[i])) {
      return false;
    }
  }

  // Check quaternion
  for (int i = 0; i < QUATERNION_SIZE; ++i) {
    if (!std::isfinite(quaternion[i])) {
      return false;
    }
  }

  // Check physical properties
  if (!std::isfinite(length) || !std::isfinite(diameter)) {
    return false;
  }

  return true;
}

void Particle::validateAndWarn() const {
  // Check position for NaN/infinity
  for (int i = 0; i < POSITION_SIZE; ++i) {
    if (!std::isfinite(position[i])) {
      PetscPrintf(PETSC_COMM_WORLD, "WARNING: Particle %d position[%d] = %g (non-finite)\n", gID, i, position[i]);
    }
  }

  // Check quaternion for NaN/infinity
  for (int i = 0; i < QUATERNION_SIZE; ++i) {
    if (!std::isfinite(quaternion[i])) {
      PetscPrintf(PETSC_COMM_WORLD, "WARNING: Particle %d quaternion[%d] = %g (non-finite)\n", gID, i, quaternion[i]);
    }
  }

  // Check physical properties
  if (!std::isfinite(length)) {
    PetscPrintf(PETSC_COMM_WORLD, "WARNING: Particle %d length = %g (non-finite)\n", gID, length);
  }

  if (!std::isfinite(diameter)) {
    PetscPrintf(PETSC_COMM_WORLD, "WARNING: Particle %d diameter = %g (non-finite)\n", gID, diameter);
  }

  // Check for extremely large values that might indicate numerical issues
  for (int i = 0; i < POSITION_SIZE; ++i) {
    if (std::abs(position[i]) > 1e6) {
      PetscPrintf(PETSC_COMM_WORLD, "WARNING: Particle %d position[%d] = %g (extremely large)\n", gID, i, position[i]);
    }
  }
}

void Particle::printState() const {
  fprintf(stdout, "Particle %d: pos=[%f, %f, %f], quat=[%f, %f, %f, %f]\n",
          gID,
          position[0], position[1], position[2],
          quaternion[0], quaternion[1],
          quaternion[2], quaternion[3]);
}

PetscInt Particle::setGID() const {
  return gID;
}

void Particle::setGID(PetscInt gID) {
  this->gID = gID;
}

const std::array<double, POSITION_SIZE>& Particle::getPosition() const {
  return position;
}

const std::array<double, 3>& Particle::getForce() const {
  return force;
}

const std::array<double, 3>& Particle::getTorque() const {
  return torque;
}

double Particle::getLength() const {
  return length;
}

double Particle::getDiameter() const {
  return diameter;
}

const std::array<double, QUATERNION_SIZE>& Particle::getQuaternion() const {
  return quaternion;
}
