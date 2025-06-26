#include "Particle.h"

#include <petsc.h>

#include <cmath>
#include <iostream>
#include <random>

#include "ParticleData.h"
#include "util/ArrayMath.h"
#include "util/Quaternion.h"

Particle::Particle(PetscInt gID, const std::array<double, POSITION_SIZE>& position, const std::array<double, QUATERNION_SIZE>& quaternion, double length, double l0, double diameter) {
  data_.gID = gID;
  data_.position = position;
  data_.quaternion = quaternion;
  data_.length = length;
  data_.l0 = l0;
  data_.diameter = diameter;
  force_ = {0.0, 0.0, 0.0};
  torque_ = {0.0, 0.0, 0.0};
  velocityLinear_ = {0.0, 0.0, 0.0};
  velocityAngular_ = {0.0, 0.0, 0.0};
  impedance_ = 0.0;
  data_.age = 1;
}

Particle::Particle(const ParticleData& data) : data_(data) {
  force_ = {0.0, 0.0, 0.0};
  torque_ = {0.0, 0.0, 0.0};
  velocityLinear_ = {0.0, 0.0, 0.0};
  velocityAngular_ = {0.0, 0.0, 0.0};
  impedance_ = 0.0;
}

void Particle::updatePosition(const PetscScalar* dC, int offset, double dt) {
  data_.position[0] += dt * PetscRealPart(dC[offset + 0]);
  data_.position[1] += dt * PetscRealPart(dC[offset + 1]);
  data_.position[2] += dt * PetscRealPart(dC[offset + 2]);
}

void Particle::updateQuaternion(const PetscScalar* dC, int offset, double dt) {
  data_.quaternion[0] += dt * PetscRealPart(dC[offset + 0]);
  data_.quaternion[1] += dt * PetscRealPart(dC[offset + 1]);
  data_.quaternion[2] += dt * PetscRealPart(dC[offset + 2]);
  data_.quaternion[3] += dt * PetscRealPart(dC[offset + 3]);
  normalizeQuaternion();
}

void Particle::updateLength(double ldot, double dt) {
  data_.length += dt * ldot;

  if (data_.length > 2 * data_.l0) {
    data_.length = 2 * data_.l0;
  }
}

void Particle::addForce(const PetscScalar* df, int offset) {
  force_[0] += PetscRealPart(df[offset + 0]);
  force_[1] += PetscRealPart(df[offset + 1]);
  force_[2] += PetscRealPart(df[offset + 2]);
}

void Particle::addTorque(const PetscScalar* df, int offset) {
  torque_[0] += PetscRealPart(df[offset + 0]);
  torque_[1] += PetscRealPart(df[offset + 1]);
  torque_[2] += PetscRealPart(df[offset + 2]);
}

void Particle::addVelocityLinear(const PetscScalar* dU) {
  velocityLinear_[0] += PetscRealPart(dU[0]);
  velocityLinear_[1] += PetscRealPart(dU[1]);
  velocityLinear_[2] += PetscRealPart(dU[2]);
}

void Particle::addVelocityAngular(const PetscScalar* dU) {
  velocityAngular_[0] += PetscRealPart(dU[0]);
  velocityAngular_[1] += PetscRealPart(dU[1]);
  velocityAngular_[2] += PetscRealPart(dU[2]);
}

void Particle::eulerStepPosition(const double* dC, double dt) {
  updatePosition(dC, 0, dt);
  updateQuaternion(dC, POSITION_SIZE, dt);

  // Final validation after complete state update
  validateAndWarn();
}

void Particle::eulerStepLength(const double ldot, double dt) {
  updateLength(ldot, dt);
}

void Particle::clearForceAndTorque() {
  force_[0] = 0.0;
  force_[1] = 0.0;
  force_[2] = 0.0;
  torque_[0] = 0.0;
  torque_[1] = 0.0;
  torque_[2] = 0.0;
}

void Particle::addForceAndTorque(const PetscScalar* f) {
  addForce(f, 0);
  addTorque(f, 3);

  // Final validation after complete state update
  validateAndWarn();
}

void Particle::addVelocity(const PetscScalar* dU) {
  addVelocityLinear(dU);
  addVelocityAngular(dU + 3);
}

void Particle::normalizeQuaternion() {
  double norm = std::sqrt(data_.quaternion[0] * data_.quaternion[0] +
                          data_.quaternion[1] * data_.quaternion[1] +
                          data_.quaternion[2] * data_.quaternion[2] +
                          data_.quaternion[3] * data_.quaternion[3]);

  data_.quaternion[0] /= norm;
  data_.quaternion[1] /= norm;
  data_.quaternion[2] /= norm;
  data_.quaternion[3] /= norm;

  // Validate after normalization
  validateAndWarn();
}

bool Particle::isValid() const {
  // Check position
  for (int i = 0; i < POSITION_SIZE; ++i) {
    if (!std::isfinite(data_.position[i])) {
      return false;
    }
  }

  // Check quaternion
  for (int i = 0; i < QUATERNION_SIZE; ++i) {
    if (!std::isfinite(data_.quaternion[i])) {
      return false;
    }
  }

  // Check physical properties
  if (!std::isfinite(data_.length) || !std::isfinite(data_.diameter)) {
    return false;
  }

  return true;
}

void Particle::validateAndWarn() const {
  // Check position for NaN/infinity
  for (int i = 0; i < POSITION_SIZE; ++i) {
    if (!std::isfinite(data_.position[i])) {
      PetscPrintf(PETSC_COMM_WORLD, "WARNING: Particle %d position[%d] = %g (non-finite)\n", data_.gID, i, data_.position[i]);
    }
  }

  // Check quaternion for NaN/infinity
  for (int i = 0; i < QUATERNION_SIZE; ++i) {
    if (!std::isfinite(data_.quaternion[i])) {
      PetscPrintf(PETSC_COMM_WORLD, "WARNING: Particle %d quaternion[%d] = %g (non-finite)\n", data_.gID, i, data_.quaternion[i]);
    }
  }

  // Check physical properties
  if (!std::isfinite(data_.length)) {
    PetscPrintf(PETSC_COMM_WORLD, "WARNING: Particle %d length = %g (non-finite)\n", data_.gID, data_.length);
  }

  if (!std::isfinite(data_.diameter)) {
    PetscPrintf(PETSC_COMM_WORLD, "WARNING: Particle %d diameter = %g (non-finite)\n", data_.gID, data_.diameter);
  }

  // Check for extremely large values that might indicate numerical issues
  for (int i = 0; i < POSITION_SIZE; ++i) {
    if (std::abs(data_.position[i]) > 1e6) {
      PetscPrintf(PETSC_COMM_WORLD, "WARNING: Particle %d position[%d] = %g (extremely large)\n", data_.gID, i, data_.position[i]);
    }
  }
}

std::optional<Particle> Particle::divide(PetscInt new_gID) {
  using namespace utils::ArrayMath;
  if (data_.length < 2 * data_.l0) {
    return std::nullopt;
  }

  auto newCenterLeft = data_.position - (0.25 * data_.length) * utils::Quaternion::getDirectionVector(data_.quaternion);
  auto newCenterRight = data_.position + (0.25 * data_.length) * utils::Quaternion::getDirectionVector(data_.quaternion);

  // update self
  data_.position = newCenterLeft;
  data_.length = data_.l0;
  data_.age++;

  // return other
  auto newParticle = Particle(new_gID, newCenterRight, data_.quaternion, data_.l0, data_.l0, data_.diameter);
  newParticle.data_.age = data_.age;
  return newParticle;
}

void Particle::printState() const {
  fprintf(stdout, "Particle %d: pos=[%f, %f, %f], quat=[%f, %f, %f, %f], length=%f, diameter=%f \n",
          data_.gID,
          data_.position[0], data_.position[1], data_.position[2],
          data_.quaternion[0], data_.quaternion[1],
          data_.quaternion[2], data_.quaternion[3],
          data_.length, data_.diameter);
}

double Particle::getVolume() const {
  const double pi = std::acos(-1.0);
  double r = data_.diameter / 2.0;
  double cylinder_vol = pi * r * r * (data_.length - data_.diameter);
  double hemisphere_vol = (4.0 / 3.0) * pi * r * r * r;
  return cylinder_vol + hemisphere_vol;
}
