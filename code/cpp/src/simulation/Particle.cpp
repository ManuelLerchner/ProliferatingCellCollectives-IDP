#include "Particle.h"

#include <petsc.h>

#include <cmath>
#include <iostream>
#include <random>

#include "ParticleData.h"
#include "util/ArrayMath.h"
#include "util/Quaternion.h"

Particle::Particle(PetscInt gID, const std::array<double, POSITION_SIZE>& position,
                   const std::array<double, QUATERNION_SIZE>& quaternion, double length, double l0, double diameter) {
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

  // Validate after position update
  double move_maginitude = std::sqrt(dt * dt * (PetscRealPart(dC[offset + 0]) * PetscRealPart(dC[offset + 0]) + PetscRealPart(dC[offset + 1]) * PetscRealPart(dC[offset + 1]) + PetscRealPart(dC[offset + 2]) * PetscRealPart(dC[offset + 2])));

  //  particlescan only move 5% of their length in a single step
  if (move_maginitude > 0.05 * data_.l0) {
    PetscPrintf(PETSC_COMM_WORLD, "WARNING: Particle %d moved an unusually large distance: %g (length=%g)\n", data_.gID, move_maginitude, data_.length);
  }
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

void Particle::setForce(const PetscScalar* df, int offset) {
  force_[0] = PetscRealPart(df[offset + 0]);
  force_[1] = PetscRealPart(df[offset + 1]);
  force_[2] = PetscRealPart(df[offset + 2]);
}

void Particle::setTorque(const PetscScalar* df, int offset) {
  torque_[0] = PetscRealPart(df[offset + 0]);
  torque_[1] = PetscRealPart(df[offset + 1]);
  torque_[2] = PetscRealPart(df[offset + 2]);
}

void Particle::setVelocityLinear(const PetscScalar* dU) {
  velocityLinear_[0] = PetscRealPart(dU[0]);
  velocityLinear_[1] = PetscRealPart(dU[1]);
  velocityLinear_[2] = PetscRealPart(dU[2]);
}

void Particle::setVelocityAngular(const PetscScalar* dU) {
  velocityAngular_[0] = PetscRealPart(dU[0]);
  velocityAngular_[1] = PetscRealPart(dU[1]);
  velocityAngular_[2] = PetscRealPart(dU[2]);
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

void Particle::setForceAndTorque(const PetscScalar* f) {
  setForce(f, 0);
  setTorque(f, 3);

  // Final validation after complete state update
  validateAndWarn();
}

void Particle::setVelocity(const PetscScalar* dU) {
  setVelocityLinear(dU);
  setVelocityAngular(dU + 3);
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

  bool has_error = false;

  for (int i = 0; i < POSITION_SIZE; ++i) {
    if (!std::isfinite(data_.position[i])) {
      PetscPrintf(PETSC_COMM_WORLD, "WARNING: Particle %d position[%d] = %g (non-finite)\n", data_.gID, i,
                  data_.position[i]);
      has_error = true;
    }
  }

  // Check quaternion for NaN/infinity
  for (int i = 0; i < QUATERNION_SIZE; ++i) {
    if (!std::isfinite(data_.quaternion[i])) {
      PetscPrintf(PETSC_COMM_WORLD, "WARNING: Particle %d quaternion[%d] = %g (non-finite)\n", data_.gID, i,
                  data_.quaternion[i]);
      has_error = true;
    }
  }

  // Check physical properties
  if (!std::isfinite(data_.length)) {
    PetscPrintf(PETSC_COMM_WORLD, "WARNING: Particle %d length = %g (non-finite)\n", data_.gID, data_.length);
    has_error = true;
  }

  if (!std::isfinite(data_.diameter)) {
    PetscPrintf(PETSC_COMM_WORLD, "WARNING: Particle %d diameter = %g (non-finite)\n", data_.gID, data_.diameter);
    has_error = true;
  }

  // Check for extremely large values that might indicate numerical issues
  for (int i = 0; i < POSITION_SIZE; ++i) {
    if (std::abs(data_.position[i]) > 1e6) {
      PetscPrintf(PETSC_COMM_WORLD, "WARNING: Particle %d position[%d] = %g (extremely large)\n", data_.gID, i,
                  data_.position[i]);
      has_error = true;
    }
  }

  if (has_error) {
    printState();
    throw std::runtime_error("Particle state validation failed.");
  }
}

std::optional<Particle> Particle::divide(PetscInt new_gID) {
  using namespace utils::ArrayMath;
  if (data_.length < 2 * data_.l0) {
    return std::nullopt;
  }

  auto l0 = data_.length / 2;

  //   random number between 0.7 and 1.3
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(l0 * 0.98,l0 * 1.02);

  double l1 = dis(gen);
  double l2 = data_.length - l1;

  auto newCenterLeft = data_.position - (l1 / 2) * utils::Quaternion::getDirectionVector(data_.quaternion);
  auto newCenterRight = data_.position + (l2 / 2) * utils::Quaternion::getDirectionVector(data_.quaternion);

  // update self
  data_.position = newCenterLeft;
  data_.length = l1;
  data_.age++;

  // return other
  auto newParticle = Particle(new_gID, newCenterRight, data_.quaternion, l2, data_.l0, data_.diameter);
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

std::array<double, 3> Particle::calculateGravitationalForce(const std::array<double, 3>& gravity) const {
  double vol = getVolume();
  return {vol * gravity[0], vol * gravity[1], vol * gravity[2]};
}

std::array<double, 6> Particle::calculateBrownianVelocity(double temperature, bool monolayer, double xi, double dt,
                                                          std::mt19937& gen) const {
  std::normal_distribution<double> dist(0.0, 1.0);

  double mobility = dist(gen);

  // Brownian velocities
  const double rot_coeff = temperature * mobility;

  return {
      0,
      0,
      0,
      monolayer ? 0.0 : rot_coeff * dist(gen) * impedance_,
      monolayer ? 0.0 : rot_coeff * dist(gen) * impedance_,
      rot_coeff * dist(gen) * impedance_};
}
