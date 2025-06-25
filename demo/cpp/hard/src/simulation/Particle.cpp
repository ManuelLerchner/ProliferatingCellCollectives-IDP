#include "Particle.h"

#include <petsc.h>

#include <cmath>
#include <iostream>
#include <random>

#include "ParticleData.h"
#include "util/ArrayMath.h"
#include "util/ParticleMPI.h"
#include "util/Quaternion.h"

Particle::Particle(PetscInt gID, const std::array<double, POSITION_SIZE>& position, const std::array<double, QUATERNION_SIZE>& quaternion, double length, double l0, double diameter) : gID(gID), position(position), quaternion(quaternion), length(length), l0(l0), diameter(diameter) {}

Particle::Particle(const ParticleData& data)
    : gID(data.gID),
      position(data.position),
      quaternion(data.quaternion),
      force(data.force),
      torque(data.torque),
      velocityLinear(data.velocityLinear),
      velocityAngular(data.velocityAngular),
      impedance(data.impedance),
      length(data.length),
      l0(data.l0),
      diameter(data.diameter) {}

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

void Particle::updateLength(double ldot, double dt) {
  length += dt * ldot;

  if (length > 2 * l0) {
    length = 2 * l0;
  }
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

void Particle::addVelocityLinear(const PetscScalar* dU) {
  velocityLinear[0] += PetscRealPart(dU[0]);
  velocityLinear[1] += PetscRealPart(dU[1]);
  velocityLinear[2] += PetscRealPart(dU[2]);
}

void Particle::addVelocityAngular(const PetscScalar* dU) {
  velocityAngular[0] += PetscRealPart(dU[0]);
  velocityAngular[1] += PetscRealPart(dU[1]);
  velocityAngular[2] += PetscRealPart(dU[2]);
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
  force[0] = 0.0;
  force[1] = 0.0;
  force[2] = 0.0;
  torque[0] = 0.0;
  torque[1] = 0.0;
  torque[2] = 0.0;
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
  double norm = std::sqrt(quaternion[0] * quaternion[0] +
                          quaternion[1] * quaternion[1] +
                          quaternion[2] * quaternion[2] +
                          quaternion[3] * quaternion[3]);

  quaternion[0] /= norm;
  quaternion[1] /= norm;
  quaternion[2] /= norm;
  quaternion[3] /= norm;

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

std::optional<Particle> Particle::divide() {
  using namespace utils::ArrayMath;
  if (length < 2 * l0) {
    return std::nullopt;
  }

  auto newCenterLeft = position - (0.25 * length) * utils::Quaternion::getDirectionVector(quaternion);
  auto newCenterRight = position + (0.25 * length) * utils::Quaternion::getDirectionVector(quaternion);

  // update self
  position = newCenterLeft;
  length = l0;

  // return other
  return Particle(-1, newCenterRight, quaternion, l0, l0, diameter);
}

void Particle::printState() const {
  fprintf(stdout, "Particle %d: pos=[%f, %f, %f], quat=[%f, %f, %f, %f], length=%f, diameter=%f \n",
          gID,
          position[0], position[1], position[2],
          quaternion[0], quaternion[1],
          quaternion[2], quaternion[3],
          length, diameter);
}

PetscInt Particle::setGID() const {
  return gID;
}

void Particle::setGID(PetscInt gID) {
  this->gID = gID;
}

PetscInt Particle::getGID() const {
  return gID;
}

PetscInt Particle::getLocalID() const {
  return localID;
}

void Particle::setLocalID(PetscInt localID) {
  this->localID = localID;
}

const std::array<double, Particle::POSITION_SIZE>& Particle::getPosition() const {
  return position;
}

double Particle::getImpedance() const {
  return impedance;
}

void Particle::setImpedance(double impedance) {
  this->impedance = impedance;
}

bool Particle::getToDelete() const {
  return toDelete;
}

void Particle::setToDelete(bool toDelete) {
  this->toDelete = toDelete;
}

const std::array<double, 3>& Particle::getForce() const {
  return force;
}

const std::array<double, 3>& Particle::getTorque() const {
  return torque;
}

const std::array<double, 3>& Particle::getVelocityLinear() const {
  return velocityLinear;
}

const std::array<double, 3>& Particle::getVelocityAngular() const {
  return velocityAngular;
}

double Particle::getLength() const {
  return length;
}

double Particle::getDiameter() const {
  return diameter;
}

double Particle::getVolume() const {
  const double r = diameter / 2.0;
  const double cylinder_vol = M_PI * r * r * length;
  const double cap_vol = (4.0 / 3.0) * M_PI * r * r * r;
  return cylinder_vol + cap_vol;
}

const std::array<double, Particle::QUATERNION_SIZE>& Particle::getQuaternion() const {
  return quaternion;
}

ParticleData Particle::getData() const {
  ParticleData data;
  data.gID = gID;
  data.position = position;
  data.quaternion = quaternion;
  data.length = length;
  data.l0 = l0;
  data.diameter = diameter;
  data.impedance = impedance;
  return data;
}
