#include "Particle.h"

#include <petsc.h>

#include <cmath>

Particle::Particle(PetscInt id, const std::array<double, POSITION_SIZE>& position, const std::array<double, QUATERNION_SIZE>& quaternion, double length, double diameter) : id(id), position(position), quaternion(quaternion), length(length), diameter(diameter) {}

void Particle::updatePosition(const PetscScalar* data, int offset, double dt) {
  position[0] += dt * PetscRealPart(data[offset + 0]);
  position[1] += dt * PetscRealPart(data[offset + 1]);
  position[2] += dt * PetscRealPart(data[offset + 2]);
}

void Particle::updateQuaternion(const PetscScalar* data, int offset, double dt) {
  quaternion[0] += dt * PetscRealPart(data[offset + 0]);
  quaternion[1] += dt * PetscRealPart(data[offset + 1]);
  quaternion[2] += dt * PetscRealPart(data[offset + 2]);
  quaternion[3] += dt * PetscRealPart(data[offset + 3]);
}

void Particle::updateState(const PetscScalar* data, int particle_index, double dt) {
  int base_offset = particle_index * STATE_SIZE;
  updatePosition(data, base_offset, dt);
  updateQuaternion(data, base_offset + POSITION_SIZE, dt);
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
  }
}

void Particle::printState() const {
  fprintf(stdout, "Particle %d: pos=[%f, %f, %f], quat=[%f, %f, %f, %f]\n",
          id,
          position[0], position[1], position[2],
          quaternion[0], quaternion[1],
          quaternion[2], quaternion[3]);
}

PetscInt Particle::getId() const {
  return id;
}

void Particle::setId(PetscInt id) {
  this->id = id;
}

const std::array<double, POSITION_SIZE>& Particle::getPosition() const {
  return position;
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

 