#pragma once
#include <petsc.h>

#include <array>
#include <optional>

// Constants for vector operations
static constexpr int POSITION_SIZE = 3;
static constexpr int QUATERNION_SIZE = 4;
static constexpr int STATE_SIZE = POSITION_SIZE + QUATERNION_SIZE;

class Particle {
 public:
  Particle(PetscInt gID, const std::array<double, POSITION_SIZE>& position, const std::array<double, QUATERNION_SIZE>& quaternion, double length, double l0, double diameter);

  void updatePosition(const PetscScalar* data, int offset, double dt);

  void updateQuaternion(const PetscScalar* data, int offset, double dt);

  void updateLength(const PetscScalar* dL, int particle_index, double dt);

  void addForce(const PetscScalar* df, int offset);

  void addTorque(const PetscScalar* df, int offset);

  void eulerStepPosition(const PetscScalar* dC, int particle_index, double dt);

  void eulerStepLength(const PetscScalar* dL, int particle_index, double dt);

  void clearForceAndTorque();

  void addForceAndTorque(const PetscScalar* f, const PetscScalar* U, int particle_index);

  void printState() const;

  std::optional<Particle> divide();

  // Validation methods
  bool isValid() const;
  void validateAndWarn() const;

  const std::array<double, POSITION_SIZE>& getPosition() const;
  const std::array<double, QUATERNION_SIZE>& getQuaternion() const;

  double getLength() const;

  double getDiameter() const;

  PetscInt setGID() const;

  void setGID(PetscInt gID);

  PetscInt getGID() const;

  double getImpedance() const;

  void setImpedance(double impedance);

  const std::array<double, 3>& getForce() const;
  const std::array<double, 3>& getTorque() const;

  static constexpr int getStateSize() {
    return STATE_SIZE;
  }

 private:
  void normalizeQuaternion();

  PetscInt gID;
  std::array<double, POSITION_SIZE> position;
  std::array<double, QUATERNION_SIZE> quaternion;

  std::array<double, 3> force;
  std::array<double, 3> torque;
  double impedance;

  double length;
  double l0;
  double diameter;
};
