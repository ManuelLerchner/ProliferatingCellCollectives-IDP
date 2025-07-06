#pragma once
#include <petsc.h>

#include <array>
#include <optional>
#include <random>
#include <vector>

#include "ParticleData.h"

class Particle {
 public:
  // Constants for vector operations
  static constexpr int POSITION_SIZE = 3;
  static constexpr int QUATERNION_SIZE = 4;
  static constexpr int STATE_SIZE = POSITION_SIZE + QUATERNION_SIZE;

  // Constructor
  Particle(PetscInt gID, const std::array<double, POSITION_SIZE>& position, const std::array<double, QUATERNION_SIZE>& quaternion, double length, double l0, double diameter);

  // Constructor from ParticleData struct
  explicit Particle(const ParticleData& data);

  void updatePosition(const PetscScalar* dC, int offset, double dt);

  void updateQuaternion(const PetscScalar* dC, int offset, double dt);

  void updateLength(double ldot, double dt);

  void addForce(const PetscScalar* df, int offset);

  void addTorque(const PetscScalar* df, int offset);

  void addVelocityLinear(const PetscScalar* dU);

  void addVelocityAngular(const PetscScalar* dU);

  void eulerStepPosition(const double* dC, double dt);

  void eulerStepLength(double ldot, double dt);

  void reset();

  void addForceAndTorque(const PetscScalar* f);

  void addVelocity(const PetscScalar* dU);

  void printState() const;

  std::optional<Particle> divide(PetscInt new_gID);

  // Validation methods
  bool isValid() const;
  void validateAndWarn() const;

  const std::array<double, POSITION_SIZE>& getPosition() const {
    return data_.position;
  }
  const std::array<double, QUATERNION_SIZE>& getQuaternion() const {
    return data_.quaternion;
  }

  double getLength() const {
    return data_.length;
  }

  double getDiameter() const {
    return data_.diameter;
  }

  void setGID(PetscInt gID) {
    data_.gID = gID;
  }

  PetscInt getGID() const {
    return data_.gID;
  }

  double getVolume() const;

  double getImpedance() const {
    return impedance_;
  }

  void setImpedance(double impedance) {
    impedance_ = impedance;
  }

  bool getToDelete() const {
    return toDelete;
  }
  void setToDelete(bool toDelete) {
    this->toDelete = toDelete;
  }

  const std::array<double, 3>& getForce() const {
    return force_;
  }
  const std::array<double, 3>& getTorque() const {
    return torque_;
  }
  const std::array<double, 3>& getVelocityLinear() const {
    return velocityLinear_;
  }
  const std::array<double, 3>& getVelocityAngular() const {
    return velocityAngular_;
  }

  static constexpr int getStateSize() {
    return STATE_SIZE;
  }

  ParticleData& getData() {
    return data_;
  }

  const ParticleData& getData() const {
    return data_;
  }

  int getAge() const {
    return data_.age;
  }

  void incrementAge() {
    data_.age++;
  }

  int getNumConstraints() const {
    return num_constraints_;
  }

  void incrementNumConstraints() {
    num_constraints_++;
  }

  std::array<double, 3> calculateGravitationalForce(const std::array<double, 3>& gravity) const;
  std::array<double, 6> calculateBrownianVelocity(double temperature, double xi, double dt, std::normal_distribution<double>& dist, std::mt19937& gen) const;

 private:
  void normalizeQuaternion();
  ParticleData data_;

  bool toDelete = false;

  // Fields not needed for MPI communication
  std::array<double, 3> force_;
  std::array<double, 3> torque_;
  std::array<double, 3> velocityLinear_;
  std::array<double, 3> velocityAngular_;
  double impedance_;
  int num_constraints_ = 0;
};
