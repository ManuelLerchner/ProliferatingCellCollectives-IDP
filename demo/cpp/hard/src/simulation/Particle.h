#pragma once
#include <petsc.h>

#include <array>
#include <optional>
#include <vector>

// Forward declaration
struct ParticleData;

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

  // Convert Particle to ParticleData struct
  ParticleData toStruct() const;

  void updatePosition(const PetscScalar* dC, int offset, double dt);

  void updateQuaternion(const PetscScalar* dC, int offset, double dt);

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

  PetscInt getLocalID() const;

  void setLocalID(PetscInt localID);

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
  PetscInt localID = -1;
  std::array<double, POSITION_SIZE> position;
  std::array<double, QUATERNION_SIZE> quaternion;

  std::array<double, 3> force = {0.0, 0.0, 0.0};
  std::array<double, 3> torque = {0.0, 0.0, 0.0};
  double impedance = 0.0;

  double length;
  double l0;
  double diameter;
};
