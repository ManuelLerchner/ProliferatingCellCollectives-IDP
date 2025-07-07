#include "ParticleLogger.h"

#include "util/ArrayMath.h"
#include "util/Quaternion.h"

namespace vtk {

ParticleLogger::ParticleLogger(const std::string& outputDirectory, const std::string& baseFilename)
    : logger_(outputDirectory, baseFilename) {
}

void ParticleLogger::log(const std::vector<Particle>& particles) {
  std::vector<std::array<double, 3>> positions(particles.size());

  std::vector<std::array<double, 3>> orientation(particles.size());
  std::vector<std::array<double, 3>> lengths(particles.size());
  std::vector<std::array<double, 3>> forces(particles.size());

  std::vector<int> gids(particles.size());
  std::vector<double> impedances(particles.size());
  std::vector<int> number_of_constraints(particles.size());
  std::vector<int> ages(particles.size());
  std::vector<std::array<double, 3>> velocity_linear(particles.size());

  for (int i = 0; i < particles.size(); i++) {
    positions[i] = particles[i].getPosition();
    orientation[i] = utils::Quaternion::getDirectionVector(particles[i].getQuaternion());
    lengths[i] = {particles[i].getLength(), particles[i].getDiameter(), particles[i].getDiameter()};

    gids[i] = particles[i].getGID();
    impedances[i] = particles[i].getImpedance();
    number_of_constraints[i] = particles[i].getNumConstraints();
    ages[i] = particles[i].getAge();
    forces[i] = particles[i].getForce();
    velocity_linear[i] = particles[i].getVelocityLinear();
  }

  // Add data to logger
  logger_.addPoints(positions);

  logger_.addPointData("gid", gids);
  logger_.addPointData("orientation", orientation);
  logger_.addPointData("lengths", lengths);
  logger_.addPointData("impedance", impedances);
  logger_.addPointData("number_of_constraints", number_of_constraints);
  logger_.addPointData("age", ages);
  logger_.addPointData("forces", forces);
  logger_.addPointData("velocity_linear", velocity_linear);

  // Write to file
  logger_.write();
}

ConstraintLogger::ConstraintLogger(const std::string& outputDirectory, const std::string& baseFilename)
    : logger_(outputDirectory, baseFilename) {
}

void ConstraintLogger::log(const std::vector<Constraint>& constraints) {
  std::vector<std::array<double, 3>> positions(constraints.size());

  std::vector<int> gidIs(constraints.size());
  std::vector<int> gidJs(constraints.size());
  std::vector<int> gids(constraints.size());
  std::vector<double> signed_distances(constraints.size());
  std::vector<double> iteration(constraints.size());

  std::vector<std::array<double, 3>> normals(constraints.size());

  int i = 0;
  for (const auto& constraint : constraints) {
    positions[i] = constraint.contactPoint;
    gidIs[i] = constraint.gidI;
    gidJs[i] = constraint.gidJ;
    gids[i] = constraint.gid;
    signed_distances[i] = constraint.signed_distance;
    normals[i] = constraint.normI;
    iteration[i] = constraint.iteration;
    i++;
  }

  logger_.addPoints(positions);
  logger_.addPointData("gidI", gidIs);
  logger_.addPointData("gidJ", gidJs);
  logger_.addPointData("gids", gids);
  logger_.addPointData("signed_distance", signed_distances);
  logger_.addPointData("normals", normals);
  logger_.addPointData("iteration", iteration);

  logger_.write();
}

DomainLogger::DomainLogger(const std::string& outputDirectory, const std::string& baseFilename)
    : logger_(outputDirectory, baseFilename) {
}

void DomainLogger::log(const std::pair<std::array<double, 3>, std::array<double, 3>>& domain) {
  using namespace utils::ArrayMath;

  auto [min, max] = domain;

  auto center = 0.5 * (min + max);
  auto scale = (max - min);

  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  std::vector<std::array<double, 3>> positions{center};
  std::vector<std::array<double, 3>> scales{scale};
  std::vector<int> ranks{rank};

  logger_.addPoints(positions);
  logger_.addPointData("scale", scales);
  logger_.addPointData("rank", ranks);

  logger_.write();
}

}  // namespace vtk
