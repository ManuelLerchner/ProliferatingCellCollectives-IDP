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

  std::vector<int> gids(particles.size());
  std::vector<double> impedances(particles.size());
  std::vector<int> number_of_constraints(particles.size());
  std::vector<int> ages(particles.size());

  for (int i = 0; i < particles.size(); i++) {
    positions[i] = particles[i].getPosition();
    orientation[i] = utils::Quaternion::getDirectionVector(particles[i].getQuaternion());
    lengths[i] = {particles[i].getLength(), particles[i].getDiameter(), particles[i].getDiameter()};

    gids[i] = particles[i].getGID();
    impedances[i] = particles[i].getImpedance();
    number_of_constraints[i] = particles[i].getNumConstraints();
    ages[i] = particles[i].getAge();
  }

  // Add data to logger
  logger_.addPoints(positions);

  logger_.addPointData("gid", gids);
  logger_.addPointData("orientation", orientation);
  logger_.addPointData("lengths", lengths);
  logger_.addPointData("impedance", impedances);
  logger_.addPointData("number_of_constraints", number_of_constraints);
  logger_.addPointData("age", ages);

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
  std::vector<double> delta0s(constraints.size());
  std::vector<double> iteration(constraints.size());

  std::vector<std::array<double, 3>> normals(constraints.size());

  for (int i = 0; i < constraints.size(); i++) {
    positions[i] = constraints[i].contactPoint;
    gidIs[i] = constraints[i].gidI;
    gidJs[i] = constraints[i].gidJ;
    gids[i] = constraints[i].gid;
    delta0s[i] = constraints[i].delta0;
    normals[i] = constraints[i].normI;
    iteration[i] = constraints[i].iteration;
  }

  logger_.addPoints(positions);
  logger_.addPointData("gidI", gidIs);
  logger_.addPointData("gidJ", gidJs);
  logger_.addPointData("gids", gids);
  logger_.addPointData("delta0", delta0s);
  logger_.addPointData("normals", normals);
  logger_.addPointData("iteration", iteration);

  logger_.write();
}

}  // namespace vtk
