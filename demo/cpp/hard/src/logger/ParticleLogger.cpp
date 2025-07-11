#include "ParticleLogger.h"

#include "util/ArrayMath.h"
#include "util/Quaternion.h"

namespace vtk {

ParticleLogger::ParticleLogger(const std::string& outputDirectory, const std::string& baseFilename, bool preserve_existing, size_t step)
    : logger_(outputDirectory, baseFilename, false, preserve_existing, step) {
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
  std::vector<int> ranks(particles.size());

  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

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
    ranks[i] = rank;
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
  logger_.addPointData("rank", ranks);

  // Write to file
  logger_.write();
}

}  // namespace vtk
