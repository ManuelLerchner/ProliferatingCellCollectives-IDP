#include "ParticleLogger.h"

#include "util/ArrayMath.h"
#include "util/Quaternion.h"

namespace vtk {

ParticleLogger::ParticleLogger(const std::string& outputDirectory, const std::string& baseFilename, bool preserve_existing, size_t step)
    : logger_(outputDirectory, baseFilename, false, preserve_existing, step) {
}

void ParticleLogger::log(const std::vector<Particle>& particles) {
  std::vector<std::array<double, 3>> positions(particles.size());

  std::vector<std::array<double, 4>> quaternions(particles.size());
  std::vector<std::array<double, 3>> orientation(particles.size());
  std::vector<double> orientation_angle(particles.size());
  std::vector<std::array<double, 3>> lengths(particles.size());
  std::vector<std::array<double, 3>> forces(particles.size());

  std::vector<int> gids(particles.size());
  std::vector<double> impedances(particles.size());
  std::vector<double> ldots(particles.size());
  std::vector<double> stresses(particles.size());
  std::vector<int> number_of_constraints(particles.size());
  std::vector<int> ages(particles.size());
  std::vector<std::array<double, 3>> velocity_linear(particles.size());
  std::vector<std::array<double, 3>> velocity_angular(particles.size());
  std::vector<int> ranks(particles.size());

  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  for (int i = 0; i < particles.size(); i++) {
    positions[i] = particles[i].getPosition();
    quaternions[i] = particles[i].getQuaternion();
    orientation[i] = utils::Quaternion::getDirectionVector(quaternions[i]);

    orientation_angle[i] = std::atan2(orientation[i][1], orientation[i][0]);
    if (orientation_angle[i] < 0) {
      orientation_angle[i] += M_PI;
    }

    lengths[i] = {particles[i].getLength(), particles[i].getDiameter(), particles[i].getDiameter()};

    gids[i] = particles[i].getGID();
    impedances[i] = particles[i].getImpedance();
    ldots[i] = particles[i].getLdot();
    stresses[i] = particles[i].getStress();
    number_of_constraints[i] = particles[i].getNumConstraints();
    ages[i] = particles[i].getAge();
    forces[i] = particles[i].getForce();
    velocity_linear[i] = particles[i].getVelocityLinear();
    velocity_angular[i] = particles[i].getVelocityAngular();
    ranks[i] = rank;
  }

  // Add data to logger
  logger_.addPoints(positions);

  logger_.addPointData("gid", gids);
  logger_.addPointData("quaternion", quaternions);
  logger_.addPointData("orientation", orientation);
  logger_.addPointData("orientation_angle", orientation_angle);
  logger_.addPointData("lengths", lengths);
  logger_.addPointData("impedance", impedances);
  logger_.addPointData("ldot", ldots);
  logger_.addPointData("stress", stresses);
  logger_.addPointData("number_of_constraints", number_of_constraints);
  logger_.addPointData("age", ages);
  logger_.addPointData("forces", forces);
  logger_.addPointData("velocity_linear", velocity_linear);
  logger_.addPointData("velocity_angular", velocity_angular);
  logger_.addPointData("rank", ranks);

  // Write to file
  logger_.write();
}

}  // namespace vtk
