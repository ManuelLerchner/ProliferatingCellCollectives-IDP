#include "ConstraintLogger.h"

#include "util/ArrayMath.h"
#include "util/Quaternion.h"

namespace vtk {

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
  std::vector<double> gamma(constraints.size());

  std::vector<std::array<double, 3>> normals(constraints.size());

  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  std::vector<int> ranks(constraints.size());

  int i = 0;
  for (const auto& constraint : constraints) {
    positions[i] = constraint.contactPoint;
    gidIs[i] = constraint.gidI;
    gidJs[i] = constraint.gidJ;
    gids[i] = constraint.gid;
    signed_distances[i] = constraint.signed_distance;
    normals[i] = constraint.normI;
    iteration[i] = constraint.iteration;
    gamma[i] = constraint.gamma;
    ranks[i] = rank;
    i++;
  }

  logger_.addPoints(positions);
  logger_.addPointData("gidI", gidIs);
  logger_.addPointData("gidJ", gidJs);
  logger_.addPointData("gids", gids);
  logger_.addPointData("signed_distance", signed_distances);
  logger_.addPointData("normals", normals);
  logger_.addPointData("iteration", iteration);
  logger_.addPointData("gamma", gamma);
  logger_.addPointData("rank", ranks);

  logger_.write();
}

}  // namespace vtk
