#include "DomainLogger.h"

#include "util/ArrayMath.h"
#include "util/Quaternion.h"

namespace vtk {

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
