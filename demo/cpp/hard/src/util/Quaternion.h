#pragma once

#include <array>

namespace utils::Quaternion {

std::array<double, 3> rotateVectorOfPositions(const std::array<double, 4>& q, const std::array<double, 3>& v);

std::array<double, 3> getDirectionVector(const std::array<double, 4>& q);

}  // namespace utils::Quaternion