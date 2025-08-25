#pragma once

#include <array>

namespace utils::Quaternion {

std::array<double, 3> rotateVectorOfPositions(const std::array<double, 4>& q, const std::array<double, 3>& v);

std::array<double, 3> getDirectionVector(const std::array<double, 4>& q);

std::array<double, 4> quaternionFromEuler(const std::array<double, 3>& euler);

std::array<double, 4> qmul(const std::array<double, 4>& q1, const std::array<double, 4>& q2);

}  // namespace utils::Quaternion