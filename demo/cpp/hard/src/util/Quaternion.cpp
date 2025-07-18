#include "Quaternion.h"

#include <cmath>

namespace utils::Quaternion {

std::array<double, 3> rotateVectorOfPositions(const std::array<double, 4>& q, const std::array<double, 3>& v) {
  // Extract quaternion components: q = [w, x, y, z]
  double w = q[0], x = q[1], y = q[2], z = q[3];

  // Extract vector components
  double vx = v[0], vy = v[1], vz = v[2];

  // Quaternion rotation formula: v' = q * v * q^(-1)
  // For unit quaternions, this can be computed as:
  // v' = v + 2 * w * (q_vec × v) + 2 * (q_vec × (q_vec × v))
  // where q_vec = [x, y, z] is the vector part of the quaternion

  // First cross product: q_vec × v
  double cross1_x = y * vz - z * vy;
  double cross1_y = z * vx - x * vz;
  double cross1_z = x * vy - y * vx;

  // Second cross product: q_vec × (q_vec × v)
  double cross2_x = y * cross1_z - z * cross1_y;
  double cross2_y = z * cross1_x - x * cross1_z;
  double cross2_z = x * cross1_y - y * cross1_x;

  // Final rotation: v' = v + 2*w*(q_vec × v) + 2*(q_vec × (q_vec × v))
  return {
      vx + 2.0 * w * cross1_x + 2.0 * cross2_x,
      vy + 2.0 * w * cross1_y + 2.0 * cross2_y,
      vz + 2.0 * w * cross1_z + 2.0 * cross2_z};
}

std::array<double, 3> getDirectionVector(const std::array<double, 4>& q) {
  return rotateVectorOfPositions(q, {1.0, 0.0, 0.0});
}

std::array<double, 4> quaternionFromEuler(const std::array<double, 3>& euler) {
  double roll = euler[0];
  double pitch = euler[1];
  double yaw = euler[2];

  double cy = std::cos(yaw * 0.5);
  double sy = std::sin(yaw * 0.5);
  double cp = std::cos(pitch * 0.5);
  double sp = std::sin(pitch * 0.5);
  double cr = std::cos(roll * 0.5);
  double sr = std::sin(roll * 0.5);

  return {
      cr * cp * cy + sr * sp * sy,
      sr * cp * cy - cr * sp * sy,
      cr * sp * cy + sr * cp * sy,
      cr * cp * sy - sr * sp * cy};
}

std::array<double, 4> qmul(const std::array<double, 4>& q1, const std::array<double, 4>& q2) {
  return {q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3],
          q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2],
          q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1],
          q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]};
}

}  // namespace utils::Quaternion