#pragma once

#include <array>

// Layout: [x,y,z, q0,q1,q2,q3, l] for each particle.
#define COMPONENTS_PER_PARTICLE 8

inline std::array<double, 3> get_position(const double* configuration_array, int local_particle_idx) {
  std::array<double, 3> position;
  int base_idx = local_particle_idx * COMPONENTS_PER_PARTICLE;
  position[0] = configuration_array[base_idx + 0];
  position[1] = configuration_array[base_idx + 1];
  position[2] = configuration_array[base_idx + 2];
  return position;
}

inline std::array<double, 4> get_quaternion(const double* configuration_array, int local_particle_idx) {
  std::array<double, 4> quaternion;
  int base_idx = local_particle_idx * COMPONENTS_PER_PARTICLE;
  quaternion[0] = configuration_array[base_idx + 3];
  quaternion[1] = configuration_array[base_idx + 4];
  quaternion[2] = configuration_array[base_idx + 5];
  quaternion[3] = configuration_array[base_idx + 6];
  return quaternion;
}

inline double get_length(const double* configuration_array, int local_particle_idx) {
  return configuration_array[local_particle_idx * COMPONENTS_PER_PARTICLE + 7];
}