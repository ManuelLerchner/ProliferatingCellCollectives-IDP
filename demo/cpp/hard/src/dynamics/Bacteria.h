#pragma once
#include <array>

struct Bacterium {
  std::array<double, 3> position;
  std::array<double, 4> quaternion;
  double length;
  double diameter;
};