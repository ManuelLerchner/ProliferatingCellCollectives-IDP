#pragma once

#include <array>

class Constraint {
 public:
  Constraint(double delta0, int gidI, int gidJ, std::array<double, 3> normI, std::array<double, 3> posI, std::array<double, 3> posJ, std::array<double, 3> labI, std::array<double, 3> labJ);

 public:
  // current overlap of the constraint
  double delta0;
  // unique global ID of particle I
  int gidI;
  // unique global ID of particle J
  int gidJ;
  // surface normal vector at the location of constraint (minimal separation) for particle I
  std::array<double, 3> normI;
  // relative constraint position on particle I
  std::array<double, 3> rPosI;
  // relative constraint position on particle J
  std::array<double, 3> rPosJ;
  // lab frame location of collision point on particle I
  std::array<double, 3> labI;
  // lab frame location of collision point on particle J
  std::array<double, 3> labJ;
};