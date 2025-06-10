#pragma once

#include <array>

class Constraint {
 public:
  // Constructor with local IDs and ownership
  Constraint(double delta0, bool violated, int gidI, int gidJ, int localI, int localJ,
             bool particleI_isLocal, bool particleJ_isLocal,
             std::array<double, 3> normI, std::array<double, 3> posI, std::array<double, 3> posJ, std::array<double, 3> contactPoint);

 public:
  // current overlap of the constraint
  double delta0;
  // whether the constraint is violated
  bool violated;
  // unique global ID of particle I
  int gidI;
  // unique global ID of particle J
  int gidJ;
  // local index of particle I on this rank (-1 if not local)
  int localI;
  // local index of particle J on this rank (-1 if not local)
  int localJ;
  // whether particle I is owned by this rank
  bool particleI_isLocal;
  // whether particle J is owned by this rank
  bool particleJ_isLocal;
  // surface normal vector at the location of constraint (minimal separation) for particle I
  std::array<double, 3> normI;
  // relative constraint position on particle I
  std::array<double, 3> rPosI;
  // relative constraint position on particle J
  std::array<double, 3> rPosJ;
  // contact point on particle I
  std::array<double, 3> contactPoint;

  void print() const;
};