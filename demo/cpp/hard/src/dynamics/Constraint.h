#pragma once

#include <array>

class Constraint {
 public:
  Constraint(double delta0, int gidI, int gidJ, std::array<double, 3> normI, std::array<double, 3> posI, std::array<double, 3> posJ, std::array<double, 3> labI, std::array<double, 3> labJ);

  // Constructor with local IDs
  Constraint(double delta0, int gidI, int gidJ, int localI, int localJ, std::array<double, 3> normI, std::array<double, 3> posI, std::array<double, 3> posJ, std::array<double, 3> labI, std::array<double, 3> labJ);

  // Constructor with local IDs and ownership
  Constraint(double delta0, int gidI, int gidJ, int localI, int localJ, bool owns, std::array<double, 3> normI, std::array<double, 3> posI, std::array<double, 3> posJ, std::array<double, 3> labI, std::array<double, 3> labJ);

 public:
  // current overlap of the constraint
  double delta0;
  // unique global ID of particle I
  int gidI;
  // unique global ID of particle J
  int gidJ;
  // local index of particle I on this rank (-1 if not local)
  int localI;
  // local index of particle J on this rank (-1 if not local)
  int localJ;
  // whether this rank owns this constraint (for constraint counting/creation)
  bool ownedByUs;
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

  void print() const;
};