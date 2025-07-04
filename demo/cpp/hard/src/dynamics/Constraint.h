#pragma once

#include <array>
#include <functional>

class Constraint {
 public:
  // Constructor with local IDs and ownership
  Constraint(double delta0, bool violated, int gidI, int gidJ,
             std::array<double, 3> normI, std::array<double, 3> posI, std::array<double, 3> posJ, std::array<double, 3> contactPoint,
             double stressI, double stressJ,
             int constraint_iterations, int gid);

  Constraint();

 public:
  // current overlap of the constraint
  double delta0;
  // whether the constraint is violated
  bool violated;
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
  // contact point on particle I
  std::array<double, 3> contactPoint;
  // stress on particle I
  double stressI;
  // stress on particle J
  double stressJ;
  // number of constraint iterations since the constraint was created
  int constraint_iterations;
  // global index of the constraint
  int gid;

  void print() const;
};
