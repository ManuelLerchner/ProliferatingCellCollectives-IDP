#pragma once

#include <array>
#include <functional>

class Constraint {
 public:
  // Constructor with local IDs and ownership
  Constraint(double signed_distance, int gidI, int gidJ,
             std::array<double, 3> normI, std::array<double, 3> posI, std::array<double, 3> posJ, std::array<double, 3> contactPoint,
             double stressI, double stressJ, int gid, int iteration, bool localI, bool localJ);

  Constraint();

 public:
  // current signed_distance of the constraint
  double signed_distance;
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
  // global index of the constraint
  int gid;
  // iteration of the constraint
  int iteration;
  // Lagrange multiplier for this constraint
  double gamma = 0.0;

  bool localI;
  bool localJ;

  void print() const;
};
