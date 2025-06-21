#pragma once

#include <array>
#include <functional>

class Constraint {
 public:
  // Constructor with local IDs and ownership
  Constraint(double delta0, bool violated, int gidI, int gidJ, int localI, int localJ,
             bool owned_by_me,
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
  // local index of particle I on this rank (-1 if not local)
  int localI;
  // local index of particle J on this rank (-1 if not local)
  int localJ;
  // whether particle I is owned by this rank
  bool owned_by_me;
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

// Custom hash for Constraint based on particle GIDs
struct ConstraintHash {
  std::size_t operator()(const Constraint& c) const {
    // A simple hash combination function
    std::size_t h1 = std::hash<int>()(c.gidI);
    std::size_t h2 = std::hash<int>()(c.gidJ);
    std::size_t h3 = std::hash<int>()(c.delta0);
    std::size_t h4 = std::hash<int>()(c.contactPoint[0]) ^ std::hash<int>()(c.contactPoint[1]) ^ std::hash<int>()(c.contactPoint[2]);
    std::size_t h5 = std::hash<int>()(c.normI[0]) ^ std::hash<int>()(c.normI[1]) ^ std::hash<int>()(c.normI[2]);
    return h1 ^ h2 ^ h3 ^ h4 ^ h5;
  }
};

// Custom equality for Constraint based on particle GIDs
struct ConstraintEqual {
  bool operator()(const Constraint& a, const Constraint& b) const {
    return a.gidI == b.gidI && a.gidJ == b.gidJ &&
           a.delta0 == b.delta0 &&
           a.contactPoint[0] == b.contactPoint[0] &&
           a.contactPoint[1] == b.contactPoint[1] &&
           a.contactPoint[2] == b.contactPoint[2] &&
           a.normI[0] == b.normI[0] &&
           a.normI[1] == b.normI[1] &&
           a.normI[2] == b.normI[2];
  }
};
