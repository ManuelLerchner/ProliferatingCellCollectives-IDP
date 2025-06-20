#include "Constraint.h"

#include <iostream>

Constraint::Constraint(double delta0, bool violated, int gidI, int gidJ, int localI, int localJ, bool particleI_isLocal, bool particleJ_isLocal, std::array<double, 3> normI, std::array<double, 3> posI, std::array<double, 3> posJ, std::array<double, 3> contactPoint, std::array<double, 3> orientationI, std::array<double, 3> orientationJ, int constraint_iterations, int gid)
    : delta0(delta0), violated(violated), gidI(gidI), gidJ(gidJ), localI(localI), localJ(localJ), particleI_isLocal(particleI_isLocal), particleJ_isLocal(particleJ_isLocal), normI(normI), rPosI(posI), rPosJ(posJ), contactPoint(contactPoint), orientationI(orientationI), orientationJ(orientationJ), constraint_iterations(constraint_iterations), gid(gid) {}

void Constraint::print() const {
  std::cout << "Constraint: " << delta0 << std::endl;
  std::cout << "violated: " << violated << std::endl;
  std::cout << "gidI: " << gidI << std::endl;
  std::cout << "gidJ: " << gidJ << std::endl;
  std::cout << "localI: " << localI << std::endl;
  std::cout << "localJ: " << localJ << std::endl;
  std::cout << "particleI_isLocal: " << particleI_isLocal << std::endl;
  std::cout << "particleJ_isLocal: " << particleJ_isLocal << std::endl;
  std::cout << "normI: " << normI[0] << ", " << normI[1] << ", " << normI[2] << std::endl;
  std::cout << "rPosI: " << rPosI[0] << ", " << rPosI[1] << ", " << rPosI[2] << std::endl;
  std::cout << "rPosJ: " << rPosJ[0] << ", " << rPosJ[1] << ", " << rPosJ[2] << std::endl;
  std::cout << "contactPoint: " << contactPoint[0] << ", " << contactPoint[1] << ", " << contactPoint[2] << std::endl;
  std::cout << "orientationI: " << orientationI[0] << ", " << orientationI[1] << ", " << orientationI[2] << std::endl;
  std::cout << "orientationJ: " << orientationJ[0] << ", " << orientationJ[1] << ", " << orientationJ[2] << std::endl;
  std::cout << "constraint_iterations: " << constraint_iterations << std::endl;
  std::cout << "gid: " << gid << std::endl;
}