#include "Constraint.h"

#include <iostream>

Constraint::Constraint(double delta0, int gidI, int gidJ, int localI, int localJ, bool particleI_isLocal, bool particleJ_isLocal, std::array<double, 3> normI, std::array<double, 3> posI, std::array<double, 3> posJ, std::array<double, 3> contactPoint)
    : delta0(delta0), gidI(gidI), gidJ(gidJ), localI(localI), localJ(localJ), particleI_isLocal(particleI_isLocal), particleJ_isLocal(particleJ_isLocal), normI(normI), rPosI(posI), rPosJ(posJ), contactPoint(contactPoint) {}

void Constraint::print() const {
  std::cout << "Constraint: " << delta0 << std::endl;
  std::cout << "gidI: " << gidI << std::endl;
  std::cout << "gidJ: " << gidJ << std::endl;
  std::cout << "normI: " << normI[0] << ", " << normI[1] << ", " << normI[2] << std::endl;
  std::cout << "rPosI: " << rPosI[0] << ", " << rPosI[1] << ", " << rPosI[2] << std::endl;
  std::cout << "rPosJ: " << rPosJ[0] << ", " << rPosJ[1] << ", " << rPosJ[2] << std::endl;
  std::cout << "contactPoint: " << contactPoint[0] << ", " << contactPoint[1] << ", " << contactPoint[2] << std::endl;
}