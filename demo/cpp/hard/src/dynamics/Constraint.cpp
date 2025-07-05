#include "Constraint.h"

#include <petsc.h>

#include <array>
#include <iostream>

Constraint::Constraint() = default;

Constraint::Constraint(double delta0, int gidI, int gidJ, std::array<double, 3> normI, std::array<double, 3> posI, std::array<double, 3> posJ, std::array<double, 3> contactPoint, double stressI, double stressJ, int gid)
    : delta0(delta0), gidI(gidI), gidJ(gidJ), normI(normI), rPosI(posI), rPosJ(posJ), contactPoint(contactPoint), stressI(stressI), stressJ(stressJ), gid(gid) {}

void Constraint::print() const {
  std::cout << "delta0: " << delta0 << std::endl;
  std::cout << "gidI: " << gidI << std::endl;
  std::cout << "gidJ: " << gidJ << std::endl;
  std::cout << "normI: " << normI[0] << ", " << normI[1] << ", " << normI[2] << std::endl;
  std::cout << "rPosI: " << rPosI[0] << ", " << rPosI[1] << ", " << rPosI[2] << std::endl;
  std::cout << "rPosJ: " << rPosJ[0] << ", " << rPosJ[1] << ", " << rPosJ[2] << std::endl;
  std::cout << "contactPoint: " << contactPoint[0] << ", " << contactPoint[1] << ", " << contactPoint[2] << std::endl;
  std::cout << "stressI: " << stressI << std::endl;
  std::cout << "stressJ: " << stressJ << std::endl;
  std::cout << "gid: " << gid << std::endl
            << std::endl;
}