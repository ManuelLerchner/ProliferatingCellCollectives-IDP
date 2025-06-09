#include "Constraint.h"

#include <iostream>

Constraint::Constraint(double delta0, int gidI, int gidJ, std::array<double, 3> normI, std::array<double, 3> posI, std::array<double, 3> posJ, std::array<double, 3> labI, std::array<double, 3> labJ)
    : delta0(delta0), gidI(gidI), gidJ(gidJ), normI(normI), rPosI(posI), rPosJ(posJ), labI(labI), labJ(labJ) {}

void Constraint::print() const {
  std::cout << "Constraint: " << delta0 << std::endl;
  std::cout << "gidI: " << gidI << std::endl;
  std::cout << "gidJ: " << gidJ << std::endl;
  std::cout << "normI: " << normI[0] << ", " << normI[1] << ", " << normI[2] << std::endl;
  std::cout << "rPosI: " << rPosI[0] << ", " << rPosI[1] << ", " << rPosI[2] << std::endl;
  std::cout << "rPosJ: " << rPosJ[0] << ", " << rPosJ[1] << ", " << rPosJ[2] << std::endl;
  std::cout << "labI: " << labI[0] << ", " << labI[1] << ", " << labI[2] << std::endl;
  std::cout << "labJ: " << labJ[0] << ", " << labJ[1] << ", " << labJ[2] << std::endl;
}