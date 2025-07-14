#include "Constraint.h"

#include <petsc.h>

#include <array>
#include <iostream>

#include "util/ArrayMath.h"

Constraint::Constraint() = default;

Constraint::Constraint(double signed_distance, int gidI, int gidJ, std::array<double, 3> normI, std::array<double, 3> posI, std::array<double, 3> posJ, std::array<double, 3> contactPoint, double stressI, double stressJ, int gid, int iteration, bool localI, bool localJ) {
  using namespace utils::ArrayMath;


  this->signed_distance = signed_distance;
  this->gidI = gidI;
  this->gidJ = gidJ;
  this->normI = normI;
  this->rPosI = posI;
  this->rPosJ = posJ;
  this->contactPoint = contactPoint;
  this->stressI = stressI;
  this->stressJ = stressJ;
  this->gid = gid;
  this->iteration = iteration;
  this->localI = localI;
  this->localJ = localJ;
}

void Constraint::print() const {
  std::cout << "signed_distance: " << signed_distance << std::endl;
  std::cout << "gidI: " << gidI << std::endl;
  std::cout << "gidJ: " << gidJ << std::endl;
  std::cout << "normI: " << normI[0] << ", " << normI[1] << ", " << normI[2] << std::endl;
  std::cout << "rPosI: " << rPosI[0] << ", " << rPosI[1] << ", " << rPosI[2] << std::endl;
  std::cout << "rPosJ: " << rPosJ[0] << ", " << rPosJ[1] << ", " << rPosJ[2] << std::endl;
  std::cout << "contactPoint: " << contactPoint[0] << ", " << contactPoint[1] << ", " << contactPoint[2] << std::endl;
  std::cout << "stressI: " << stressI << std::endl;
  std::cout << "stressJ: " << stressJ << std::endl;
  std::cout << "gid: " << gid << std::endl;
  std::cout << "iteration: " << iteration << std::endl;
  std::cout << "localI: " << localI << std::endl;
  std::cout << "localJ: " << localJ << std::endl;
  std::cout << std::endl;
}