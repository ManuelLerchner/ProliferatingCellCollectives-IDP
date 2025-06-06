#include "Constraint.h"

Constraint::Constraint(double delta0, int gidI, int gidJ, std::array<double, 3> normI, std::array<double, 3> posI, std::array<double, 3> posJ, std::array<double, 3> labI, std::array<double, 3> labJ)
    : delta0(delta0), gidI(gidI), gidJ(gidJ), normI(normI), rPosI(posI), rPosJ(posJ), labI(labI), labJ(labJ) {}