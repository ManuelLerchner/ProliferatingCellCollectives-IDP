#pragma once

#include <petsc.h>

#include "dynamics/ParticleManager.h"

Vec solveLCP(Mat A, Vec b);