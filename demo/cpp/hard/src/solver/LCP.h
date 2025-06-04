#pragma once

#include <petsc.h>

#include "dynamics/BacterialSystem.h"

Vec solveLCP(Mat A, Vec b);