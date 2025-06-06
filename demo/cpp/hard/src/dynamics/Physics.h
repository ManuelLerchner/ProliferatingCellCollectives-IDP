#pragma once

#include <petsc.h>

#include <memory>

std::unique_ptr<Mat> calculate_MobilityMatrix(Vec configuration, int number_of_particles);