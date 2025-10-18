#include "BBPGDLogger.h"

#include <iostream>

namespace vtk {

void BBPGDLogger::collect(BBPGDStep step) {
  step.recursive_iteration = recursive_iteration;
  steps_.push_back(step);
}

void BBPGDLogger::log() {

  for (const auto& step : steps_) {
    logger_.addFieldData("step", step.step);
    logger_.addFieldData("recursive_iteration", step.recursive_iteration);
    logger_.addFieldData("total_constraints", step.total_constraints);
    logger_.addFieldData("residual", step.residual);
    logger_.addFieldData("step_size", step.step_size);
    logger_.addFieldData("linear", step.linear);
    logger_.addFieldData("quadratic", step.quadratic);
    logger_.addFieldData("growth", step.growth);
    logger_.addFieldData("total", step.total);
    logger_.write();
  }

  steps_.clear();
}

}  // namespace vtk