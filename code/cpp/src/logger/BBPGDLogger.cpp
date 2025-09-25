#include "BBPGDLogger.h"

#include <iostream>

namespace vtk {

void BBPGDLogger::collect(const BBPGDStep& step) {
  steps_.push_back(step);
}

void BBPGDLogger::log() {
  MPI_Barrier(PETSC_COMM_WORLD);

  for (const auto& step : steps_) {
    logger_.addFieldData("iteration", step.iteration);
    logger_.addFieldData("residual", step.residual);
    logger_.addFieldData("step_size", step.step_size);
    logger_.addFieldData("grad_norm", step.grad_norm);
    logger_.addFieldData("gamma_norm", step.gamma_norm);
    logger_.write();
  }

   steps_.clear();
}

}  // namespace vtk