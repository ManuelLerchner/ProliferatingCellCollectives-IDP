#include "SimulationLogger.h"

namespace vtk {

SimulationLogger::SimulationLogger(const std::string& outputDirectory, const std::string& baseFilename)
    : logger_(outputDirectory, baseFilename, true) {
}

void SimulationLogger::log(const SimulationStep& step) {
  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  // Add all metrics as field data
  logger_.addFieldData("simulation_time_s", step.simulation_time_s);
  logger_.addFieldData("step_duration_s", step.step_duration_s);
  logger_.addFieldData("run_time_s", step.run_time_s);
  logger_.addFieldData("num_particles", step.num_particles);
  logger_.addFieldData("num_constraints", step.num_constraints);
  logger_.addFieldData("recursive_iterations", step.recursive_iterations);
  logger_.addFieldData("bbpgd_iterations", step.bbpgd_iterations);
  logger_.addFieldData("max_overlap", step.max_overlap);
  logger_.addFieldData("residual", step.residual);
  logger_.addFieldData("dt_s", step.dt_s);

  // Write to file
  logger_.write();
}

}  // namespace vtk