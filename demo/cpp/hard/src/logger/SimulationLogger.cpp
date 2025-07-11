#include "SimulationLogger.h"

namespace vtk {

SimulationLogger::SimulationLogger(const std::string& outputDirectory, const std::string& baseFilename, bool preserve_existing, size_t step)
    : logger_(outputDirectory, baseFilename, true, preserve_existing, step) {
}

void SimulationLogger::log(const SimulationStep& step) {
  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  // Add all metrics as field data
  logger_.addFieldData("simulation_time_s", step.simulation_time_s);
  logger_.addFieldData("step_duration_s", step.step_duration_s);
  logger_.addFieldData("step", step.step);

  logger_.addFieldData("num_particles", step.num_particles);
  logger_.addFieldData("num_constraints", step.num_constraints);
  logger_.addFieldData("colony_radius", step.colony_radius);

  logger_.addFieldData("recursive_iterations", step.recursive_iterations);
  logger_.addFieldData("bbpgd_iterations", step.bbpgd_iterations);
  logger_.addFieldData("max_overlap", step.max_overlap);
  logger_.addFieldData("residual", step.residual);

  // Add new metrics
  logger_.addFieldData("memory_usage_mb", step.memory_usage_mb);
  logger_.addFieldData("peak_memory_mb", step.peak_memory_mb);
  logger_.addFieldData("cpu_time_s", step.cpu_time_s);
  logger_.addFieldData("mpi_comm_time_s", step.mpi_comm_time_s);
  logger_.addFieldData("load_imbalance", step.load_imbalance);

  logger_.addFieldData("dt_s", step.dt_s);
  // Write to file
  logger_.write();
}

}  // namespace vtk