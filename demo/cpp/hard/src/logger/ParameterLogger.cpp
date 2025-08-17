#include "ParameterLogger.h"

#include <iostream>

namespace vtk {

ParameterLogger::ParameterLogger(const std::string& outputDirectory, const std::string& baseFilename, bool preserve_existing, size_t step)
    : logger_(outputDirectory, baseFilename, true, preserve_existing, step) {
}

void ParameterLogger::log(const SimulationParameters& params) {
  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  // Add all metrics as field data
  logger_.addFieldData("dt_s", params.sim_config.dt_s);
  logger_.addFieldData("end_time", params.sim_config.end_time);
  logger_.addFieldData("log_frequency_seconds", params.sim_config.log_frequency_seconds);
  logger_.addFieldData("min_box_size_x", params.sim_config.min_box_size.x);
  logger_.addFieldData("min_box_size_y", params.sim_config.min_box_size.y);
  logger_.addFieldData("min_box_size_z", params.sim_config.min_box_size.z);

  logger_.addFieldData("xi", params.physics_config.xi);
  logger_.addFieldData("TAU", params.physics_config.TAU);
  logger_.addFieldData("l0", params.physics_config.l0);
  logger_.addFieldData("LAMBDA", params.physics_config.LAMBDA);
  logger_.addFieldData("temperature", params.physics_config.temperature);
  logger_.addFieldData("k_cc", params.physics_config.k_cc);
  logger_.addFieldData("gamma_n", params.physics_config.gamma_n);
  logger_.addFieldData("gamma_t", params.physics_config.gamma_t);
  logger_.addFieldData("cell_mu", params.physics_config.cell_mu);
  logger_.addFieldData("alpha", params.physics_config.alpha);
  logger_.addFieldData("baumgarte_factor", params.physics_config.baumgarte_factor);

  logger_.addFieldData("max_bbpgd_iterations", params.solver_config.max_bbpgd_iterations);
  logger_.addFieldData("max_recursive_iterations", params.solver_config.max_recursive_iterations);
  logger_.addFieldData("linked_cell_size", params.solver_config.linked_cell_size);
  logger_.addFieldData("growth_factor", params.solver_config.growth_factor);
  logger_.addFieldData("particle_preallocation_factor", params.solver_config.particle_preallocation_factor);

  logger_.addFieldData("mode", params.mode == "hard" ? 0 : 1);

  // Write to file
  logger_.write();
}

}  // namespace vtk