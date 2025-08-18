#pragma once

#include <petsc.h>

#include "util/Config.h"

template <typename T>
void getOption(const char* name, T& value) {
  if constexpr (std::is_same_v<T, PetscReal>) {
    PetscOptionsGetReal(NULL, NULL, name, &value, NULL);
  } else if constexpr (std::is_same_v<T, PetscInt>) {
    PetscOptionsGetInt(NULL, NULL, name, &value, NULL);
  } else if constexpr (std::is_same_v<T, PetscBool>) {
    PetscOptionsGetBool(NULL, NULL, name, &value, NULL);
  }
}

template <typename T>
void printParam(const char* label, const T& value) {
  PetscPrintf(PETSC_COMM_WORLD, "%-25s: ", label);
  if constexpr (std::is_floating_point_v<T>)
    PetscPrintf(PETSC_COMM_WORLD, "%.6f\n", value);
  else if constexpr (std::is_integral_v<T>)
    PetscPrintf(PETSC_COMM_WORLD, "%d\n", value);
  else if constexpr (std::is_same_v<T, bool>)
    PetscPrintf(PETSC_COMM_WORLD, "%s\n", value ? "true" : "false");
  else
    PetscPrintf(PETSC_COMM_WORLD, "%s\n", value);
}

void dumpParameters(const SimulationParameters& params) {
  PetscPrintf(PETSC_COMM_WORLD, "\n=== SIMULATION CONFIGURATION ===\n");
  printParam("Time step (dt)", params.sim_config.dt_s);
  printParam("End time", params.sim_config.end_time);
  printParam("Log frequency", params.sim_config.log_frequency_seconds);
  PetscPrintf(PETSC_COMM_WORLD, "Min box size             : [%.3f, %.3f, %.3f]\n",
              params.sim_config.min_box_size.x,
              params.sim_config.min_box_size.y,
              params.sim_config.min_box_size.z);

  PetscPrintf(PETSC_COMM_WORLD, "\n=== PHYSICS CONFIGURATION ===\n");
  printParam("Xi (viscosity)", params.physics_config.xi);
  printParam("TAU (growth time)", params.physics_config.TAU);
  printParam("l0 (reference length)", params.physics_config.l0);
  printParam("LAMBDA", params.physics_config.getLambdaDimensionless());
  printParam("Temperature", params.physics_config.temperature);
  printParam("cell_mu (friction)", params.physics_config.cell_mu);
  printParam("alpha", params.physics_config.alpha);
  printParam("Baumgarte factor", params.physics_config.baumgarte_factor);
  printParam("Monolayer mode", (bool)params.physics_config.monolayer);

  PetscPrintf(PETSC_COMM_WORLD, "\n=== SOLVER CONFIGURATION ===\n");
  printParam("Tolerance", params.solver_config.tolerance);
  printParam("Allowed overlap", params.solver_config.allowed_overlap);
  printParam("Max BBPGD iterations", params.solver_config.max_bbpgd_iterations);
  printParam("Max recursive iterations", params.solver_config.max_recursive_iterations);
  printParam("Linked cell size", params.solver_config.linked_cell_size);
  printParam("Growth factor", params.solver_config.growth_factor);
  printParam("Particle prealloc factor", params.solver_config.particle_preallocation_factor);

  if (!params.starter_vtk.empty() || !params.mode.empty()) {
    PetscPrintf(PETSC_COMM_WORLD, "\n=== FILE/MODE OPTIONS ===\n");
    if (!params.starter_vtk.empty())
      printParam("Starter VTK file", params.starter_vtk.c_str());
    if (!params.mode.empty())
      printParam("Simulation mode", params.mode.c_str());
  }

  PetscPrintf(PETSC_COMM_WORLD, "\n=== DERIVED VALUES ===\n");
  printParam("Lambda dimensionless", params.physics_config.getLambdaDimensionless());
  PetscPrintf(PETSC_COMM_WORLD, "Tolerance/l0 ratio       : %.2e\n",
              params.solver_config.tolerance / params.physics_config.l0);
  PetscPrintf(PETSC_COMM_WORLD, "Overlap/l0 ratio         : %.2e\n",
              params.solver_config.allowed_overlap / params.physics_config.l0);
  PetscPrintf(PETSC_COMM_WORLD, "Cell size/l0 ratio       : %.3f\n",
              params.solver_config.linked_cell_size / params.physics_config.l0);

  PetscPrintf(PETSC_COMM_WORLD, "\n================================\n\n");
}

SimulationParameters parseCommandLineOrDefaults() {
  SimulationParameters params;

  // Default configs
  params.sim_config = {
      .dt_s = 1.0,
      .end_time = 700 * 60,
      .log_frequency_seconds = 60,
      .min_box_size = {2.0, 2.0, 0},
  };

  params.physics_config = {
      .xi = 200 * 3600,
      .TAU = 54 * 60,
      .l0 = 1.0,
      .LAMBDA = 1e-2,
      .temperature = 1e-30,
      .cell_mu = 0.2,
      .alpha = 0.5,
      .baumgarte_factor = 0.05,
      .monolayer = PETSC_TRUE,
  };

  params.solver_config = {
      .tolerance = 1e-3,
      .allowed_overlap = 1e-2,
      .max_bbpgd_iterations = 100000,
      .max_recursive_iterations = 50,
      .linked_cell_size = 2.2,
      .growth_factor = 1.5,
      .particle_preallocation_factor = 12,
  };

  // Parse overrides
  getOption("-dt", params.sim_config.dt_s);
  getOption("-end_time", params.sim_config.end_time);
  getOption("-log_frequency", params.sim_config.log_frequency_seconds);

  getOption("-xi", params.physics_config.xi);
  getOption("-tau", params.physics_config.TAU);
  getOption("-l0", params.physics_config.l0);
  getOption("-lambda", params.physics_config.LAMBDA);
  getOption("-temperature", params.physics_config.temperature);
  getOption("-cell_mu", params.physics_config.cell_mu);
  getOption("-alpha", params.physics_config.alpha);
  getOption("-baumgarte_factor", params.physics_config.baumgarte_factor);
  getOption("-monolayer", params.physics_config.monolayer);

  getOption("-tolerance", params.solver_config.tolerance);
  getOption("-allowed_overlap", params.solver_config.allowed_overlap);
  getOption("-max_bbpgd_iterations", params.solver_config.max_bbpgd_iterations);
  getOption("-max_recursive_iterations", params.solver_config.max_recursive_iterations);
  getOption("-linked_cell_size", params.solver_config.linked_cell_size);
  getOption("-growth_factor", params.solver_config.growth_factor);
  getOption("-particle_preallocation_factor", params.solver_config.particle_preallocation_factor);

  // Check l0-dependent values if not explicitly set
  PetscBool tolerance_set, overlap_set, cell_size_set;
  PetscOptionsHasName(NULL, NULL, "-tolerance", &tolerance_set);
  PetscOptionsHasName(NULL, NULL, "-allowed_overlap", &overlap_set);
  PetscOptionsHasName(NULL, NULL, "-linked_cell_size", &cell_size_set);

  if (!tolerance_set) params.solver_config.tolerance = params.physics_config.l0 / 1e3;
  if (!overlap_set) params.solver_config.allowed_overlap = params.physics_config.l0 / 1e2;
  if (!cell_size_set) params.solver_config.linked_cell_size = params.physics_config.l0 * 2.2;

  params.sim_config.min_box_size = {
      params.physics_config.l0 + 1,
      params.physics_config.l0 + 1,
      0};

  // File options
  char starter_vtk_cstr[PETSC_MAX_PATH_LEN] = "";
  char mode_cstr[PETSC_MAX_PATH_LEN] = "";
  PetscBool starter_vtk_set, mode_set;

  PetscOptionsGetString(NULL, NULL, "-starter_vtk", starter_vtk_cstr,
                        sizeof(starter_vtk_cstr), &starter_vtk_set);
  PetscOptionsGetString(NULL, NULL, "-mode", mode_cstr,
                        sizeof(mode_cstr), &mode_set);

  if (starter_vtk_set) params.starter_vtk = starter_vtk_cstr;
  if (mode_set) params.mode = mode_cstr;

  return params;
}