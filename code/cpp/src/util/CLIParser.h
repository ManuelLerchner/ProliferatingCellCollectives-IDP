#pragma once

#include <petsc.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "util/Config.h"

// Helper function to print help text with current default values
void printHelp(const SimulationParameters& params) {
  PetscPrintf(PETSC_COMM_WORLD, "\nAvailable parameters:\n\n");

  // Simulation Configuration
  PetscPrintf(PETSC_COMM_WORLD, "\nSimulation Configuration:\n");
  PetscPrintf(PETSC_COMM_WORLD, "  %-25s %-50s [default: %g]\n", "-dt", "Time step in seconds",
              params.sim_config.dt_s);
  PetscPrintf(PETSC_COMM_WORLD, "  %-25s %-50s [default: %g]\n", "-end_radius", "Final radius to simulate",
              params.sim_config.end_radius);
  PetscPrintf(PETSC_COMM_WORLD, "  %-25s %-50s [default: %g]\n", "-log_every_colony_radius_delta",
              "Log once colony radius increases by this amount (0 to disable)",
              params.sim_config.log_every_colony_radius_delta);
  PetscPrintf(PETSC_COMM_WORLD, "  %-25s %-50s [default: %g]\n", "-log_every_sim_time_delta",
              "Log every this many seconds of simulation time (0 to disable)",
              params.sim_config.log_every_colony_radius_delta);

  // Physics Configuration
  PetscPrintf(PETSC_COMM_WORLD, "\nPhysics Configuration:\n");
  PetscPrintf(PETSC_COMM_WORLD, "  %-25s %-50s [default: %g]\n", "-xi", "Viscosity parameter",
              params.physics_config.xi);
  PetscPrintf(PETSC_COMM_WORLD, "  %-25s %-50s [default: %g]\n", "-tau", "Growth time constant",
              params.physics_config.TAU);
  PetscPrintf(PETSC_COMM_WORLD, "  %-25s %-50s [default: %g]\n", "-l0", "Reference length", params.physics_config.l0);
  PetscPrintf(PETSC_COMM_WORLD, "  %-25s %-50s [default: %g]\n", "-lambda", "Lambda parameter",
              params.physics_config.LAMBDA);
  PetscPrintf(PETSC_COMM_WORLD, "  %-25s %-50s [default: %g]\n", "-temperature", "Simulation temperature",
              params.physics_config.temperature);
  PetscPrintf(PETSC_COMM_WORLD, "  %-25s %-50s [default: %g]\n", "-kcc", "Collision constant",
              params.physics_config.kcc);
  PetscPrintf(PETSC_COMM_WORLD, "  %-25s %-50s [default: %s]\n", "-monolayer", "Enable monolayer mode",
              params.physics_config.monolayer ? "true" : "false");

  // Solver Configuration
  PetscPrintf(PETSC_COMM_WORLD, "\nSolver Configuration:\n");
  PetscPrintf(PETSC_COMM_WORLD, "  %-25s %-50s [default: %g]\n", "-tolerance", "Solver tolerance",
              params.solver_config.tolerance);
  PetscPrintf(PETSC_COMM_WORLD, "  %-25s %-50s [default: %ld]\n", "-max_bbpgd_iterations", "Maximum BBPGD iterations",
              params.solver_config.max_bbpgd_iterations);
  PetscPrintf(PETSC_COMM_WORLD, "  %-25s %-50s [default: %ld]\n", "-max_recursive_iterations",
              "Maximum recursive iterations", params.solver_config.max_recursive_iterations);
  PetscPrintf(PETSC_COMM_WORLD, "  %-25s %-50s [default: %g]\n", "-linked_cell_size", "Size of linked cells",
              params.solver_config.linked_cell_size);
  PetscPrintf(PETSC_COMM_WORLD, "  %-25s %-50s [default: %g]\n", "-growth_factor", "Growth factor",
              params.solver_config.growth_factor);
  PetscPrintf(PETSC_COMM_WORLD, "  %-25s %-50s [default: %g]\n", "-particle_preallocation_factor",
              "Particle preallocation factor", params.solver_config.particle_preallocation_factor);

  // File/Mode Options
  PetscPrintf(PETSC_COMM_WORLD, "\nFile/Mode Options:\n");
  PetscPrintf(PETSC_COMM_WORLD, "  %-25s %-50s [default: %s]\n", "-starter_vtk", "Path to starter VTK file",
              params.starter_vtk.empty() ? "none" : params.starter_vtk.c_str());
  PetscPrintf(PETSC_COMM_WORLD, "  %-25s %-50s [default: %s] (required)\n", "-mode", "Simulation mode",
              params.mode.empty() ? "none" : params.mode.c_str());

  PetscPrintf(PETSC_COMM_WORLD, "\n");
}

template <typename T>
void getOption(const char* name, T& value) {
  PetscBool optionSet = PETSC_FALSE;

  if constexpr (std::is_same_v<T, PetscReal>) {
    PetscOptionsGetReal(NULL, NULL, name, &value, &optionSet);
  } else if constexpr (std::is_same_v<T, PetscInt>) {
    PetscOptionsGetInt(NULL, NULL, name, &value, &optionSet);
  } else if constexpr (std::is_same_v<T, PetscBool>) {
    PetscOptionsGetBool(NULL, NULL, name, &value, &optionSet);
  }

  // Print a message if the option was set
  if (optionSet) {
    if constexpr (std::is_floating_point_v<T>)
      PetscPrintf(PETSC_COMM_WORLD, "Set %s = %g\n", name, value);
    else if constexpr (std::is_integral_v<T>)
      PetscPrintf(PETSC_COMM_WORLD, "Set %s = %ld\n", name, value);
    else if constexpr (std::is_same_v<T, PetscBool>)
      PetscPrintf(PETSC_COMM_WORLD, "Set %s = %s\n", name, value ? "true" : "false");
  }
}

template <typename T>
void printParam(const char* label, const T& value) {
  PetscPrintf(PETSC_COMM_WORLD, "%-25s: ", label);

  if constexpr (std::is_floating_point_v<T>) {
    PetscPrintf(PETSC_COMM_WORLD, "%.6f\n", value);
  } else if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
    PetscPrintf(PETSC_COMM_WORLD, "%lld\n", static_cast<long long>(value));
  } else if constexpr (std::is_integral_v<T> && std::is_unsigned_v<T>) {
    PetscPrintf(PETSC_COMM_WORLD, "%llu\n", static_cast<unsigned long long>(value));
  } else if constexpr (std::is_same_v<T, bool>) {
    PetscPrintf(PETSC_COMM_WORLD, "%s\n", value ? "true" : "false");
  } else {
    PetscPrintf(PETSC_COMM_WORLD, "%s\n", value);
  }
}

void dumpParameters(const SimulationParameters& params) {
  PetscPrintf(PETSC_COMM_WORLD, "\n=== SIMULATION CONFIGURATION ===\n");
  printParam("Time step (dt)", params.sim_config.dt_s);
  printParam("End radius", params.sim_config.end_radius);
  printParam("Log every colony radius delta", params.sim_config.log_every_colony_radius_delta);
  printParam("Log every sim time delta", params.sim_config.log_every_colony_radius_delta);
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
  printParam("kcc", params.physics_config.kcc);
  printParam("Monolayer mode", (bool)params.physics_config.monolayer);

  PetscPrintf(PETSC_COMM_WORLD, "\n=== SOLVER CONFIGURATION ===\n");
  printParam("Tolerance", params.solver_config.tolerance);
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
  PetscPrintf(PETSC_COMM_WORLD, "Cell size/l0 ratio       : %.3f\n",
              params.solver_config.linked_cell_size / params.physics_config.l0);

  PetscPrintf(PETSC_COMM_WORLD, "\n================================\n\n");
}

SimulationParameters parseCommandLineOrDefaults() {
  SimulationParameters params;

  // Default configs
  params.sim_config = {
      .dt_s = 0.5 * 1e-4,
      .end_radius = 50,
      .log_every_sim_time_delta = 100000,
      .log_every_colony_radius_delta = 0.5,
      .min_box_size = {2.0, 2.0, 0},
  };

  params.physics_config = {
      .xi = 1,
      .TAU = 1,
      .l0 = 1.0,
      .LAMBDA = 1e-2,
      .temperature = 1e-28,
      .kcc = 20000,
      .monolayer = PETSC_TRUE,
  };

  params.solver_config = {
      .tolerance = 1e-3,
      .max_bbpgd_iterations = 500000,
      .max_recursive_iterations = 50,
      .linked_cell_size = 2.5,
      .growth_factor = 1.5,
      .particle_preallocation_factor = 20,
  };

  // Check for help flag
  PetscBool help = PETSC_FALSE;
  PetscOptionsGetBool(NULL, NULL, "-help", &help, NULL);
  if (help) {
    printHelp(params);
    PetscFinalize();
    exit(0);
  }

  // Parse overrides
  getOption("-dt", params.sim_config.dt_s);
  getOption("-end_radius", params.sim_config.end_radius);
  getOption("-log_every_colony_radius_delta", params.sim_config.log_every_colony_radius_delta);
  getOption("-log_every_sim_time_delta", params.sim_config.log_every_sim_time_delta);

  getOption("-xi", params.physics_config.xi);
  getOption("-tau", params.physics_config.TAU);
  getOption("-l0", params.physics_config.l0);
  getOption("-lambda", params.physics_config.LAMBDA);
  getOption("-temperature", params.physics_config.temperature);
  getOption("-kcc", params.physics_config.kcc);
  getOption("-monolayer", params.physics_config.monolayer);

  getOption("-tolerance", params.solver_config.tolerance);
  getOption("-max_bbpgd_iterations", params.solver_config.max_bbpgd_iterations);
  getOption("-max_recursive_iterations", params.solver_config.max_recursive_iterations);
  getOption("-linked_cell_size", params.solver_config.linked_cell_size);
  getOption("-growth_factor", params.solver_config.growth_factor);
  getOption("-particle_preallocation_factor", params.solver_config.particle_preallocation_factor);

  // Check l0-dependent values if not explicitly set
  PetscBool cell_size_set;
  PetscOptionsHasName(NULL, NULL, "-linked_cell_size", &cell_size_set);

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

  // Check required parameters
  if (params.mode.empty()) {
    PetscPrintf(PETSC_COMM_WORLD,
                "Error: Required parameter -mode not provided. Use -help for usage information.\n");
    PetscFinalize();
    exit(1);
  }

  return params;
}
