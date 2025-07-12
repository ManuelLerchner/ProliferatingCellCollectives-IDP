#include <mpi.h>
#include <petsc.h>
#include <petscconf.h>

#include <filesystem>
#include <string>

#include "loader/VTKStateLoader.h"
#include "spatial/Domain.h"
#include "util/Config.h"

void printUsage(const char* program) {
  PetscPrintf(PETSC_COMM_WORLD, "Usage: %s [--starter-vtk <path>]\n", program);
  PetscPrintf(PETSC_COMM_WORLD, "  --starter-vtk: Path to VTK directory or PVTU file to initialize from\n");
}

int main(int argc, char** argv) {
  PetscInitialize(&argc, &argv, 0, 0);

  {
    PetscLogDefaultBegin();

    int total_ranks;
    MPI_Comm_size(PETSC_COMM_WORLD, &total_ranks);

    PetscPrintf(PETSC_COMM_WORLD, "Running simulation with %d ranks\n", total_ranks);

    double DT = 60;
    double END_TIME = 1000 * 60;
    double LOG_FREQUENCY = 1 * 60;

    PhysicsConfig physic_config = {
        .xi = 200 * 3600,
        .TAU = 54 * 60,
        .l0 = 1,
        .LAMBDA = 2.44e-3,
        .temperature = 0.01,
        .monolayer = true,
    };

    SimulationConfig sim_config = {
        .dt_s = DT,
        .end_time = END_TIME,
        .log_frequency_seconds = LOG_FREQUENCY,
        .min_box_size = {physic_config.l0 + 2, physic_config.l0 + 2, 0},

        .enable_adaptive_dt = false,
        .target_bbpgd_iterations = 500,
        .dt_adjust_frequency = 10,
        .dt_adjust_factor = 0.1,
        .min_dt = 1e-6,
        .max_dt = DT};

    SolverConfig solver_config = {
        .tolerance = physic_config.l0 / 1e3,
        .allowed_overlap = physic_config.l0 / 1e2,
        .max_bbpgd_iterations = 10000,
        .max_recursive_iterations = 50,
        .linked_cell_size = physic_config.l0 * 2.2,
        .growth_factor = 1.1,
        .particle_preallocation_factor = 10,
    };

    // Use PETSc's option system
    char starter_vtk_cstr[PETSC_MAX_PATH_LEN];
    PetscBool starter_vtk_set;

    PetscOptionsGetString(NULL, NULL, "-starter_vtk", starter_vtk_cstr,
                          sizeof(starter_vtk_cstr), &starter_vtk_set);

    std::string starter_vtk;
    if (starter_vtk_set) {
      starter_vtk = std::string(starter_vtk_cstr);
    }

    std::optional<Domain> domain;

    if (starter_vtk.empty()) {
      domain = Domain(sim_config, physic_config, solver_config, !starter_vtk.empty());
    } else {
      domain = Domain::initializeFromVTK(sim_config, physic_config, solver_config, starter_vtk);
    }

    if (starter_vtk.empty()) {
      int rank;
      MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
      if (rank == 0) {
        double angle = 0;
        Particle p1 = Particle(0, {rank * 3.0, 0, 0}, {cos(angle / 2), 0, 0, sin(angle / 2)},
                               physic_config.l0, physic_config.l0, physic_config.l0 / 2);
        domain->queueNewParticles({p1});
      }
    }

    domain->run();

    // PetscLogView(PETSC_VIEWER_STDOUT_WORLD);
  }
  PetscFinalize();
  return EXIT_SUCCESS;
}