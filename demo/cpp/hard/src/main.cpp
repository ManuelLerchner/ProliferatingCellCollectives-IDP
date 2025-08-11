#include <mpi.h>
#include <petsc.h>
#include <petscconf.h>

#include <filesystem>
#include <string>

#include "loader/VTKStateLoader.h"
#include "spatial/Domain.h"
#include "util/Config.h"

int main(int argc, char** argv) {
  PetscInitialize(&argc, &argv, 0, 0);

  {
    PetscLogDefaultBegin();

    int total_ranks;
    MPI_Comm_size(PETSC_COMM_WORLD, &total_ranks);

    PetscPrintf(PETSC_COMM_WORLD, "Running simulation with %d ranks\n", total_ranks);

    double DT = 0.5;
    double END_TIME = 550 * 60;
    double LOG_FREQUENCY = 1 * 60;

    PhysicsConfig physic_config = {
        .xi = 200 * 3600,
        .TAU = 54 * 60,
        .l0 = 1,
        .LAMBDA = 2.44e-3,
        .temperature = 1e-30,
        .monolayer = true,
    };

    SimulationConfig sim_config = {
        .dt_s = DT,
        .end_time = END_TIME,
        .log_frequency_seconds = LOG_FREQUENCY,
        .min_box_size = {physic_config.l0 + 1, physic_config.l0 + 1, 0},
    };

    SolverConfig solver_config = {
        .tolerance = physic_config.l0 / 1e3,
        .allowed_overlap = physic_config.l0 / 1e2,
        .max_bbpgd_iterations = 100000,
        .max_recursive_iterations = 50,
        .linked_cell_size = physic_config.l0 * 2.2,
        .growth_factor = 1.5,
        .particle_preallocation_factor = 12,
    };

    // Use PETSc's option system
    char starter_vtk_cstr[PETSC_MAX_PATH_LEN];
    PetscBool starter_vtk_set;

    PetscOptionsGetString(NULL, NULL, "-starter_vtk", starter_vtk_cstr,
                          sizeof(starter_vtk_cstr), &starter_vtk_set);

    char mode_cstr[PETSC_MAX_PATH_LEN];
    PetscBool mode_set;

    PetscOptionsGetString(NULL, NULL, "-mode", mode_cstr,
                          sizeof(mode_cstr), &mode_set);

    std::string mode = std::string(mode_cstr);

    std::string starter_vtk;
    if (starter_vtk_set) {
      starter_vtk = std::string(starter_vtk_cstr);
    }

    std::optional<Domain> domain;

    if (starter_vtk.empty()) {
      domain = Domain(sim_config, physic_config, solver_config, !starter_vtk.empty(), 0, mode);
    } else {
      domain = Domain::initializeFromVTK(sim_config, physic_config, solver_config, starter_vtk, mode);
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