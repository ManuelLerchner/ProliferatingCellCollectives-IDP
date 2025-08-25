#include <mpi.h>
#include <petsc.h>
#include <petscconf.h>

#include <filesystem>
#include <optional>
#include <string>
#include <type_traits>

#include "loader/VTKStateLoader.h"
#include "spatial/Domain.h"
#include "util/CLIParser.h"

std::optional<Domain> createDomain(const SimulationParameters& params) {
  if (params.starter_vtk.empty()) {
    // Create a new domain
    Domain domain(params.sim_config, params.physics_config,
                  params.solver_config, false, 0, params.mode);

    // Add initial particle
    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    if (rank == 0) {
      Particle p1 = Particle(0, {rank * 3.0, 0, 0}, {1, 0, 0, 0},
                             params.physics_config.l0,
                             params.physics_config.l0,
                             params.physics_config.l0 / 2);

      domain.queueNewParticles({p1});
    }
    return domain;
  } else {
    // Initialize from VTK file
    return Domain::initializeFromVTK(params.sim_config, params.physics_config,
                                     params.solver_config,
                                     params.starter_vtk, params.mode);
  }
}

int main(int argc, char** argv) {
  PetscInitialize(&argc, &argv, 0, 0);
  {
    PetscLogDefaultBegin();

    int total_ranks;
    MPI_Comm_size(PETSC_COMM_WORLD, &total_ranks);
    PetscPrintf(PETSC_COMM_WORLD, "Running simulation with %d ranks\n", total_ranks);

    auto params = parseCommandLineOrDefaults();
    dumpParameters(params);

    auto domain = createDomain(params);
    domain->run();
  }
  PetscFinalize();
  return EXIT_SUCCESS;
}
