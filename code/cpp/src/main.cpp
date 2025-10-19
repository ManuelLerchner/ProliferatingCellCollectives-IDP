#include <mpi.h>
#include <omp.h>
#include <petsc.h>
#include <petscconf.h>
#include <petscsys.h>

#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>

#include "loader/VTKStateLoader.h"
#include "spatial/Domain.h"
#include "util/CLIParser.h"

std::optional<Domain> createDomain(SimulationParameters& params) {
  if (params.starter_vtk.empty()) {
    // Create a new domain
    Domain domain(params, false, 0, params.mode);

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
    return Domain::initializeFromVTK(params,
                                     params.starter_vtk, params.mode);
  }
}

int main(int argc, char** argv) {
  PetscInitialize(&argc, &argv, 0, 0);
  {
    PetscLogDefaultBegin();
    PetscMemorySetGetMaximumUsage();

    int total_ranks;
    MPI_Comm_size(PETSC_COMM_WORLD, &total_ranks);

    // Get maximum available CPUs
    int max_cpus = std::thread::hardware_concurrency();

    int nthreads = omp_get_max_threads();

    // Check if requested threads exceed available CPUs
    if (nthreads * total_ranks > max_cpus) {
      PetscPrintf(PETSC_COMM_WORLD,
                  "Error: total requested threads (%d ranks * %d threads = %d) exceed system CPU count (%d)\n",
                  total_ranks, nthreads, nthreads * total_ranks, max_cpus);
      PetscPrintf(PETSC_COMM_WORLD, "Requested threads exceed available CPU cores");
      exit(EXIT_FAILURE);
    }

    PetscPrintf(PETSC_COMM_WORLD,
                "Starting simulation with %d MPI ranks and %d OpenMP threads per rank (total %d threads)\n",
                total_ranks, nthreads, nthreads * total_ranks);

    auto params = parseCommandLineOrDefaults();
    dumpParameters(params);

    auto domain = createDomain(params);
    domain->run();
  }
  PetscFinalize();
  return EXIT_SUCCESS;
}
