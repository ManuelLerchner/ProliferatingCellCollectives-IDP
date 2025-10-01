#pragma once

#include <mpi.h>
#include <petsc.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <unistd.h>

#include <cstdio>
#include <vector>

#include "PetscRaii.h"
namespace utils {

// Get current memory usage in MB
inline double getCurrentMemoryUsageMB() {
  PetscLogDouble t;
  PetscCallAbort(PETSC_COMM_WORLD, PetscMemoryGetCurrentUsage(&t));

  return t / (1024.0 * 1024.0);  // Convert bytes to MB
}

// Get peak memory usage in MB
inline double getPeakMemoryUsageMB() {
  PetscLogDouble t;
  PetscCallAbort(PETSC_COMM_WORLD, PetscMemoryGetMaximumUsage(&t));

  return t / (1024.0 * 1024.0);  // Convert bytes to MB
}

// Get CPU time in seconds
inline double getCPUTimeSeconds() {
  struct rusage rusage;
  getrusage(RUSAGE_SELF, &rusage);
  return rusage.ru_utime.tv_sec + rusage.ru_utime.tv_usec / 1e6 +
         rusage.ru_stime.tv_sec + rusage.ru_stime.tv_usec / 1e6;
}

// Calculate load imbalance based on particles per rank
inline double calculateLoadImbalance(int localParticles) {
  int rank, size;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  MPI_Comm_size(PETSC_COMM_WORLD, &size);

  // Gather particle counts from all ranks
  std::vector<int> particlesPerRank(size);
  MPI_Allgather(&localParticles, 1, MPI_INT,
                particlesPerRank.data(), 1, MPI_INT,
                PETSC_COMM_WORLD);

  // Calculate average and maximum
  double sum = 0;
  int max = 0;
  for (int count : particlesPerRank) {
    sum += count;
    max = std::max(max, count);
  }
  double avg = sum / size;

  // Load imbalance is max/avg ratio
  return avg > 0 ? max / avg : 1.0;
}

// MPI communication time tracking
class MPITimeTracker {
  static double commTimeTotal;
  static double lastStart;

 public:
  static void startOperation() {
    MPI_Barrier(PETSC_COMM_WORLD);  // Synchronize before timing
    lastStart = MPI_Wtime();
  }

  static void endOperation() {
    double end = MPI_Wtime();
    commTimeTotal += end - lastStart;
  }

  static double getCommTime() {
    return commTimeTotal;
  }

  static void reset() {
    commTimeTotal = 0.0;
  }
};

double MPITimeTracker::commTimeTotal = 0.0;
double MPITimeTracker::lastStart = 0.0;

}  // namespace utils