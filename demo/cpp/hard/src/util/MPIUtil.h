#pragma once

#include <petsc.h>

#include <limits>
#include <type_traits>

// Helper to map C++ types to MPI_Datatype
template <typename T>
inline MPI_Datatype getMpiDataType() {
  if (std::is_same<T, PetscInt>::value) return MPIU_INT;
  if (std::is_same<T, int>::value) return MPI_INT;
  if (std::is_same<T, double>::value) return MPI_DOUBLE;
  PetscCallAbort(PETSC_COMM_WORLD, MPI_Abort(PETSC_COMM_WORLD, 1));
  return MPI_DATATYPE_NULL;  // Should not be reached
}

// Generic helper for MPI_Allreduce on a single scalar value
template <typename T>
inline T globalReduce(T local_val, MPI_Op op) {
  T global_val;
  MPI_Datatype mpi_type = getMpiDataType<T>();
  PetscCallAbort(PETSC_COMM_WORLD, MPI_Allreduce(&local_val, &global_val, 1, mpi_type, op, PETSC_COMM_WORLD));
  return global_val;
}

// Generic helper for MPI_Allreduce on an array of values
template <typename T>
inline void globalReduce_v(const T* local_vals, T* global_vals, int count, MPI_Op op) {
  MPI_Datatype mpi_type = getMpiDataType<T>();
  PetscCallAbort(PETSC_COMM_WORLD, MPI_Allreduce(local_vals, global_vals, count, mpi_type, op, PETSC_COMM_WORLD));
}