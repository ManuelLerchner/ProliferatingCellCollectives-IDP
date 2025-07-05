#pragma once

#include <stdexcept>
#include <vector>

#include "Config.h"
#include "PetscRaii.h"

size_t vecSize(const VecWrapper& vec) {
  PetscInt size;
  PetscCallAbort(PETSC_COMM_WORLD, VecGetSize(vec, &size));
  return size;
}

class DynamicVecWrapper {
 private:
  VecWrapper vec;
  PetscInt global_size;
  PetscInt global_capacity;
  double growth_factor;
  int min_local_buffer;

 public:
  DynamicVecWrapper(int min_local_preallocation_size, double growth_factor_) {
    min_local_buffer = min_local_preallocation_size;
    growth_factor = growth_factor_;
    global_size = 0;
    vec = VecWrapper::Create(min_local_buffer);
    PetscCallAbort(PETSC_COMM_WORLD, VecGetSize(vec, &global_capacity));
  }

  void ensureCapacity(PetscInt additional_items) {
    PetscInt required_global_size = global_size + additional_items;
    if (global_capacity < required_global_size) {
      PetscInt new_global_capacity = std::max(required_global_size, static_cast<PetscInt>(global_capacity * growth_factor));
      PetscPrintf(PETSC_COMM_WORLD, "\nLOG: DynamicVecWrapper reallocating from capacity %d to %d (size: %d, required: %d)\n", global_capacity, new_global_capacity, global_size, required_global_size);

      int mpi_size;
      MPI_Comm_size(PETSC_COMM_WORLD, &mpi_size);
      PetscInt new_local_capacity = (new_global_capacity + mpi_size - 1) / mpi_size;

      VecWrapper new_vec = VecWrapper::Create(new_local_capacity);
      if (global_size > 0) {
        IS is_from, is_to;
        PetscCallAbort(PETSC_COMM_WORLD, ISCreateStride(PETSC_COMM_WORLD, global_size, 0, 1, &is_from));
        PetscCallAbort(PETSC_COMM_WORLD, ISCreateStride(PETSC_COMM_WORLD, global_size, 0, 1, &is_to));
        VecScatter scatter;
        PetscCallAbort(PETSC_COMM_WORLD, VecScatterCreate(vec, is_from, new_vec, is_to, &scatter));
        PetscCallAbort(PETSC_COMM_WORLD, VecScatterBegin(scatter, vec, new_vec, INSERT_VALUES, SCATTER_FORWARD));
        PetscCallAbort(PETSC_COMM_WORLD, VecScatterEnd(scatter, vec, new_vec, INSERT_VALUES, SCATTER_FORWARD));
        PetscCallAbort(PETSC_COMM_WORLD, VecScatterDestroy(&scatter));
        PetscCallAbort(PETSC_COMM_WORLD, ISDestroy(&is_from));
        PetscCallAbort(PETSC_COMM_WORLD, ISDestroy(&is_to));
      }
      vec = std::move(new_vec);
      PetscCallAbort(PETSC_COMM_WORLD, VecGetSize(vec, &global_capacity));
    }
  }

  operator VecWrapper&() { return vec; }
  operator const VecWrapper&() const { return vec; }
  operator Vec() const { return vec; }
  PetscInt getSize() const { return global_size; }

  void updateSize(PetscInt size_delta) {
    global_size += size_delta;
  }

  void copyTo(VecWrapper& other) const {
    PetscCallAbort(PETSC_COMM_WORLD, VecCopy(vec, other));
  }
};

class DynamicMatWrapper {
 private:
  MatWrapper mat;
  PetscInt global_ncols;
  PetscInt global_col_capacity;
  double growth_factor;
  int min_local_col_buffer;

 public:
  DynamicMatWrapper(PetscInt local_rows, int min_local_col_preallocation, double growth_factor_) {
    growth_factor = growth_factor_;
    min_local_col_buffer = min_local_col_preallocation;
    global_ncols = 0;

    int mpi_size;
    MPI_Comm_size(PETSC_COMM_WORLD, &mpi_size);
    PetscInt initial_global_col_capacity = min_local_col_buffer * mpi_size;

    mat = MatWrapper::CreateAIJ(local_rows, min_local_col_buffer, initial_global_col_capacity);

    PetscCallAbort(PETSC_COMM_WORLD, MatGetSize(mat, NULL, &global_col_capacity));
  }

  void ensureCapacity(PetscInt additional_cols) {
    PetscInt required_global_cols = global_ncols + additional_cols;
    if (global_col_capacity < required_global_cols) {
      PetscInt new_global_col_capacity = std::max(required_global_cols, static_cast<PetscInt>(global_col_capacity * growth_factor));
      PetscPrintf(PETSC_COMM_WORLD, "\nLOG: DynamicMatWrapper reallocating from capacity %d to %d (cols: %d, required: %d )\n", global_col_capacity, new_global_col_capacity, global_ncols, required_global_cols);

      PetscInt m_local;
      MatGetLocalSize(mat, &m_local, NULL);

      int mpi_size;
      MPI_Comm_size(PETSC_COMM_WORLD, &mpi_size);
      PetscInt new_local_ncols = (new_global_col_capacity + mpi_size - 1) / mpi_size;

      MatWrapper new_mat = MatWrapper::CreateAIJ(m_local, new_local_ncols, new_global_col_capacity);

      if (global_ncols > 0) {
        PetscInt Istart, Iend;
        MatGetOwnershipRange(mat, &Istart, &Iend);
        for (PetscInt i = Istart; i < Iend; ++i) {
          PetscInt ncols_in_row;
          const PetscInt* cols;
          const PetscScalar* vals;
          PetscCallAbort(PETSC_COMM_WORLD, MatGetRow(mat, i, &ncols_in_row, &cols, &vals));
          if (ncols_in_row > 0) {
            PetscCallAbort(PETSC_COMM_WORLD, MatSetValues(new_mat, 1, &i, ncols_in_row, cols, vals, INSERT_VALUES));
          }
          PetscCallAbort(PETSC_COMM_WORLD, MatRestoreRow(mat, i, &ncols_in_row, &cols, &vals));
        }
      }

      PetscCallAbort(PETSC_COMM_WORLD, MatAssemblyBegin(new_mat, MAT_FINAL_ASSEMBLY));
      PetscCallAbort(PETSC_COMM_WORLD, MatAssemblyEnd(new_mat, MAT_FINAL_ASSEMBLY));

      mat = std::move(new_mat);
      PetscCallAbort(PETSC_COMM_WORLD, MatGetSize(mat, NULL, &global_col_capacity));
    }
  }

  operator MatWrapper&() { return mat; }
  operator const MatWrapper&() const { return mat; }
  operator Mat() const { return mat; }
  PetscInt getNumCols() const { return global_ncols; }

  void updateSize(PetscInt ncols_delta) {
    global_ncols += ncols_delta;
  }
};
