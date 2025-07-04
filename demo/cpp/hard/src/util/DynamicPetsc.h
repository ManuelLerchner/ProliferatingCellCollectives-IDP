#pragma once

#include <stdexcept>
#include <vector>

#include "Config.h"
#include "PetscRaii.h"

class DynamicVecWrapper {
 private:
  VecWrapper vec;
  PetscInt size;
  PetscInt capacity;
  double growth_factor;
  int min_buffer;

 public:
  DynamicVecWrapper(SolverConfig solver_config) {
    growth_factor = solver_config.growth_factor;
    min_buffer = solver_config.min_preallocation_size;
    size = 0;
    capacity = min_buffer;
    vec = VecWrapper::Create(min_buffer);
  }

  void ensureCapacity(PetscInt additional_items) {
    PetscInt required_size = size + additional_items;
    if (capacity < required_size) {
      PetscInt new_capacity = std::max(required_size, static_cast<PetscInt>(capacity * growth_factor));
      PetscPrintf(PETSC_COMM_WORLD, "\nLOG: DynamicVecWrapper reallocating from capacity %d to %d (size: %d, required: %d)\n", capacity, new_capacity, size, required_size);

      VecWrapper new_vec = VecWrapper::Create(new_capacity);
      if (size > 0) {
        IS is_from, is_to;
        PetscCallAbort(PETSC_COMM_WORLD, ISCreateStride(PETSC_COMM_WORLD, size, 0, 1, &is_from));
        PetscCallAbort(PETSC_COMM_WORLD, ISCreateStride(PETSC_COMM_WORLD, size, 0, 1, &is_to));
        VecScatter scatter;
        PetscCallAbort(PETSC_COMM_WORLD, VecScatterCreate(vec, is_from, new_vec, is_to, &scatter));
        PetscCallAbort(PETSC_COMM_WORLD, VecScatterBegin(scatter, vec, new_vec, INSERT_VALUES, SCATTER_FORWARD));
        PetscCallAbort(PETSC_COMM_WORLD, VecScatterEnd(scatter, vec, new_vec, INSERT_VALUES, SCATTER_FORWARD));
        PetscCallAbort(PETSC_COMM_WORLD, VecScatterDestroy(&scatter));
        PetscCallAbort(PETSC_COMM_WORLD, ISDestroy(&is_from));
        PetscCallAbort(PETSC_COMM_WORLD, ISDestroy(&is_to));
      }
      vec = std::move(new_vec);
      capacity = new_capacity;
    }
  }

  operator VecWrapper&() { return vec; }
  operator const VecWrapper&() const { return vec; }
  operator Vec() const { return vec; }
  PetscInt getSize() const { return size; }

  void updateSize(PetscInt size_delta) {
    size += size_delta;
  }

  void copyTo(VecWrapper& other) const {
    PetscCallAbort(PETSC_COMM_WORLD, VecCopy(vec, other));
  }
};

class DynamicMatWrapper {
 private:
  MatWrapper mat;
  PetscInt ncols;
  PetscInt capacity;
  double growth_factor;
  int min_buffer;

 public:
  DynamicMatWrapper(SolverConfig solver_config, PetscInt local_rows) {
    growth_factor = solver_config.growth_factor;
    min_buffer = solver_config.min_preallocation_size;
    ncols = 0;
    capacity = min_buffer;
    mat = MatWrapper::CreateAIJ(local_rows, min_buffer);
  }

  void ensureCapacity(PetscInt additional_cols) {
    PetscInt required_cols = ncols + additional_cols;
    if (capacity < required_cols) {
      PetscInt new_capacity = std::max(required_cols, static_cast<PetscInt>(capacity * growth_factor));
      PetscPrintf(PETSC_COMM_WORLD, "\nLOG: DynamicMatWrapper reallocating from capacity %d to %d (cols: %d, required: %d )\n", capacity, new_capacity, ncols, required_cols);

      PetscInt m_local;
      MatGetLocalSize(mat, &m_local, NULL);

      MatWrapper new_mat = MatWrapper::CreateAIJ(m_local, new_capacity);

      if (ncols > 0) {
        PetscInt Istart, Iend;
        MatGetOwnershipRange(mat, &Istart, &Iend);
        for (PetscInt i = Istart; i < Iend; ++i) {
          PetscInt ncols;
          const PetscInt* cols;
          const PetscScalar* vals;
          PetscCallAbort(PETSC_COMM_WORLD, MatGetRow(mat, i, &ncols, &cols, &vals));
          if (ncols > 0) {
            PetscCallAbort(PETSC_COMM_WORLD, MatSetValues(new_mat, 1, &i, ncols, cols, vals, INSERT_VALUES));
          }
          PetscCallAbort(PETSC_COMM_WORLD, MatRestoreRow(mat, i, &ncols, &cols, &vals));
        }
      }

      PetscCallAbort(PETSC_COMM_WORLD, MatAssemblyBegin(new_mat, MAT_FINAL_ASSEMBLY));
      PetscCallAbort(PETSC_COMM_WORLD, MatAssemblyEnd(new_mat, MAT_FINAL_ASSEMBLY));

      mat = std::move(new_mat);
      capacity = new_capacity;
    }
  }

  operator MatWrapper&() { return mat; }
  operator const MatWrapper&() const { return mat; }
  operator Mat() const { return mat; }
  PetscInt getNumCols() const { return ncols; }

  void updateSize(PetscInt ncols_delta) {
    ncols += ncols_delta;
  }
};

size_t vecSize(const VecWrapper& vec) {
  PetscInt size;
  PetscCallAbort(PETSC_COMM_WORLD, VecGetSize(vec, &size));
  return size;
}