#pragma once

#include <stdexcept>
#include <vector>

#include "Config.h"
#include "PetscRaii.h"

class DynamicVecWrapper {
 private:
  VecWrapper vec;
  PetscInt local_capacity;
  int size;

  double growth_factor;

 public:
  DynamicVecWrapper(int capacity, double growth_factor_) {
    growth_factor = growth_factor_;
    local_capacity = capacity;
    size = 0;
    vec = VecWrapper::Create(capacity);
  }

  bool ensureCapacity(PetscInt additional_items) {
    PetscInt required_local_capacity = size + additional_items;

    auto max_required_capacity = globalReduce(required_local_capacity, MPI_MAX);

    if (local_capacity < max_required_capacity) {
      PetscInt new_local_capacity = max_required_capacity * growth_factor;

      PetscPrintf(PETSC_COMM_WORLD, "Resizing vec from %d to %d\n", local_capacity, new_local_capacity);

      VecWrapper new_vec = VecWrapper::Create(new_local_capacity);

      vec = std::move(new_vec);
      local_capacity = new_local_capacity;
      size = 0;
      return true;
    }
    return false;
  }

  operator VecWrapper&() { return vec; }
  operator const VecWrapper&() const { return vec; }
  operator Vec() const { return vec; }

  PetscInt getSize() const { return size; }

  PetscInt getCapacity() const { return local_capacity; }

  void incrementSize(PetscInt size_delta) {
    size += size_delta;
  }
};

class DynamicMatWrapper {
 private:
  MatWrapper mat;
  PetscInt local_rows;
  PetscInt local_capacity;
  PetscInt size;

  double growth_factor;

 public:
  DynamicMatWrapper(PetscInt local_rows_, int capacity_, double growth_factor_) {
    growth_factor = growth_factor_;
    local_capacity = capacity_;
    local_rows = local_rows_;
    size = 0;

    mat = MatWrapper::CreateAIJ(local_rows, capacity_);
    MatSetOption(mat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
  }

  bool ensureCapacity(PetscInt additional_cols) {
    PetscInt required_local_capacity = size + additional_cols;

    auto max_required_capacity = globalReduce(required_local_capacity, MPI_MAX);

    if (local_capacity < max_required_capacity) {
      PetscInt new_local_capacity = max_required_capacity * growth_factor;

      MatWrapper new_mat = MatWrapper::CreateAIJ(local_rows, new_local_capacity);
      MatSetOption(new_mat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);

      mat = std::move(new_mat);
      local_capacity = new_local_capacity;
      size = 0;
      return true;
    }
    return false;
  }

  operator MatWrapper&() { return mat; }
  operator const MatWrapper&() const { return mat; }
  operator Mat() const { return mat; }

  PetscInt getSize() const { return size; }

  PetscInt getCapacity() const { return local_capacity; }

  void incrementSize(PetscInt size_delta) {
    size += size_delta;
  }
};
