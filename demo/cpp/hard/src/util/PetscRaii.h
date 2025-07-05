#pragma once

#include <petsc.h>

#include <utility>

#include "MPIUtil.h"

template <typename T>
struct PetscObjectTraits;

template <typename T, typename Traits = PetscObjectTraits<T>>
class SmartPetscObject {
 private:
  T obj_ = nullptr;

 public:
  SmartPetscObject() = default;

  explicit SmartPetscObject(T obj) : obj_(obj) {}

  // --- Rule of Five: Manage ownership correctly ---

  // 1. Destructor: This is the core of RAII.
  ~SmartPetscObject() {
    if (obj_ != nullptr) {
      Traits::Destroy(&obj_);
      obj_ = nullptr;
    }
  }

  // 2. Copy constructor is deleted.
  SmartPetscObject(const SmartPetscObject&) = delete;

  // 3. Copy assignment is deleted for the same reason.
  SmartPetscObject& operator=(const SmartPetscObject&) = delete;

  // 4. Move constructor.
  SmartPetscObject(SmartPetscObject&& other) noexcept : obj_(other.obj_) {
    // Leave the source object in a valid but null state.
    other.obj_ = nullptr;
  }

  // 5. Move assignment.
  SmartPetscObject& operator=(SmartPetscObject&& other) noexcept {
    if (this != &other) {
      if (obj_ != nullptr) {
        Traits::Destroy(&obj_);
      }
      obj_ = other.obj_;
      other.obj_ = nullptr;
    }
    return *this;
  }

  // --- Usability Methods ---

  // Implicit conversion to the underlying PETSc type.
  // This allows you to pass the wrapper directly to PETSc functions.
  // Example: MatView(my_wrapped_mat, PETSC_VIEWER_STDOUT_WORLD);
  operator T() const {
    return obj_;
  }

  // Get a reference to the internal pointer.
  // This is essential for creation functions like MatCreate(comm, &mat).
  // Usage: MatCreate(PETSC_COMM_WORLD, my_wrapper.get_ref());
  T* get_ref() {
    if (obj_ != nullptr) {
      Traits::Destroy(&obj_);
      obj_ = nullptr;
    }
    return &obj_;
  }

  // Check if the wrapper holds a valid (non-null) object.
  explicit operator bool() const {
    return obj_ != nullptr;
  }
};

// --- Traits Specializations for different PETSc types ---

template <>
struct PetscObjectTraits<Vec> {
  static void Destroy(Vec* obj) { VecDestroy(obj); }
};

template <>
struct PetscObjectTraits<Mat> {
  static void Destroy(Mat* obj) { MatDestroy(obj); }
};

template <>
struct PetscObjectTraits<ISLocalToGlobalMapping> {
  static void Destroy(ISLocalToGlobalMapping* obj) { ISLocalToGlobalMappingDestroy(obj); }
};

template <>
struct PetscObjectTraits<IS> {
  static void Destroy(IS* obj) { ISDestroy(obj); }
};

class VecSubViewWrapper {
 private:
  Vec parent_ = nullptr;
  IS is_ = nullptr;
  Vec sub_vec_ = nullptr;
  bool owns_is_ = false;

 public:
  VecSubViewWrapper(Vec parent, IS is, bool transfer_is_ownership = false) : parent_(parent), is_(is), owns_is_(transfer_is_ownership) {
    PetscCallAbort(PETSC_COMM_WORLD, VecGetSubVector(parent_, is_, &sub_vec_));
  }
  ~VecSubViewWrapper() {
    if (sub_vec_ != nullptr) {
      PetscCallAbort(PETSC_COMM_WORLD, VecRestoreSubVector(parent_, is_, &sub_vec_));
    }
    if (owns_is_ && is_ != nullptr) {
      PetscCallAbort(PETSC_COMM_WORLD, ISDestroy(&is_));
    }
  }

  // Rule of Five to prevent copying and manage moves
  VecSubViewWrapper(const VecSubViewWrapper&) = delete;
  VecSubViewWrapper& operator=(const VecSubViewWrapper&) = delete;

  VecSubViewWrapper(VecSubViewWrapper&& other) noexcept : parent_(other.parent_), is_(other.is_), sub_vec_(other.sub_vec_), owns_is_(other.owns_is_) {
    other.parent_ = nullptr;
    other.is_ = nullptr;
    other.sub_vec_ = nullptr;
    other.owns_is_ = false;
  }

  VecSubViewWrapper& operator=(VecSubViewWrapper&& other) noexcept {
    if (this != &other) {
      if (sub_vec_ != nullptr) {
        PetscCallAbort(PETSC_COMM_WORLD, VecRestoreSubVector(parent_, is_, &sub_vec_));
      }
      if (owns_is_ && is_ != nullptr) {
        PetscCallAbort(PETSC_COMM_WORLD, ISDestroy(&is_));
      }
      parent_ = other.parent_;
      is_ = other.is_;
      sub_vec_ = other.sub_vec_;
      owns_is_ = other.owns_is_;
      other.parent_ = nullptr;
      other.is_ = nullptr;
      other.sub_vec_ = nullptr;
      other.owns_is_ = false;
    }
    return *this;
  }

  operator Vec() const { return sub_vec_; }
};

class MatWrapper : public SmartPetscObject<Mat, PetscObjectTraits<Mat>> {
 public:
  using SmartPetscObject<Mat, PetscObjectTraits<Mat>>::SmartPetscObject;
  static MatWrapper CreateEmpty(PetscInt local_rows);
  static MatWrapper CreateAIJ(PetscInt local_rows, PetscInt local_cols);
  static MatWrapper CreateAIJ(PetscInt local_rows, PetscInt local_cols, PetscInt global_cols);

  PetscInt getRows() const {
    PetscInt rows;
    PetscCallAbort(PETSC_COMM_WORLD, MatGetLocalSize(*this, &rows, NULL));
    return rows;
  }

  PetscInt getCols() const {
    PetscInt cols;
    PetscCallAbort(PETSC_COMM_WORLD, MatGetLocalSize(*this, NULL, &cols));
    return cols;
  }
};

class VecWrapper : public SmartPetscObject<Vec, PetscObjectTraits<Vec>> {
 public:
  // Inherit constructors
  using SmartPetscObject<Vec, PetscObjectTraits<Vec>>::SmartPetscObject;

  // Static factory methods for convenience
  static VecWrapper Like(const VecWrapper& other);
  static VecWrapper FromMat(const MatWrapper& mat);
  static VecWrapper FromMatRows(const MatWrapper& mat);
  static VecWrapper CreateEmpty();
  static VecWrapper Create(PetscInt local_size);
  static VecWrapper CreateWithGlobalSize(PetscInt global_size);
};

class ISWrapper : public SmartPetscObject<IS, PetscObjectTraits<IS>> {
 public:
  using SmartPetscObject<IS, PetscObjectTraits<IS>>::SmartPetscObject;
};

inline VecWrapper VecWrapper::Like(const VecWrapper& other) {
  VecWrapper new_obj;
  PetscCallAbort(PETSC_COMM_WORLD, VecDuplicate(other, new_obj.get_ref()));
  return new_obj;
}

inline VecWrapper VecWrapper::FromMat(const MatWrapper& mat) {
  VecWrapper new_obj;
  PetscCallAbort(PETSC_COMM_WORLD, MatCreateVecs(mat, NULL, new_obj.get_ref()));
  return new_obj;
}

inline VecWrapper VecWrapper::FromMatRows(const MatWrapper& mat) {
  VecWrapper new_obj;
  PetscCallAbort(PETSC_COMM_WORLD, MatCreateVecs(mat, new_obj.get_ref(), NULL));
  return new_obj;
}

inline VecWrapper VecWrapper::CreateEmpty() {
  VecWrapper new_obj;
  PetscCallAbort(PETSC_COMM_WORLD, VecCreate(PETSC_COMM_WORLD, new_obj.get_ref()));
  PetscCallAbort(PETSC_COMM_WORLD, VecSetSizes(new_obj, 0, PETSC_DETERMINE));
  PetscCallAbort(PETSC_COMM_WORLD, VecSetType(new_obj, VECSTANDARD));
  PetscCallAbort(PETSC_COMM_WORLD, VecSetFromOptions(new_obj));
  return new_obj;
}

inline VecWrapper VecWrapper::Create(PetscInt local_size) {
  VecWrapper new_obj;
  PetscCallAbort(PETSC_COMM_WORLD, VecCreate(PETSC_COMM_WORLD, new_obj.get_ref()));
  PetscCallAbort(PETSC_COMM_WORLD, VecSetSizes(new_obj, local_size, PETSC_DETERMINE));
  PetscCallAbort(PETSC_COMM_WORLD, VecSetFromOptions(new_obj));
  return new_obj;
}

inline VecWrapper VecWrapper::CreateWithGlobalSize(PetscInt global_size) {
  VecWrapper new_obj;
  PetscCallAbort(PETSC_COMM_WORLD, VecCreate(PETSC_COMM_WORLD, new_obj.get_ref()));
  PetscCallAbort(PETSC_COMM_WORLD, VecSetSizes(new_obj, PETSC_DECIDE, global_size));
  PetscCallAbort(PETSC_COMM_WORLD, VecSetFromOptions(new_obj));
  PetscCallAbort(PETSC_COMM_WORLD, VecSetUp(new_obj));
  return new_obj;
}

inline MatWrapper MatWrapper::CreateEmpty(PetscInt local_rows) {
  MatWrapper new_obj;
  PetscCallAbort(PETSC_COMM_WORLD, MatCreate(PETSC_COMM_WORLD, new_obj.get_ref()));
  PetscCallAbort(PETSC_COMM_WORLD, MatSetSizes(new_obj, local_rows, 0, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCallAbort(PETSC_COMM_WORLD, MatSetType(new_obj, MATMPIAIJ));
  PetscCallAbort(PETSC_COMM_WORLD, MatSetFromOptions(new_obj));
  PetscCallAbort(PETSC_COMM_WORLD, MatAssemblyBegin(new_obj, MAT_FINAL_ASSEMBLY));
  PetscCallAbort(PETSC_COMM_WORLD, MatAssemblyEnd(new_obj, MAT_FINAL_ASSEMBLY));
  return new_obj;
}

inline MatWrapper MatWrapper::CreateAIJ(PetscInt local_rows, PetscInt local_cols) {
  MatWrapper new_obj;
  PetscCallAbort(PETSC_COMM_WORLD, MatCreate(PETSC_COMM_WORLD, new_obj.get_ref()));
  PetscCallAbort(PETSC_COMM_WORLD, MatSetSizes(new_obj, local_rows, local_cols, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCallAbort(PETSC_COMM_WORLD, MatSetType(new_obj, MATAIJ));
  PetscCallAbort(PETSC_COMM_WORLD, MatSetFromOptions(new_obj));
  PetscCallAbort(PETSC_COMM_WORLD, MatSetUp(new_obj));
  return new_obj;
}

inline MatWrapper MatWrapper::CreateAIJ(PetscInt local_rows, PetscInt local_cols, PetscInt global_cols) {
  MatWrapper new_obj;
  PetscCallAbort(PETSC_COMM_WORLD, MatCreate(PETSC_COMM_WORLD, new_obj.get_ref()));
  PetscCallAbort(PETSC_COMM_WORLD, MatSetSizes(new_obj, local_rows, local_cols, PETSC_DETERMINE, global_cols));
  PetscCallAbort(PETSC_COMM_WORLD, MatSetType(new_obj, MATAIJ));
  PetscCallAbort(PETSC_COMM_WORLD, MatSetFromOptions(new_obj));
  PetscCallAbort(PETSC_COMM_WORLD, MatSetUp(new_obj));
  return new_obj;
}
