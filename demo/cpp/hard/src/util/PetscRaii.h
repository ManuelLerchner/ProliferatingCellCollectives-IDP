#pragma once

#include <petsc.h>

#include <utility>

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
    if (obj_) {
      // The traits class knows the correct destroy function.
      Traits::destroy(&obj_);
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
      // Destroy the object we are currently holding, if any.
      if (obj_) {
        Traits::destroy(&obj_);
      }
      // Steal the object from the other wrapper.
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

  // Explicitly get the raw PETSc object.
  T get() const {
    return obj_;
  }

  // Get a reference to the internal pointer.
  // This is essential for creation functions like MatCreate(comm, &mat).
  // Usage: MatCreate(PETSC_COMM_WORLD, my_wrapper.get_ref());
  T* get_ref() {
    // If we already hold an object, destroy it before it's overwritten
    // to prevent memory leaks.
    if (obj_) {
      Traits::destroy(&obj_);
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
struct PetscObjectTraits<Mat> {
  static void destroy(Mat* obj) { MatDestroy(obj); }
};

template <>
struct PetscObjectTraits<Vec> {
  static void destroy(Vec* obj) { VecDestroy(obj); }
};

template <>
struct PetscObjectTraits<ISLocalToGlobalMapping> {
  static void destroy(ISLocalToGlobalMapping* obj) { ISLocalToGlobalMappingDestroy(obj); }
};

using MatWrapper = SmartPetscObject<Mat>;
using VecWrapper = SmartPetscObject<Vec>;
using ISLocalToGlobalMappingWrapper = SmartPetscObject<ISLocalToGlobalMapping>;