cmake_minimum_required(VERSION 3.20.3)

# Use system-installed PETSc via pkg-config
find_package(PkgConfig REQUIRED)
pkg_check_modules(PETSC REQUIRED PETSc)

# Create an empty target for compatibility with existing CMakeLists
add_custom_target(petsc)

# Set PETSc include and library directories from pkg-config
set(PETSC_INCLUDE_DIRS ${PETSC_INCLUDE_DIRS})
set(PETSC_LIBRARIES ${PETSC_LIBRARIES})

include_directories(${PETSC_INCLUDE_DIRS})
link_directories(${PETSC_LIBRARY_DIRS})

# Add PETSc definitions
add_definitions(-DPETSC_USE_EXTERN_CXX)

message(STATUS "Using system PETSc")
message(STATUS "PETSc include directories: ${PETSC_INCLUDE_DIRS}")
message(STATUS "PETSc library directories: ${PETSC_LIBRARY_DIRS}")
message(STATUS "PETSc libraries: ${PETSC_LIBRARIES}")