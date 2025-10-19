cmake_minimum_required(VERSION 3.20.3)

# Define the PETSc external project
include(ExternalProject)

set(PETSC_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/petsc")

# if debug is true, set PETSC_ARCH to arch-linux-c-debug
if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    set(PETSC_ARCH "arch-linux-c-debug")
else()
    set(PETSC_ARCH "arch-linux-c-opt")
endif()

set(PETSC_LIB_DIR "${PETSC_BUILD_DIR}/${PETSC_ARCH}/lib")
set(PETSC_LIBRARIES "${PETSC_LIB_DIR}/libpetsc.so") # or .a if static

# Check if PETSc is already built
if(EXISTS "${PETSC_LIBRARIES}")
    message(STATUS "PETSc library already exists at ${PETSC_LIBRARIES}")
    set(PETSC_ALREADY_BUILT TRUE)
else()
    set(PETSC_ALREADY_BUILT FALSE)
endif()

# print if library exists
message(STATUS "PETSc library exists: ${PETSC_LIBRARIES}")

# Download, configure, and build PETSc only if not already built
if(NOT PETSC_ALREADY_BUILT)
    ExternalProject_Add(petsc_external
        GIT_REPOSITORY https://gitlab.com/petsc/petsc.git
        GIT_TAG release
        SOURCE_DIR "${PETSC_BUILD_DIR}"
        PREFIX "petsc"
        CONFIGURE_COMMAND ./configure --with-debugging=$<IF:$<CONFIG:Debug>,1,0> --with-openmp=1 --with-openmp-kernels=1 --with-fc=0 --download-f2cblaslapack --with-shared-libraries=1
        BUILD_COMMAND make all
        INSTALL_COMMAND ""
        BUILD_IN_SOURCE 1
        LOG_DOWNLOAD ON
        LOG_CONFIGURE ON
        LOG_BUILD ON
        LOG_DIR "${CMAKE_CURRENT_BINARY_DIR}/logs"
        STAMP_DIR "${CMAKE_CURRENT_BINARY_DIR}/stamp"
    )

    # Create a custom target that can be used as a dependency
    add_custom_target(petsc DEPENDS petsc_external)
else()
    # Create an empty target if PETSc is already built
    add_custom_target(petsc)
endif()

# Set PETSC_DIR to the build directory
set(PETSC_DIR "${PETSC_BUILD_DIR}")

# Set PETSc include and library directories
set(PETSC_INCLUDE_DIRS "${PETSC_DIR}/include;${PETSC_DIR}/${PETSC_ARCH}/include")

include_directories(${PETSC_INCLUDE_DIRS})
link_directories(${PETSC_LIB_DIR})

# Optionally, add PETSc definitions
add_definitions(-DPETSC_USE_EXTERN_CXX)

# Add a custom command to check for PETSc library after build
add_custom_command(
    OUTPUT "${PETSC_LIBRARIES}"
    COMMAND ${CMAKE_COMMAND} -E echo "Checking for PETSc library..."
    COMMAND ${CMAKE_COMMAND} -E make_directory "${PETSC_LIB_DIR}"
    DEPENDS petsc
    COMMENT "Waiting for PETSc library to be built..."
)