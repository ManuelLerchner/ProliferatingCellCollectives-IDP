cmake_minimum_required(VERSION 3.20.3)


project(ddm_library)

# PkgConfig 
find_package(PkgConfig)

# PETSc
if (PKG_CONFIG_FOUND)
    pkg_check_modules(PETSC PETSc)
endif()

if (PETSC_FOUND)
    list(APPEND COMPILE_OPTIONS ${PETSC_CFLAGS})

    include_directories(${PETSC_INCLUDE_DIRS})
    
    set(LINK_FLAGS "${LINK_FLAGS} ${PETSC_LDFLAGS}")   
    
    list(APPEND LIBRARIES ${PETSC_LINK_LIBRARIES})   
    
    set(CMAKE_REQUIRED_FLAGS ${PETSC_CFLAGS})   
    
    set(CMAKE_REQUIRED_INCLUDES "${PETSC_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "PETSc not found. Please install PETSc or set the PETSC_DIR environment variable.")
endif()
