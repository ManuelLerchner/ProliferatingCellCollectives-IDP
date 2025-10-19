# Find all source files except main.cpp
file(
    GLOB_RECURSE
    MY_SRC
    "*.cpp"
    "*.h"
)

# Create executable with main.cpp and all other sources
add_executable(cellcollectives main.cpp ${MY_SRC})

include(petsc)
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    set(OpenMP_STATIC_LIB ON)
    target_link_libraries(cellcollectives PUBLIC OpenMP::OpenMP_CXX)
endif()

# Add dependency on petsc target
add_dependencies(cellcollectives petsc)

target_include_directories(
    cellcollectives
    PUBLIC
    ${CELL_COLLECTIVES_SOURCE_DIR}/src/
    ${PETSC_INCLUDE_DIRS}
    ${MPI_CXX_INCLUDE_DIRS}
    ${OpenMP_CXX_INCLUDE_DIRS}
)

target_link_libraries(
    cellcollectives
    PUBLIC
    ${PETSC_LIBRARIES}
    ${MPI_CXX_LIBRARIES}
    ${MPI_CXX_EXTRA_LIBRARIES}
    OpenMP::OpenMP_CXX
)

target_compile_definitions(
    cellcollectives
    PRIVATE
    ${MPI_CXX_COMPILE_DEFINITIONS}
    ${OpenMP_CXX_COMPILE_DEFINITIONS}
)
