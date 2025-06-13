# Find all source files except main.cpp
file(
    GLOB_RECURSE
    MY_SRC
    "*.cpp"
    "*.h"
)

# Remove main.cpp from the list if you want to specify it separately
list(REMOVE_ITEM MY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

# Create executable with main.cpp and all other sources
add_executable(cellcollectives main.cpp ${MY_SRC})

include(petsc)
find_package(MPI REQUIRED)

# Add dependency on petsc target
add_dependencies(cellcollectives petsc)

target_include_directories(
    cellcollectives
    PUBLIC
    ${CELL_COLLECTIVES_SOURCE_DIR}/src/
    ${PETSC_INCLUDE_DIRS}
    ${MPI_CXX_INCLUDE_DIRS}
)

target_link_libraries(
    cellcollectives
    PUBLIC
    ${PETSC_LIBRARIES}
    ${MPI_CXX_LIBRARIES}
    ${MPI_CXX_EXTRA_LIBRARIES}
)

target_compile_definitions(
    cellcollectives
    PRIVATE
    ${MPI_CXX_COMPILE_DEFINITIONS}
)
