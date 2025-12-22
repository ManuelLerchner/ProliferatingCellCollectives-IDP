# Find all source files except main.cpp
file(
    GLOB_RECURSE
    MY_SRC
    "*.cpp"
    "*.h"
)

# Remove main.cpp from the library sources
list(FILTER MY_SRC EXCLUDE REGEX ".*main\\.cpp$")

include(petsc)
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Create a library with all source files (excluding main.cpp)
add_library(cellcollectives_lib STATIC ${MY_SRC})

if(OpenMP_CXX_FOUND)
    set(OpenMP_STATIC_LIB ON)
    target_link_libraries(cellcollectives_lib PUBLIC OpenMP::OpenMP_CXX)
endif()

# Add dependency on petsc target
add_dependencies(cellcollectives_lib petsc)

target_include_directories(
    cellcollectives_lib
    PUBLIC
    ${CELL_COLLECTIVES_SOURCE_DIR}/src/
    ${PETSC_INCLUDE_DIRS}
    ${MPI_CXX_INCLUDE_DIRS}
    ${OpenMP_CXX_INCLUDE_DIRS}
)

target_link_libraries(
    cellcollectives_lib
    PUBLIC
    ${PETSC_LIBRARIES}
    ${MPI_CXX_LIBRARIES}
    ${MPI_CXX_EXTRA_LIBRARIES}
    OpenMP::OpenMP_CXX
)

target_compile_definitions(
    cellcollectives_lib
    PUBLIC
    ${MPI_CXX_COMPILE_DEFINITIONS}
    ${OpenMP_CXX_COMPILE_DEFINITIONS}
)

# Create executable that links to the library
add_executable(cellcollectives main.cpp)
target_link_libraries(cellcollectives PRIVATE cellcollectives_lib)
