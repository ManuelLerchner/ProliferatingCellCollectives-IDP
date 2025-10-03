cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(ProliferatingCellCollectives)

set(CMAKE_CXX_STANDARD 23)

set(CELL_COLLECTIVES_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}) 

# append custom modules to the module path
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=undefined -fsanitize=address -fsanitize=leak")
set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=undefined -fsanitize=address -fsanitize=leak ")


set(CMAKE_CXX_FLAGS  "-O3 -ffast-math -DNDEBUG -std=c++20 -march=native -fpic -ftree-vectorize -flto -funroll-loops -falign-functions=32 -fassociative-math -freciprocal-math ")

# let ccmake and cmake-gui offer the default build type options
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug;Release;RelWithDebInfo;MinSizeRel")

# set Release as the default build type if it is not yet set.
if (NOT CMAKE_BUILD_TYPE)
    set(
            CMAKE_BUILD_TYPE
            "Release"
            CACHE
            STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
            FORCE
    )
endif ()

enable_testing()

# autopas
add_subdirectory(src)
