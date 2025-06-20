cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(ProliferatingCellCollectives)

set(CMAKE_CXX_STANDARD 23)

set(CELL_COLLECTIVES_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}) 

# append custom modules to the module path
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

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
