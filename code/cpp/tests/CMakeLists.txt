# Enable code coverage if requested
option(ENABLE_COVERAGE "Enable code coverage" OFF)
if(ENABLE_COVERAGE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage -fprofile-arcs -ftest-coverage")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
endif()

# Find all test source files
file(GLOB TEST_SOURCES "*.cpp")

# Create test executable
add_executable(unit_tests ${TEST_SOURCES})

# Link against Google Test, the library, and other dependencies
target_link_libraries(unit_tests
    PRIVATE
    GTest::gtest
    cellcollectives_lib
)

# Include test directories
target_include_directories(unit_tests
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CELL_COLLECTIVES_SOURCE_DIR}/src/
)

# Discover tests for CTest
include(GoogleTest)
gtest_discover_tests(unit_tests)
