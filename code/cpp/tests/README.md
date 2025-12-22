# Unit Tests

This directory contains unit tests for the ProliferatingCellCollectives project using Google Test framework.

## Test Coverage

The test suite includes comprehensive tests for:

- **ArrayMath** (26 tests): Vector operations including add, subtract, multiply, dot product, cross product, normalization, distances, and norms
- **Quaternion** (16 tests): Quaternion operations including multiplication, Euler angle conversion, vector rotation, and direction vectors
- **SpherocylinderCell** (14 tests): Segment-segment distance calculations using DCPQuery for collision detection
- **Particle** (20 tests): Particle construction, state management, getters/setters, volume calculations, and constraint tracking
- **SpatialGrid** (15 tests): Construction, particle insertion, collision pair detection, domain boundaries, negative coordinates, multiple particles per cell
- **CollisionDetector** (16 tests): Collision detector construction, bounds updates, particle endpoint calculations, rotations, symmetry verification, 3D transformations
- **BBPGD Solver** (6 tests): Result structure tests, iteration tracking, convergence (full solver tests require complex PETSc/MPI setup)
- **Constraint** (16 tests): Constraint construction, signed distances, normal vectors, stress values, locality flags, contact points, particle IDs

**Total: 127 tests**

## Building and Running Tests

### Prerequisites

```bash
sudo apt-get install -y cmake g++ libopenmpi-dev openmpi-bin petsc-dev lcov
```

### Build Tests

```bash
cd code/cpp
mkdir -p build
cd build
cmake ..
make -j$(nproc) unit_tests
```

### Run Tests

```bash
./tests/unit_tests
```

### Build with Coverage

```bash
cd code/cpp
mkdir -p build
cd build
cmake -DENABLE_COVERAGE=ON ..
make -j$(nproc) unit_tests
./tests/unit_tests
```

### Generate Coverage Report

```bash
# Capture coverage data
lcov --capture --directory . --output-file coverage.info --ignore-errors mismatch

# Filter out system files and test files
lcov --remove coverage.info '/usr/*' '*/build/_deps/*' '*/tests/*' \
     --output-file coverage_filtered.info --ignore-errors unused

# Generate HTML report
genhtml coverage_filtered.info --output-directory coverage_report

# View the report
xdg-open coverage_report/index.html
```

## Test Structure

Each test file follows this structure:

- `test_array_math.cpp`: Tests for array/vector mathematical operations
- `test_quaternion.cpp`: Tests for quaternion operations and rotations
- `test_spherocylinder.cpp`: Tests for geometric distance calculations
- `test_particle.cpp`: Tests for particle state management and physics
- `test_spatial_grid.cpp`: Tests for spatial partitioning and collision pair finding
- `test_collision_detector.cpp`: Tests for collision detection and particle endpoint calculations
- `test_bbpgd_solver.cpp`: Tests for BBPGD solver result structures
- `test_constraint.cpp`: Tests for constraint creation and management

## CI/CD

Tests are automatically run on every push and pull request via GitHub Actions. The workflow:

1. Builds the project with coverage enabled
2. Runs all unit tests
3. Generates coverage reports
4. Uploads coverage artifacts
5. Comments coverage statistics on pull requests

See `.github/workflows/test.yml` for details.

## Adding New Tests

To add new tests:

1. Create a new test file in this directory (e.g., `test_myclass.cpp`)
2. Include Google Test and the class to test:
   ```cpp
   #include <gtest/gtest.h>
   #include "path/to/MyClass.h"
   ```
3. Write your tests:
   ```cpp
   TEST(MyClassTest, TestName) {
       // Arrange
       MyClass obj;
       
       // Act
       auto result = obj.doSomething();
       
       // Assert
       EXPECT_EQ(result, expectedValue);
   }
   ```
4. The test will automatically be included in the build via `file(GLOB TEST_SOURCES "*.cpp")` in `CMakeLists.txt`
5. Rebuild and run: `make unit_tests && ./tests/unit_tests`

## Test Guidelines

- Write focused tests that test one thing at a time
- Use descriptive test names that explain what is being tested
- Follow the Arrange-Act-Assert pattern
- Test both normal cases and edge cases
- Use appropriate assertion macros:
  - `EXPECT_EQ` / `ASSERT_EQ` for equality
  - `EXPECT_NEAR` / `ASSERT_NEAR` for floating-point comparisons
  - `EXPECT_TRUE` / `EXPECT_FALSE` for boolean conditions
  - `EXPECT_THROW` for exception testing
- Aim for high code coverage while maintaining test quality
