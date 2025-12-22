# Unit Tests

This directory contains unit tests for the ProliferatingCellCollectives project using Google Test framework.

## Test Coverage

The test suite includes comprehensive tests for:

- **ArrayMath** (26 tests): Vector operations including add, subtract, multiply, dot product, cross product, normalization, distances, and norms
- **Quaternion** (16 tests): Quaternion operations including multiplication, Euler angle conversion, vector rotation, and direction vectors
- **SpherocylinderCell** (14 tests): Segment-segment distance calculations using DCPQuery for collision detection
- **Particle** (20 tests): Particle construction, state management, getters/setters, volume calculations, and constraint tracking
- **SpatialGrid** (13 tests): Construction, particle insertion, collision pair detection, domain boundaries, negative coordinates, multiple particles per cell
- **CollisionDetector** (16 tests): Collision detector construction, bounds updates, particle endpoint calculations, rotations, symmetry verification, 3D transformations
- **BBPGD Solver** (15 tests): Result structure tests, iteration tracking, convergence testing (residual decrease, tolerance checks, monotonic decrease, early stopping), quadratic function minimization with actual BBPGD solver calls
- **Constraint** (16 tests): Constraint construction, signed distances, normal vectors, stress values, locality flags, contact points, particle IDs
- **MPI Tests** (11 tests): MPI initialization, rank coordination, global reduce operations (sum/min/max), vector reductions, cross-rank communication validation
- **PETSc Tests** (8 tests): VecWrapper operations (create, set/get, AXPY, norm, Like), MatWrapper creation, RAII semantics, move operations

**Total: 155 tests**

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

