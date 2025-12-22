# Unit Tests

This directory contains unit tests for the ProliferatingCellCollectives project using Google Test framework.


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

