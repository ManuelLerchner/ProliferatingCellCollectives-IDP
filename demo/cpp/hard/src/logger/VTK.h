#pragma once

#include <array>
#include <chrono>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

// Forward declarations
class Particle;
class Constraint;

namespace vtk {

/**
 * @brief Represents different VTK data types
 */
enum class DataType {
  Float32,
  Float64,
  Int32,
  UInt8,
  String
};

/**
 * @brief Location where data is associated
 */
enum class DataLocation {
  Point,
  Cell,
  Field
};

/**
 * @brief Represents a field to be written to VTK files
 */
struct VTKField {
  std::string name;
  std::vector<double> data;
  int components;
  DataType data_type;
  DataLocation location;

  VTKField(const std::string& name, const std::vector<double>& data,
           int components = 1, DataType type = DataType::Float32,
           DataLocation loc = DataLocation::Point)
      : name(name), data(data), components(components), data_type(type), location(loc) {}

  VTKField(const std::string& name, const std::vector<std::array<double, 3>>& vector_data,
           DataType type = DataType::Float32, DataLocation loc = DataLocation::Point)
      : name(name), components(3), data_type(type), location(loc) {
    data.reserve(vector_data.size() * 3);
    for (const auto& vec : vector_data) {
      data.insert(data.end(), vec.begin(), vec.end());
    }
  }

  VTKField(const std::string& name, const std::vector<int>& int_data,
           DataType type = DataType::Int32, DataLocation loc = DataLocation::Point)
      : name(name), components(1), data_type(type), location(loc) {
    data.reserve(int_data.size());
    for (int val : int_data) {
      data.push_back(static_cast<double>(val));
    }
  }
};

/**
 * @brief Container for field data (global simulation metadata)
 */
struct VTKFieldData {
  struct FieldEntry {
    std::string value_str;
    double value_num;
    DataType type;
    bool is_numeric;
  };

  std::map<std::string, FieldEntry> fields;

  void addField(const std::string& name, double value) {
    fields[name] = {std::to_string(value), value, DataType::Float64, true};
  }

  void addField(const std::string& name, int value) {
    fields[name] = {std::to_string(value), static_cast<double>(value), DataType::Int32, true};
  }

  void addField(const std::string& name, const std::string& value) {
    fields[name] = {value, 0.0, DataType::String, false};
  }
};

/**
 * @brief Container for VTK geometry data
 */
struct VTKGeometry {
  std::vector<std::array<double, 3>> positions;
  std::vector<int> connectivity;
  std::vector<int> offsets;
  std::vector<uint8_t> cell_types;

  VTKGeometry(const std::vector<std::array<double, 3>>& pos) : positions(pos) {
    // Default to vertex cells
    int n_points = positions.size();
    connectivity.resize(n_points);
    offsets.resize(n_points);
    cell_types.resize(n_points);

    for (int i = 0; i < n_points; ++i) {
      connectivity[i] = i;
      offsets[i] = i + 1;
      cell_types[i] = 1;  // VTK_VERTEX
    }
  }
};

/**
 * @brief Abstract base class for extracting simulation data for VTK output
 */
class VTKDataExtractor {
 public:
  virtual ~VTKDataExtractor() = default;

  virtual VTKGeometry extractGeometry(const void* state) = 0;
  virtual std::vector<VTKField> extractPointData(const void* state) = 0;
  virtual std::vector<VTKField> extractCellData(const void* state) = 0;
  virtual VTKFieldData extractFieldData(const void* state, int timestep, double dt, double elapsed_time, bool is_substep) = 0;
};

/**
 * @brief Simulation state container for particles
 */
struct ParticleSimulationState {
  std::vector<Particle> particles;
  std::vector<Constraint> constraints;
  std::vector<std::array<double, 3>> forces;
  std::vector<std::array<double, 3>> torques;
  std::vector<std::array<double, 3>> velocities_linear;
  std::vector<std::array<double, 3>> velocities_angular;
  std::vector<double> impedance;
  double residual;
  int constraint_iterations;
  int bbpgd_iterations;
  double l0;
};

/**
 * @brief Data extractor for particle simulations
 */
class ParticleDataExtractor : public VTKDataExtractor {
 public:
  VTKGeometry extractGeometry(const void* state) override;
  std::vector<VTKField> extractPointData(const void* state) override;
  std::vector<VTKField> extractCellData(const void* state) override;
  VTKFieldData extractFieldData(const void* state, int timestep, double dt, double elapsed_time, bool is_substep) override;
};

/**
 * @brief Data extractor for constraint visualization
 */
class ConstraintDataExtractor : public VTKDataExtractor {
 public:
  VTKGeometry extractGeometry(const void* state) override;
  std::vector<VTKField> extractPointData(const void* state) override;
  std::vector<VTKField> extractCellData(const void* state) override;
  VTKFieldData extractFieldData(const void* state, int timestep, double dt, double elapsed_time, bool is_substep) override;
};

/**
 * @brief Generic logger for writing PVTU (parallel VTK Unstructured Grid) files
 */
class VTKLogger {
 public:
  VTKLogger(const std::string& output_dir, const std::string& prefix,
            std::unique_ptr<VTKDataExtractor> extractor);
  ~VTKLogger();

  void setDataExtractor(std::unique_ptr<VTKDataExtractor> extractor);
  void logTimestep(int frame_number, double dt, const void* state, bool is_substep);
  void clearDataFolder();

 private:
  void setupDirectories();
  std::string getVTUFilename(int frame_number) const;
  std::string getPVTUFilename(int frame_number) const;
  std::string dataTypeToString(DataType type) const;

  void addVTUFile(int frame_number, double dt, const VTKGeometry& geometry,
                  const std::vector<VTKField>& point_fields,
                  const std::vector<VTKField>& cell_fields,
                  const VTKFieldData& field_data);
  void writeTimestepPVTU(int frame_number, const std::vector<VTKField>& point_fields,
                         const std::vector<VTKField>& cell_fields, const VTKFieldData& field_data);

  void writeDataArray(std::ofstream& file, const VTKField& field, const std::string& indent);

  std::string output_dir_;
  std::string data_dir_;
  std::string prefix_;
  std::unique_ptr<VTKDataExtractor> data_extractor_;
  std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
};

/**
 * @brief Generic simulation logger using VTK output
 */
class SimulationLogger {
 public:
  SimulationLogger(std::unique_ptr<VTKLogger> vtk_logger, int log_every_n_iterations = 1);

  bool shouldLog(int iteration) const;
  void logTimestepComplete(double dt, const void* state);
  void logSubstep(double dt, const void* state);

 private:
  std::unique_ptr<VTKLogger> vtk_logger_;
  int current_timestep_ = 0;
  int frame_counter_ = 0;
  int log_every_n_iterations_;
};

// Factory functions
std::unique_ptr<SimulationLogger> createParticleLogger(const std::string& output_dir, int log_every_n_iterations = 1);
std::unique_ptr<SimulationLogger> createConstraintLogger(const std::string& output_dir, int log_every_n_iterations = 1);

}  // namespace vtk
