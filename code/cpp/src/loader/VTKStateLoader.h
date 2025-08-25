#pragma once

#include <array>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "simulation/Particle.h"

namespace vtk {

struct LoadedState {
  std::vector<Particle> particles;
  double simulation_time_s;
  double dt_s;
  int step;
};

// Helper struct to store VTU data arrays
struct VTUDataArray {
  std::string name;
  std::string type;
  int num_components;
  std::vector<double> data;  // Store all numeric data as doubles for simplicity
};

class VTKStateLoader {
 public:
  VTKStateLoader(const std::string& vtk_directory);

  // Load state from a specific timestep
  LoadedState loadState(int timestep);

  // Load state from the latest timestep
  LoadedState loadLatestState();

  // Get the latest timestep number available
  int getLatestTimestep() const;

 private:
  std::string vtk_directory_;
  std::filesystem::path data_dir_;

  // Helper functions to load different components
  std::vector<Particle> loadParticles(int timestep);
  std::optional<int> findLatestTimestep() const;

  // Manual PVTU/VTU parsing helpers
  std::vector<std::string> getPieceFiles(const std::string& pvtu_file);
  std::map<std::string, VTUDataArray> readVTUFile(const std::string& vtu_file);
  std::map<std::string, double> readFieldData(const std::string& pvtu_file);

  // Helper to parse numeric values from string
  template <typename T>
  std::vector<T> parseNumericData(const std::string& data_str);
};

}  // namespace vtk