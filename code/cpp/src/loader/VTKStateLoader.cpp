#include "VTKStateLoader.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>

#include "util/Quaternion.h"

namespace vtk {

VTKStateLoader::VTKStateLoader(const std::string& vtk_directory)
    : vtk_directory_(vtk_directory),
      data_dir_(std::filesystem::path(vtk_directory) / "data") {
  if (!std::filesystem::exists(vtk_directory_) || !std::filesystem::exists(data_dir_)) {
    throw std::runtime_error("VTK directory or data subdirectory does not exist");
  }
}

LoadedState VTKStateLoader::loadLatestState() {
  auto latest_timestep = findLatestTimestep();
  if (!latest_timestep) {
    throw std::runtime_error("No timesteps found in VTK directory");
  }
  return loadState(*latest_timestep);
}

int VTKStateLoader::getLatestTimestep() const {
  auto latest = findLatestTimestep();
  if (!latest) {
    throw std::runtime_error("No timesteps found in VTK directory");
  }
  return *latest;
}

std::optional<int> VTKStateLoader::findLatestTimestep() const {
  std::optional<int> latest_timestep;
  std::regex pattern("simulation_\\d{7}\\.pvtu");  // Matches exactly 7 digits

  // Only search in the root directory
  for (const auto& entry : std::filesystem::directory_iterator(vtk_directory_)) {
    if (entry.is_regular_file()) {
      std::string filename = entry.path().filename().string();
      std::string filepath = entry.path().string();

      // Check for simulation timestep files
      std::smatch match;
      if (std::regex_match(filename, match, pattern)) {
        // Extract the number from the filename (e.g., "simulation_0000123.pvtu" -> "0000123")
        std::string timestep_str = filename.substr(11, 7);
        int timestep = std::stoi(timestep_str);
        if (!latest_timestep || timestep > *latest_timestep) {
          latest_timestep = timestep;
        }
      }
    }
  }

  if (!latest_timestep) {
    PetscPrintf(PETSC_COMM_WORLD, "No timesteps found in VTK directory\n");
    PetscPrintf(PETSC_COMM_WORLD, "Directory contents:\n");
    for (const auto& entry : std::filesystem::directory_iterator(vtk_directory_)) {
      PetscPrintf(PETSC_COMM_WORLD, "  %s\n", entry.path().string().c_str());
    }
  }

  return latest_timestep;
}

LoadedState VTKStateLoader::loadState(int timestep) {
  LoadedState state;

  std::stringstream step_basename_ss;
  // Load simulation metadata
  step_basename_ss << "simulation_" << std::setw(7) << std::setfill('0') << timestep;
  std::string sim_file = (std::filesystem::path(vtk_directory_) / (step_basename_ss.str() + ".pvtu")).string();
  auto field_data = readFieldData(sim_file);

  // Extract simulation metadata
  state.simulation_time_s = field_data["simulation_time_s"];

  state.dt_s = field_data["dt_s"];
  state.step = static_cast<int>(field_data["step"]);

  // Load particles
  state.particles = loadParticles(timestep);

  return state;
}

std::vector<Particle> VTKStateLoader::loadParticles(int timestep) {
  std::vector<Particle> particles;

  std::stringstream step_basename_ss;
  step_basename_ss << "particles_" << std::setw(7) << std::setfill('0') << timestep;

  std::string pvtu_file = (std::filesystem::path(vtk_directory_) / (step_basename_ss.str() + ".pvtu")).string();
  auto piece_files = getPieceFiles(pvtu_file);

  for (const auto& piece_file : piece_files) {
    std::string full_piece_path = (std::filesystem::path(vtk_directory_) / piece_file).string();
    auto data_arrays = readVTUFile(full_piece_path);

    // Get required arrays
    const auto& positions = data_arrays["Points"].data;

    const auto& gids = data_arrays["gid"].data;
    const auto& quaternions = data_arrays["quaternion"].data;
    const auto& lengths = data_arrays["lengths"].data;
    const auto& forces = data_arrays["forces"].data;
    const auto& velocity_linear = data_arrays["velocity_linear"].data;
    const auto& velocity_angular = data_arrays["velocity_angular"].data;

    size_t num_particles = positions.size() / 3;  // 3 components per position

    for (size_t i = 0; i < num_particles; i++) {
      std::array<double, 3> position = {
          positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]};

      std::array<double, 4> quaternion = {quaternions[i * 4], quaternions[i * 4 + 1], quaternions[i * 4 + 2], quaternions[i * 4 + 3]};

      // Create particle with basic properties
      Particle p(static_cast<PetscInt>(gids[i]),
                 position,
                 quaternion,
                 lengths[i * 3],  // length
                 1,               // l0 (initial length)
                 0.5);            // diameter

      // Set force
      const PetscScalar force_array[6] = {
          forces[i * 3], forces[i * 3 + 1], forces[i * 3 + 2],
          0.0, 0.0, 0.0  // No torque data in VTK
      };
      p.setForceAndTorque(force_array);

      // Set velocity
      const PetscScalar velocity_array[6] = {
          velocity_linear[i * 3], velocity_linear[i * 3 + 1], velocity_linear[i * 3 + 2],
          0.0, 0.0, 0.0  // No angular velocity data in VTK
      };
      p.setVelocity(velocity_array);

      // Set angular velocity
      const PetscScalar angular_velocity_array[6] = {
          velocity_angular[i * 3], velocity_angular[i * 3 + 1], velocity_angular[i * 3 + 2],
          0.0, 0.0, 0.0  // No angular velocity data in VTK
      };
      p.setVelocityAngular(angular_velocity_array);

      particles.push_back(std::move(p));
    }
  }

  return particles;
}

std::vector<std::string> VTKStateLoader::getPieceFiles(const std::string& pvtu_file) {
  std::vector<std::string> piece_files;
  std::ifstream file(pvtu_file);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open PVTU file: " + pvtu_file);
  }

  std::string line;
  std::regex piece_pattern(".*Source=\"([^\"]+)\".*");

  while (std::getline(file, line)) {
    std::smatch match;
    if (std::regex_match(line, match, piece_pattern)) {
      piece_files.push_back(match[1]);
    }
  }

  return piece_files;
}

std::map<std::string, VTUDataArray> VTKStateLoader::readVTUFile(const std::string& vtu_file) {
  std::map<std::string, VTUDataArray> data_arrays;
  std::ifstream file(vtu_file);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open VTU file: " + vtu_file);
  }
  std::string line;
  // Simplified regex to match DataArray opening tags
  std::regex array_pattern("<DataArray\\s+[^>]*>");

  std::string current_array_name;
  std::string current_array_type;
  int current_components = 1;
  bool reading_data = false;
  std::stringstream data_buffer;

  while (std::getline(file, line)) {
    // Skip lines that start with "line:" (debug output)
    if (line.find("line:") == 0) {
      continue;
    }

    if (!reading_data) {
      std::smatch match;
      if (std::regex_search(line, match, array_pattern)) {
        // Extract attributes from the matched DataArray tag
        std::string tag = match[0].str();

        // Extract Name attribute
        std::regex name_regex("Name=\"([^\"]+)\"");
        std::smatch name_match;
        if (std::regex_search(tag, name_match, name_regex)) {
          current_array_name = name_match[1].str();
        }

        // Extract type attribute
        std::regex type_regex("type=\"([^\"]+)\"");
        std::smatch type_match;
        if (std::regex_search(tag, type_match, type_regex)) {
          current_array_type = type_match[1].str();
        }

        // Extract NumberOfComponents attribute
        std::regex comp_regex("NumberOfComponents=\"([^\"]+)\"");
        std::smatch comp_match;
        if (std::regex_search(tag, comp_match, comp_regex)) {
          current_components = std::stoi(comp_match[1].str());
        }

        reading_data = true;
        data_buffer.str("");
        data_buffer.clear();
      }
    } else {
      // Check if we've reached the end of the DataArray
      if (line.find("</DataArray>") != std::string::npos) {
        // Process the collected data
        if (!current_array_name.empty()) {
          VTUDataArray array_data;
          array_data.name = current_array_name;
          array_data.type = current_array_type;
          array_data.num_components = current_components;

          // Parse the data values
          std::string data_str = data_buffer.str();
          std::istringstream iss(data_str);
          std::string value;

          while (iss >> value) {
            try {
              array_data.data.push_back(std::stod(value));
            } catch (const std::exception& e) {
              // Skip invalid values
              throw std::runtime_error("Invalid value: " + value);
              continue;
            }
          }

          data_arrays[current_array_name] = array_data;
        }

        // Reset for next array
        reading_data = false;
        current_array_name.clear();
        current_array_type.clear();
        current_components = 1;
      } else {
        // Accumulate data lines
        data_buffer << line << " ";
      }
    }
  }

  return data_arrays;
}

std::map<std::string, double> VTKStateLoader::readFieldData(const std::string& pvtu_file) {
  std::map<std::string, double> field_data;

  // First get the directory of the pvtu file
  std::filesystem::path pvtu_path(pvtu_file);
  std::string pvtu_dir = pvtu_path.parent_path().string();

  // Read the pvtu file to get the piece file
  std::ifstream pvtu_stream(pvtu_file);
  if (!pvtu_stream.is_open()) {
    throw std::runtime_error("Could not open PVTU file: " + pvtu_file);
  }

  std::string line;
  std::regex piece_pattern("<Piece\\s+Source=\"([^\"]+)\".*>");
  std::string vtu_file;

  // Find the first piece file
  while (std::getline(pvtu_stream, line)) {
    std::smatch piece_match;
    if (std::regex_search(line, piece_match, piece_pattern)) {
      vtu_file = pvtu_dir + "/" + std::string(piece_match[1]);
      break;
    }
  }
  pvtu_stream.close();

  if (vtu_file.empty()) {
    throw std::runtime_error("No piece file found in PVTU: " + pvtu_file);
  }

  // Now read the actual VTU file
  std::ifstream vtu_stream(vtu_file);
  if (!vtu_stream.is_open()) {
    throw std::runtime_error("Could not open VTU file: " + vtu_file);
  }

  std::regex field_pattern("<DataArray.*Name=\"([^\"]+)\".*>");
  std::regex value_pattern("\\s*([\\d.-]+)\\s*");

  std::string current_field;
  bool reading_value = false;

  while (std::getline(vtu_stream, line)) {
    if (!reading_value) {
      std::smatch field_match;
      if (std::regex_search(line, field_match, field_pattern)) {
        current_field = field_match[1];
        reading_value = true;
      }
    } else {
      std::smatch value_match;
      if (std::regex_search(line, value_match, value_pattern)) {
        try {
          field_data[current_field] = std::stod(value_match[1]);
        } catch (const std::exception& e) {
          // Skip if value cannot be converted to double
        }
        reading_value = false;
      }
    }

    // Stop when we reach the end of FieldData section
    if (line.find("</FieldData>") != std::string::npos) {
      break;
    }
  }

  return field_data;
}

template <typename T>
std::vector<T> VTKStateLoader::parseNumericData(const std::string& data_str) {
  std::vector<T> result;
  std::istringstream iss(data_str);
  T value;
  while (iss >> value) {
    result.push_back(value);
  }
  return result;
}

}  // namespace vtk