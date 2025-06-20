#include "VTK.h"

#include <mpi.h>

#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "dynamics/Constraint.h"
#include "simulation/Particle.h"
#include "util/Quaternion.h"

namespace vtk {

// VTKLogger Implementation
VTKLogger::VTKLogger(const std::string& output_dir, const std::string& prefix,
                     std::unique_ptr<VTKDataExtractor> extractor)
    : output_dir_(output_dir), prefix_(prefix), data_extractor_(std::move(extractor)) {
  data_dir_ = output_dir_ + "/data";
  start_time_ = std::chrono::high_resolution_clock::now();
  setupDirectories();
}

VTKLogger::~VTKLogger() = default;

void VTKLogger::setDataExtractor(std::unique_ptr<VTKDataExtractor> extractor) {
  data_extractor_ = std::move(extractor);
}

void VTKLogger::setupDirectories() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Only rank 0 creates/removes directories
  if (rank == 0) {
    // Remove existing directory if it exists
    if (std::filesystem::exists(output_dir_)) {
      try {
        std::filesystem::remove_all(output_dir_);
      } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Warning: Could not remove existing directory: " << e.what() << std::endl;
      }
    }

    // Create output directories
    std::filesystem::create_directories(output_dir_);
    std::filesystem::create_directories(data_dir_);
  }

  // Synchronize all processes to ensure directories are created
  MPI_Barrier(MPI_COMM_WORLD);
}

std::string VTKLogger::getVTUFilename(int frame_number) const {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::stringstream ss;
  ss << prefix_ << std::setfill('0') << std::setw(7) << frame_number
     << "_rank_" << std::setfill('0') << std::setw(4) << rank << ".vtu";
  return ss.str();
}

std::string VTKLogger::getPVTUFilename(int frame_number) const {
  std::stringstream ss;
  ss << prefix_ << std::setfill('0') << std::setw(7) << frame_number << ".pvtu";
  return ss.str();
}

std::string VTKLogger::dataTypeToString(DataType type) const {
  switch (type) {
    case DataType::Float32:
      return "Float32";
    case DataType::Float64:
      return "Float64";
    case DataType::Int32:
      return "Int32";
    case DataType::UInt8:
      return "UInt8";
    case DataType::String:
      return "String";
    default:
      return "Float32";
  }
}

void VTKLogger::writeTimestepPVTU(int frame_number, const std::vector<VTKField>& point_fields,
                                  const std::vector<VTKField>& cell_fields, const VTKFieldData& field_data) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Only rank 0 writes PVTU files
  if (rank != 0) return;

  std::string pvtu_filename = getPVTUFilename(frame_number);
  std::string pvtu_path = output_dir_ + "/" + pvtu_filename;

  std::ofstream file(pvtu_path);
  file << "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\" ?>\n";
  file << "<VTKFile byte_order=\"LittleEndian\" type=\"PUnstructuredGrid\" version=\"0.1\">\n";
  file << "  <PUnstructuredGrid GhostLevel=\"0\">\n";

  // Field data
  if (!field_data.fields.empty()) {
    file << "    <PFieldData>\n";
    for (const auto& [name, info] : field_data.fields) {
      file << "      <PDataArray Name=\"" << name << "\" NumberOfComponents=\"1\" "
           << "format=\"ascii\" type=\"" << dataTypeToString(info.type) << "\"/>\n";
    }
    file << "    </PFieldData>\n";
  }

  // Point data
  if (!point_fields.empty()) {
    file << "    <PPointData>\n";
    for (const auto& field : point_fields) {
      file << "      <PDataArray Name=\"" << field.name << "\" "
           << "NumberOfComponents=\"" << field.components << "\" "
           << "format=\"ascii\" type=\"" << dataTypeToString(field.data_type) << "\"/>\n";
    }
    file << "    </PPointData>\n";
  }

  // Cell data
  if (!cell_fields.empty()) {
    file << "    <PCellData>\n";
    for (const auto& field : cell_fields) {
      file << "      <PDataArray Name=\"" << field.name << "\" "
           << "NumberOfComponents=\"" << field.components << "\" "
           << "format=\"ascii\" type=\"" << dataTypeToString(field.data_type) << "\"/>\n";
    }
    file << "    </PCellData>\n";
  } else {
    file << "    <PCellData/>\n";
  }

  // Points
  file << "    <PPoints>\n";
  file << "      <PDataArray Name=\"positions\" NumberOfComponents=\"3\" "
       << "format=\"ascii\" type=\"Float32\"/>\n";
  file << "    </PPoints>\n";

  // Cells
  file << "    <PCells>\n";
  file << "      <PDataArray Name=\"connectivity\" NumberOfComponents=\"1\" "
       << "format=\"ascii\" type=\"Int32\"/>\n";
  file << "      <PDataArray Name=\"offsets\" NumberOfComponents=\"1\" "
       << "format=\"ascii\" type=\"Int32\"/>\n";
  file << "      <PDataArray Name=\"types\" NumberOfComponents=\"1\" "
       << "format=\"ascii\" type=\"UInt8\"/>\n";
  file << "    </PCells>\n";

  // Add piece references for all ranks
  for (int r = 0; r < size; ++r) {
    std::stringstream rank_vtu_name;
    rank_vtu_name << prefix_ << std::setfill('0') << std::setw(7) << frame_number
                  << "_rank_" << std::setfill('0') << std::setw(4) << r << ".vtu";
    file << "    <Piece Source=\"./data/" << rank_vtu_name.str() << "\"/>\n";
  }

  file << "  </PUnstructuredGrid>\n";
  file << "</VTKFile>\n";
}

void VTKLogger::addVTUFile(int frame_number, double dt, const VTKGeometry& geometry,
                           const std::vector<VTKField>& point_fields,
                           const std::vector<VTKField>& cell_fields,
                           const VTKFieldData& field_data) {
  std::string vtu_filename = getVTUFilename(frame_number);
  std::string vtu_path = data_dir_ + "/" + vtu_filename;

  int n_points = geometry.positions.size();
  int n_cells = geometry.offsets.size();

  std::ofstream file(vtu_path);
  file << "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\" ?>\n";
  file << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
  file << "  <UnstructuredGrid>\n";

  // Field Data
  if (!field_data.fields.empty()) {
    file << "    <FieldData>\n";
    for (const auto& [name, info] : field_data.fields) {
      file << "      <DataArray type=\"" << dataTypeToString(info.type) << "\" Name=\"" << name << "\" "
           << "NumberOfTuples=\"1\">";

      // Cast to appropriate type if needed
      if (info.type == DataType::Float32 && info.is_numeric) {
        file << static_cast<float>(info.value_num);
      } else {
        file << info.value_str;
      }

      file << "</DataArray>\n";
    }
    file << "    </FieldData>\n";
  }

  file << "    <Piece NumberOfCells=\"" << n_cells << "\" NumberOfPoints=\"" << n_points << "\">\n";

  // Point Data
  if (!point_fields.empty()) {
    file << "      <PointData>\n";
    for (const auto& field : point_fields) {
      writeDataArray(file, field, "        ");
    }
    file << "      </PointData>\n";
  }

  // Cell Data
  if (!cell_fields.empty()) {
    file << "      <CellData>\n";
    for (const auto& field : cell_fields) {
      writeDataArray(file, field, "        ");
    }
    file << "      </CellData>\n";
  } else {
    file << "      <CellData>\n";
    file << "      </CellData>\n";
  }

  // Points
  file << "      <Points>\n";
  file << "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" "
       << "Name=\"positions\" format=\"ascii\">\n";
  for (const auto& pos : geometry.positions) {
    file << "          " << static_cast<float>(pos[0]) << " "
         << static_cast<float>(pos[1]) << " " << static_cast<float>(pos[2]) << "\n";
  }
  file << "        </DataArray>\n";
  file << "      </Points>\n";

  // Cells
  file << "      <Cells>\n";

  // Connectivity
  file << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
  for (int conn : geometry.connectivity) {
    file << "          " << conn << "\n";
  }
  file << "        </DataArray>\n";

  // Offsets
  file << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
  for (int offset : geometry.offsets) {
    file << "          " << offset << "\n";
  }
  file << "        </DataArray>\n";

  // Cell types
  file << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
  for (uint8_t cell_type : geometry.cell_types) {
    file << "          " << static_cast<int>(cell_type) << "\n";
  }
  file << "        </DataArray>\n";

  file << "      </Cells>\n";
  file << "    </Piece>\n";
  file << "  </UnstructuredGrid>\n";
  file << "</VTKFile>\n";
}

void VTKLogger::writeDataArray(std::ofstream& file, const VTKField& field, const std::string& indent) {
  file << indent << "<DataArray type=\"" << dataTypeToString(field.data_type) << "\" "
       << "NumberOfComponents=\"" << field.components << "\" "
       << "Name=\"" << field.name << "\" format=\"ascii\">\n";

  if (field.components == 1) {
    for (double value : field.data) {
      if (field.data_type == DataType::Float32) {
        file << indent << "  " << static_cast<float>(value) << "\n";
      } else if (field.data_type == DataType::Int32) {
        file << indent << "  " << static_cast<int>(value) << "\n";
      } else {
        file << indent << "  " << value << "\n";
      }
    }
  } else {
    for (size_t i = 0; i < field.data.size(); i += field.components) {
      file << indent << "  ";
      for (int j = 0; j < field.components && i + j < field.data.size(); ++j) {
        if (field.data_type == DataType::Float32) {
          file << static_cast<float>(field.data[i + j]);
        } else if (field.data_type == DataType::Int32) {
          file << static_cast<int>(field.data[i + j]);
        } else {
          file << field.data[i + j];
        }
        if (j < field.components - 1) file << " ";
      }
      file << "\n";
    }
  }

  file << indent << "</DataArray>\n";
}

void VTKLogger::logTimestep(int frame_number, double dt, const void* state, bool is_substep) {
  if (!data_extractor_) {
    throw std::runtime_error("No data extractor provided");
  }

  auto now = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(now - start_time_);
  double elapsed_time = elapsed.count();

  auto geometry = data_extractor_->extractGeometry(state);
  auto point_fields = data_extractor_->extractPointData(state);
  auto cell_fields = data_extractor_->extractCellData(state);
  auto field_data = data_extractor_->extractFieldData(state, frame_number, dt, elapsed_time, is_substep);

  addVTUFile(frame_number, dt, geometry, point_fields, cell_fields, field_data);
  writeTimestepPVTU(frame_number, point_fields, cell_fields, field_data);
}

void VTKLogger::clearDataFolder() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Only rank 0 clears the data folder
  if (rank == 0) {
    for (const auto& entry : std::filesystem::directory_iterator(data_dir_)) {
      std::cout << "Removing " << entry.path().filename() << std::endl;
      std::filesystem::remove(entry.path());
    }
  }

  // Synchronize all processes
  MPI_Barrier(MPI_COMM_WORLD);
}

// ParticleDataExtractor Implementation
VTKGeometry ParticleDataExtractor::extractGeometry(const void* state) {
  const auto* sim_state = static_cast<const ParticleSimulationState*>(state);

  std::vector<std::array<double, 3>> positions;
  positions.reserve(sim_state->particles.size());

  for (const auto& particle : sim_state->particles) {
    positions.push_back(particle.getPosition());
  }

  return VTKGeometry(positions);
}

std::vector<VTKField> ParticleDataExtractor::extractPointData(const void* state) {
  const auto* sim_state = static_cast<const ParticleSimulationState*>(state);
  std::vector<VTKField> fields;

  const size_t n_particles = sim_state->particles.size();

  // Get MPI rank information
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Extract quaternions and calculate directions
  std::vector<std::array<double, 3>> directions;
  directions.reserve(n_particles);

  for (const auto& particle : sim_state->particles) {
    auto quat = particle.getQuaternion();
    auto dir = utils::Quaternion::getDirectionVector(quat);
    directions.push_back(dir);
  }

  // Extract lengths (particle length, diameter/2, diameter/2)
  std::vector<std::array<double, 3>> lengths;
  lengths.reserve(n_particles);

  for (const auto& particle : sim_state->particles) {
    double length = particle.getLength();
    double diameter = particle.getDiameter();
    lengths.push_back({length, diameter, diameter});
  }

  // Extract particle IDs
  std::vector<int> ids;
  ids.reserve(n_particles);
  for (const auto& particle : sim_state->particles) {
    ids.push_back(particle.getLocalID());
  }

  std::vector<int> gIDs;
  gIDs.reserve(n_particles);
  for (const auto& particle : sim_state->particles) {
    gIDs.push_back(particle.getGID());
  }

  // Extract type IDs (all zeros for now)
  std::vector<int> type_ids(n_particles, 0);

  // Create rank field - each particle gets labeled with its owning rank
  std::vector<int> ranks(n_particles, rank);

  // Always add all fields, even if empty, to maintain consistency across ranks
  // Forces field - use sim_state forces if available, otherwise create empty
  std::vector<std::array<double, 3>> forces;
  if (!sim_state->forces.empty() && sim_state->forces.size() == n_particles) {
    forces = sim_state->forces;
  } else {
    forces.resize(n_particles, {0.0, 0.0, 0.0});
  }
  fields.emplace_back("forces", forces);

  // Torques field - use sim_state torques if available, otherwise create empty
  std::vector<std::array<double, 3>> torques;
  if (!sim_state->torques.empty() && sim_state->torques.size() == n_particles) {
    torques = sim_state->torques;
  } else {
    torques.resize(n_particles, {0.0, 0.0, 0.0});
  }
  fields.emplace_back("torques", torques);

  // Velocities_linear - use sim_state velocities if available, otherwise create empty
  std::vector<std::array<double, 3>> velocities_linear;
  if (!sim_state->velocities_linear.empty() && sim_state->velocities_linear.size() == n_particles) {
    velocities_linear = sim_state->velocities_linear;
  } else {
    velocities_linear.resize(n_particles, {0.0, 0.0, 0.0});
  }
  fields.emplace_back("velocities_linear", velocities_linear);

  // Velocities_angular - use sim_state velocities if available, otherwise create empty
  std::vector<std::array<double, 3>> velocities_angular;
  if (!sim_state->velocities_angular.empty() && sim_state->velocities_angular.size() == n_particles) {
    velocities_angular = sim_state->velocities_angular;
  } else {
    velocities_angular.resize(n_particles, {0.0, 0.0, 0.0});
  }
  fields.emplace_back("velocities_angular", velocities_angular);

  // Always add these fields
  fields.emplace_back("directions", directions);

  // Impedance field - use sim_state impedance if available, otherwise create empty
  std::vector<double> impedance;
  for (const auto& particle : sim_state->particles) {
    impedance.push_back(particle.getImpedance());
  }
  fields.emplace_back("impedance", impedance);

  fields.emplace_back("lengths", lengths);
  fields.emplace_back("typeIds", type_ids, DataType::Int32);
  fields.emplace_back("ids", ids, DataType::Int32);
  fields.emplace_back("gIDs", gIDs, DataType::Int32);

  // Add rank information as point data
  fields.emplace_back("rank", ranks, DataType::Int32);

  return fields;
}

std::vector<VTKField> ParticleDataExtractor::extractCellData(const void* state) {
  // No cell data for particle simulation
  return {};
}

VTKFieldData ParticleDataExtractor::extractFieldData(const void* state, int timestep, double dt, double elapsed_time, bool is_substep) {
  const auto* sim_state = static_cast<const ParticleSimulationState*>(state);

  // Get MPI rank and size information
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  VTKFieldData field_data;
  field_data.addField("elapsed_time", elapsed_time);
  field_data.addField("simulation_time", dt * timestep);
  field_data.addField("bbpgd_iterations", sim_state->bbpgd_iterations);
  field_data.addField("constraint_iterations", sim_state->constraint_iterations);
  field_data.addField("residual", sim_state->residual);
  field_data.addField("num_particles", static_cast<int>(sim_state->particles.size()));
  field_data.addField("is_substep", is_substep ? 1 : 0);

  // Add MPI information
  field_data.addField("mpi_rank", rank);
  field_data.addField("mpi_size", size);

  return field_data;
}

// ConstraintDataExtractor Implementation
VTKGeometry ConstraintDataExtractor::extractGeometry(const void* state) {
  const auto* sim_state = static_cast<const ParticleSimulationState*>(state);

  std::vector<std::array<double, 3>> positions;
  positions.reserve(sim_state->constraints.size());

  for (const auto& constraint : sim_state->constraints) {
    // Validate and sanitize contact point to avoid VTK parsing errors
    auto center = constraint.contactPoint;

    // Check for invalid/corrupted values and fix them
    for (int i = 0; i < 3; ++i) {
      if (std::isnan(center[i]) || std::isinf(center[i]) || std::abs(center[i]) > 1e10) {
        center[i] = 0.0;  // Replace invalid values with 0
      }
    }

    positions.push_back(center);
  }

  return VTKGeometry(positions);
}

std::vector<VTKField> ConstraintDataExtractor::extractPointData(const void* state) {
  const auto* sim_state = static_cast<const ParticleSimulationState*>(state);
  std::vector<VTKField> fields;

  const size_t n_constraints = sim_state->constraints.size();

  // Get MPI rank for ownership information
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Extract constraint information (keeping only essential fields)
  std::vector<std::array<double, 3>> normals;
  std::vector<std::array<double, 3>> contact_positions;
  std::vector<int> ranks;
  std::vector<double> overlap_magnitudes;
  std::vector<int> gidI;
  std::vector<int> gidJ;
  std::vector<int> violated;
  std::vector<int> constraint_iterations;

  // Reserve space for all vectors (one value per constraint)
  normals.reserve(n_constraints);
  contact_positions.reserve(n_constraints);
  ranks.reserve(n_constraints);
  overlap_magnitudes.reserve(n_constraints);
  violated.reserve(n_constraints);
  constraint_iterations.reserve(n_constraints);
  gidI.reserve(n_constraints);
  gidJ.reserve(n_constraints);

  int constraint_index = 0;
  for (const auto& constraint : sim_state->constraints) {
    // Basic constraint data
    normals.push_back(constraint.normI);
    overlap_magnitudes.push_back(-constraint.delta0);  // Positive overlap value
    violated.push_back(constraint.violated ? 1 : 0);
    constraint_iterations.push_back(constraint.constraint_iterations);

    // Contact positions - sanitize for VTK output
    auto sanitized_contact = constraint.contactPoint;
    for (int i = 0; i < 3; ++i) {
      if (std::isnan(sanitized_contact[i]) || std::isinf(sanitized_contact[i]) || std::abs(sanitized_contact[i]) > 1e10) {
        sanitized_contact[i] = 0.0;
      }
    }
    contact_positions.push_back(sanitized_contact);

    // Rank information
    ranks.push_back(rank);
    gidI.push_back(constraint.gidI);
    gidJ.push_back(constraint.gidJ);

    constraint_index++;
  }

  // Add essential fields only
  fields.emplace_back("normals", normals);
  fields.emplace_back("overlap_magnitudes", overlap_magnitudes);
  fields.emplace_back("violated", violated, DataType::Int32);
  fields.emplace_back("rank", ranks, DataType::Int32);
  fields.emplace_back("contact_positions", contact_positions);
  fields.emplace_back("constraint_iterations", constraint_iterations, DataType::Int32);
  fields.emplace_back("gidI", gidI, DataType::Int32);
  fields.emplace_back("gidJ", gidJ, DataType::Int32);

  return fields;
}

std::vector<VTKField> ConstraintDataExtractor::extractCellData(const void* state) {
  // No cell data for constraint simulation
  return {};
}

VTKFieldData ConstraintDataExtractor::extractFieldData(const void* state, int timestep, double dt, double elapsed_time, bool is_substep) {
  const auto* sim_state = static_cast<const ParticleSimulationState*>(state);

  // Get MPI rank and size information
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  VTKFieldData field_data;

  // Basic constraint information
  field_data.addField("num_constraints", static_cast<int>(sim_state->constraints.size()));
  field_data.addField("elapsed_time", elapsed_time);
  field_data.addField("simulation_time", dt * timestep);

  // Solver information
  field_data.addField("bbpgd_iterations", sim_state->bbpgd_iterations);
  field_data.addField("constraint_iterations", sim_state->constraint_iterations);
  field_data.addField("residual", sim_state->residual);
  field_data.addField("is_substep", is_substep ? 1 : 0);

  // MPI information
  field_data.addField("mpi_rank", rank);
  field_data.addField("mpi_size", size);

  // Count cross-rank constraints
  int cross_rank_count = 0;
  double max_constraint_overlap = 0.0;

  for (const auto& constraint : sim_state->constraints) {
    if (constraint.localI == -1 || constraint.localJ == -1) {
      cross_rank_count++;
    }
    max_constraint_overlap = std::max(max_constraint_overlap, -constraint.delta0);
  }

  field_data.addField("cross_rank_constraints", cross_rank_count);
  field_data.addField("max_constraint_overlap", max_constraint_overlap);

  return field_data;
}

// SimulationLogger Implementation
SimulationLogger::SimulationLogger(std::unique_ptr<VTKLogger> vtk_logger, int log_every_n_iterations)
    : vtk_logger_(std::move(vtk_logger)), log_every_n_iterations_(log_every_n_iterations) {}

bool SimulationLogger::shouldLog(int iteration) const {
  return iteration % log_every_n_iterations_ == 0;
}

void SimulationLogger::logTimestepComplete(double dt, const void* state) {
  if (shouldLog(current_timestep_)) {
    vtk_logger_->logTimestep(frame_counter_, dt, state, false);
    frame_counter_++;
  }
  current_timestep_++;
}

void SimulationLogger::logSubstep(double dt, const void* state) {
  if (shouldLog(current_timestep_)) {
    vtk_logger_->logTimestep(frame_counter_, dt, state, true);
    frame_counter_++;
  }
}

// Factory functions
std::unique_ptr<SimulationLogger> createParticleLogger(const std::string& output_dir, int log_every_n_iterations) {
  auto extractor = std::make_unique<ParticleDataExtractor>();
  auto vtk_logger = std::make_unique<VTKLogger>(output_dir, "Particles_", std::move(extractor));
  return std::make_unique<SimulationLogger>(std::move(vtk_logger), log_every_n_iterations);
}

std::unique_ptr<SimulationLogger> createConstraintLogger(const std::string& output_dir, int log_every_n_iterations) {
  auto extractor = std::make_unique<ConstraintDataExtractor>();
  auto vtk_logger = std::make_unique<VTKLogger>(output_dir, "Constraints_", std::move(extractor));
  return std::make_unique<SimulationLogger>(std::move(vtk_logger), log_every_n_iterations);
}

}  // namespace vtk
