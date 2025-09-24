#pragma once

#include <petsc.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

// Helper function to convert array to string
template <typename T, size_t N>
std::string arrayToString(const std::array<T, N>& arr) {
  std::ostringstream oss;
  for (size_t i = 0; i < N; ++i) {
    if (i > 0) oss << " ";
    oss << arr[i];
  }
  return oss.str();
}

// Minimal VTK-compatible data array interface
class VTKDataArray {
 public:
  virtual ~VTKDataArray() = default;
  virtual const char* GetName() const = 0;
  virtual int GetNumberOfComponents() const = 0;
  virtual int GetNumberOfTuples() const = 0;
  virtual std::string GetDataType() const = 0;
  virtual std::string GetDataAsString() const = 0;
};

template <typename T>
class VTKTypedDataArray : public VTKDataArray {
  std::string name;
  int numComponents;
  std::vector<T> data;

 public:
  VTKTypedDataArray(const std::string& name, int numComponents, const std::vector<T>& data)
      : name(name), numComponents(numComponents), data(data) {}

  const char* GetName() const override { return name.c_str(); }
  int GetNumberOfComponents() const override { return numComponents; }
  int GetNumberOfTuples() const override { return data.size() / numComponents; }

  std::string GetDataType() const override {
    if constexpr (std::is_same_v<T, int>)
      return "Int32";
    else if constexpr (std::is_same_v<T, float>)
      return "Float32";
    else if constexpr (std::is_same_v<T, double>)
      return "Float64";
    else if constexpr (std::is_same_v<T, uint8_t>)
      return "UInt8";
    else if constexpr (std::is_same_v<T, long long>)
      return "Int64";
    else if constexpr (std::is_same_v<T, size_t>)
      return "UInt64";
    else if constexpr (std::is_same_v<T, long>)
      return "Int64";
    else
      throw std::runtime_error("VTKTypedDataArray: Unsupported data type: " + std::string(typeid(T).name()));
  }

  std::string GetDataAsString() const override {
    std::ostringstream oss;
    if constexpr (std::is_arithmetic_v<T>) {
      for (size_t i = 0; i < data.size(); ++i) {
        if (i > 0) oss << " ";
        oss << data[i];
      }
    }
    return oss.str();
  }
};

// Specialization for std::array
template <typename T, size_t N>
class VTKTypedDataArray<std::array<T, N>> : public VTKDataArray {
  std::string name;
  std::vector<std::array<T, N>> data;

 public:
  VTKTypedDataArray(const std::string& name, int /*numComponents*/, const std::vector<std::array<T, N>>& data)
      : name(name), data(data) {}

  const char* GetName() const override { return name.c_str(); }
  int GetNumberOfComponents() const override { return N; }
  int GetNumberOfTuples() const override { return data.size(); }

  std::string GetDataType() const override {
    if constexpr (std::is_same_v<T, int>)
      return "Int32";
    else if constexpr (std::is_same_v<T, float>)
      return "Float32";
    else if constexpr (std::is_same_v<T, double>)
      return "Float64";
    else if constexpr (std::is_same_v<T, uint8_t>)
      return "UInt8";
    else if constexpr (std::is_same_v<T, long long>)
      return "Int64";
    else if constexpr (std::is_same_v<T, size_t>)
      return "UInt64";
    else if constexpr (std::is_same_v<T, long>)
      return "Int64";
    else
      throw std::runtime_error("GetDataType: Unsupported data type: " + std::string(typeid(T).name()));
  }

  std::string GetDataAsString() const override {
    std::ostringstream oss;
    for (size_t i = 0; i < data.size(); ++i) {
      if (i > 0) oss << "\n          ";
      oss << arrayToString(data[i]);
    }
    return oss.str();
  }
};

class VTKPoints {
  std::vector<std::array<float, 3>> positions;

 public:
  void InsertNextPoint(float x, float y, float z) {
    positions.push_back({x, y, z});
  }

  const auto& GetPositions() const { return positions; }
};

class VTKCellArray {
  std::vector<std::vector<int32_t>> connectivity;

 public:
  void InsertNextCell(const std::vector<int32_t>& cell) {
    connectivity.push_back(cell);
  }

  const auto& GetConnectivity() const { return connectivity; }
};

class VTKFieldData {
  std::vector<std::unique_ptr<VTKDataArray>> arrays;

 public:
  void AddArray(std::unique_ptr<VTKDataArray> array) {
    arrays.push_back(std::move(array));
  }

  const auto& GetArrays() const { return arrays; }
};

class VTKUnstructuredGrid {
  VTKPoints points;
  VTKCellArray cells;
  VTKFieldData pointData;
  VTKFieldData fieldData;

 public:
  VTKPoints& GetPoints() { return points; }
  VTKCellArray& GetCells() { return cells; }
  VTKFieldData& GetPointData() { return pointData; }
  VTKFieldData& GetFieldData() { return fieldData; }

  const VTKPoints& GetPoints() const { return points; }
  const VTKCellArray& GetCells() const { return cells; }
  const VTKFieldData& GetPointData() const { return pointData; }
  const VTKFieldData& GetFieldData() const { return fieldData; }
};

template <typename T>
class VTKDataLogger {
  VTKUnstructuredGrid grid;
  std::string outputDirectory;
  std::string filePrefix;
  int mpiRank;
  int mpiSize;
  int step = 0;
  bool onlyRankZero = false;

  std::map<std::string, std::function<std::vector<std::unique_ptr<VTKDataArray>>(const std::string&, const T&)>> converters;

  // Private helper functions for writing VTK files
  void writeFieldData(std::ofstream& out);
  void writePoints(std::ofstream& out);
  void writeCells(std::ofstream& out);
  void writePointData(std::ofstream& out);
  void writePFieldData(std::ofstream& out);
  void writePPointData(std::ofstream& out);
  void writeVTUFile(const std::string& filename);
  void writePVTUFile(const std::string& pfilename, const std::string& step_basename);

 public:
  VTKDataLogger(const std::string& outputDirectory, const std::string& filePrefix, bool onlyRankZero = false, bool preserve_existing = false, size_t step = 0)
      : outputDirectory(outputDirectory),
        filePrefix(filePrefix),
        onlyRankZero(onlyRankZero),
        step(step) {
    MPI_Comm_rank(PETSC_COMM_WORLD, &mpiRank);
    MPI_Comm_size(PETSC_COMM_WORLD, &mpiSize);

    // On rank 0, create directories and optionally clear previous output for this prefix.
    if (mpiRank == 0) {
      std::filesystem::path dir(outputDirectory);
      std::filesystem::path dataDir = dir / "data";

      // Create directories. This is safe to call even if they exist.
      std::filesystem::create_directories(dataDir);

      // Only clear previous output if preserve_existing is false
      if (!preserve_existing) {
        auto clear_files = [&](const std::filesystem::path& path_to_clear, const std::string& extension) {
          if (!std::filesystem::exists(path_to_clear)) return;
          for (const auto& entry : std::filesystem::directory_iterator(path_to_clear)) {
            const std::string filename = entry.path().filename().string();
            if (entry.is_regular_file() && entry.path().extension() == extension &&
                filename.rfind(filePrefix, 0) == 0) {  // C++17 equivalent of starts_with
              std::filesystem::remove(entry.path());
            }
          }
        };

        clear_files(dir, ".pvtu");
        clear_files(dataDir, ".vtu");
      }
    }
  }

  // Add point data
  template <typename DataType>
  void addPointData(const std::string& name, const std::vector<DataType>& data, int numComponents = 1) {
    grid.GetPointData().AddArray(
        std::make_unique<VTKTypedDataArray<DataType>>(name, numComponents, data));
  }

  // Add field data
  template <typename DataType>
  void addFieldData(const std::string& name, const DataType& value) {
    if constexpr (std::is_arithmetic_v<DataType>) {
      std::vector<DataType> vec{value};
      grid.GetFieldData().AddArray(
          std::make_unique<VTKTypedDataArray<DataType>>(name, 1, vec));
    } else {
      if (converters.find(typeid(DataType).name()) != converters.end()) {
        auto arrays = converters[typeid(DataType).name()](name, value);
        for (auto& array : arrays) {
          grid.GetFieldData().AddArray(std::move(array));
        }
      }
    }
  }

  // Add points
  void addPoints(const std::vector<std::array<double, 3>>& positions) {
    for (const auto& pos : positions) {
      grid.GetPoints().InsertNextPoint(pos[0], pos[1], pos[2]);
    }
  }

  // Add cells
  void addCells(const std::vector<std::vector<int32_t>>& connectivity) {
    for (const auto& conn : connectivity) {
      grid.GetCells().InsertNextCell(conn);
    }
  }

  // Register converter
  template <typename CustomType>
  void registerConverter(const std::string& namePrefix,
                         std::function<std::vector<std::unique_ptr<VTKDataArray>>(const std::string&, const CustomType&)> converter) {
    converters[typeid(CustomType).name()] = converter;
  }

  // Write to file
  void write() {
    // --- Filename Construction ---
    // Base name for this step (e.g., "my_sim_0000000")
    std::ostringstream step_basename_ss;
    step_basename_ss << filePrefix << "_" << std::setw(7) << std::setfill('0') << step;
    const std::string step_basename = step_basename_ss.str();

    // VTU filename leaf part (e.g., "my_sim_0000000_rank_0000.vtu")
    std::ostringstream vtu_leafname_ss;
    vtu_leafname_ss << step_basename << "_rank_" << std::setw(4) << std::setfill('0') << mpiRank
                    << ".vtu";

    // Full path for the VTU file (e.g., "vtk_output/data/my_sim_..._rank_....vtu")
    std::filesystem::path vtu_filepath =
        std::filesystem::path(outputDirectory) / "data" / vtu_leafname_ss.str();

    writeVTUFile(vtu_filepath.string());

    if (mpiRank == 0) {
      // Full path for the PVTU file (e.g., "vtk_output/my_sim_....pvtu")
      std::filesystem::path pvtu_filepath =
          std::filesystem::path(outputDirectory) / (step_basename + ".pvtu");
      writePVTUFile(pvtu_filepath.string(), step_basename);
    }

    step++;
    // Reset grid data to ensure the next step starts fresh
    grid = VTKUnstructuredGrid();
  }
};

// Implementation of VTKDataLogger member functions
template <typename T>
void VTKDataLogger<T>::writeVTUFile(const std::string& filename) {
  if (onlyRankZero && mpiRank != 0) {
    return;
  }

  std::ofstream out(filename);
  out << "<?xml version=\"1.0\"?>\n";
  out << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
  out << "  <UnstructuredGrid>\n";
  writeFieldData(out);
  out << "    <Piece NumberOfPoints=\"" << grid.GetPoints().GetPositions().size()
      << "\" NumberOfCells=\"" << grid.GetCells().GetConnectivity().size() << "\">\n";

  writePoints(out);
  writeCells(out);
  writePointData(out);

  out << "    </Piece>\n";
  out << "  </UnstructuredGrid>\n";
  out << "</VTKFile>\n";
}

template <typename T>
void VTKDataLogger<T>::writePVTUFile(const std::string& pfilename, const std::string& step_basename) {
  std::ofstream out(pfilename);
  out << "<?xml version=\"1.0\"?>\n";
  out << "<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
  out << "  <PUnstructuredGrid GhostLevel=\"0\">\n";

  writePFieldData(out);
  writePPointData(out);

  out << "    <PPoints>\n";
  out << "      <PDataArray type=\"Float32\" Name=\"Points\" NumberOfComponents=\"3\"/>\n";
  out << "    </PPoints>\n";

  for (int i = 0; i < (onlyRankZero ? 1 : mpiSize); ++i) {
    std::ostringstream piece_source_ss;
    // Path is relative to the .pvtu file, which is in the parent directory of the .vtu files.
    piece_source_ss << "data/" << step_basename << "_rank_" << std::setw(4) << std::setfill('0') << i
                    << ".vtu";
    out << "    <Piece Source=\"" << piece_source_ss.str() << "\"/>\n";
  }

  out << "  </PUnstructuredGrid>\n";
  out << "</VTKFile>\n";
}

template <typename T>
void VTKDataLogger<T>::writeFieldData(std::ofstream& out) {
  const auto& arrays = grid.GetFieldData().GetArrays();
  if (!arrays.empty()) {
    out << "    <FieldData>\n";
    for (const auto& array : arrays) {
      out << "      <DataArray type=\"" << array->GetDataType()
          << "\" Name=\"" << array->GetName()
          << "\" NumberOfTuples=\"" << array->GetNumberOfTuples()
          << "\" format=\"ascii\">\n";
      out << "        " << array->GetDataAsString() << "\n";
      out << "      </DataArray>\n";
    }
    out << "    </FieldData>\n";
  }
}

template <typename T>
void VTKDataLogger<T>::writePoints(std::ofstream& out) {
  const auto& positions = grid.GetPoints().GetPositions();
  out << "      <Points>\n";
  out << "        <DataArray type=\"Float32\" Name=\"Points\" NumberOfComponents=\"3\" format=\"ascii\">\n";
  std::ostringstream oss;
  for (const auto& pos : positions) {
    oss << "          " << pos[0] << " " << pos[1] << " " << pos[2] << "\n";
  }
  out << oss.str();
  out << "        </DataArray>\n";
  out << "      </Points>\n";
}

template <typename T>
void VTKDataLogger<T>::writeCells(std::ofstream& out) {
  const auto& connectivity = grid.GetCells().GetConnectivity();

  out << "      <Cells>\n";

  std::ostringstream conn_oss, off_oss, type_oss;
  int offset = 0;
  for (const auto& cell : connectivity) {
    conn_oss << "          ";
    for (const auto& idx : cell) {
      conn_oss << idx << " ";
    }
    conn_oss << "\n";
    offset += cell.size();
    off_oss << "          " << offset << "\n";
    type_oss << "          " << "1\n";  // VTK_VERTEX = 1
  }

  out << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
  out << conn_oss.str();
  out << "        </DataArray>\n";

  out << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
  out << off_oss.str();
  out << "        </DataArray>\n";

  out << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
  out << type_oss.str();
  out << "        </DataArray>\n";

  out << "      </Cells>\n";
}

template <typename T>
void VTKDataLogger<T>::writePointData(std::ofstream& out) {
  const auto& arrays = grid.GetPointData().GetArrays();
  if (!arrays.empty()) {
    out << "      <PointData>\n";
    for (const auto& array : arrays) {
      out << "        <DataArray type=\"" << array->GetDataType()
          << "\" Name=\"" << array->GetName()
          << "\" NumberOfComponents=\"" << array->GetNumberOfComponents()
          << "\" format=\"ascii\">\n";
      out << "          " << array->GetDataAsString() << "\n";
      out << "        </DataArray>\n";
    }
    out << "      </PointData>\n";
  }
}
template <typename T>
void VTKDataLogger<T>::writePFieldData(std::ofstream& out) {
  const auto& arrays = grid.GetFieldData().GetArrays();
  if (!arrays.empty()) {
    out << "    <PFieldData>\n";
    for (const auto& array : arrays) {
      out << "      <PDataArray type=\"" << array->GetDataType()
          << "\" Name=\"" << array->GetName()
          << "\" NumberOfComponents=\"" << array->GetNumberOfComponents()
          << "\"/>\n";
    }
    out << "    </PFieldData>\n";
  }
}

template <typename T>
void VTKDataLogger<T>::writePPointData(std::ofstream& out) {
  const auto& arrays = grid.GetPointData().GetArrays();
  if (!arrays.empty()) {
    out << "    <PPointData>\n";
    for (const auto& array : arrays) {
      out << "      <PDataArray type=\"" << array->GetDataType()
          << "\" Name=\"" << array->GetName()
          << "\" NumberOfComponents=\"" << array->GetNumberOfComponents()
          << "\"/>\n";
    }
    out << "    </PPointData>\n";
  }
}
