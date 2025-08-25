import os
import shutil
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
from constraint import ConstraintBlock
from quaternion import getDirectionVector


@dataclass
class SimulationState:
    """Container for simulation state at a given iteration/timestep"""

    C: np.ndarray
    l: np.ndarray
    L: np.ndarray
    max_overlap: float
    forces: np.ndarray
    torques: np.ndarray
    impedance: np.ndarray
    constraint_iterations: int
    avg_bbpgd_iterations: int
    l0: float
    constraints: list[ConstraintBlock]


@dataclass
class VTKField:
    """Represents a field to be written to VTK files"""
    name: str
    data: np.ndarray
    components: int = 1
    data_type: str = "Float32"
    location: str = "point"  # "point", "cell", or "field"

    def __post_init__(self):
        self.data = np.array(self.data)
        if self.components == 1 and self.data.ndim > 1:
            self.data = self.data.flatten()


@dataclass
class VTKFieldData:
    """Container for field data (global simulation metadata)"""
    fields: Dict[str, Union[float, int, str]] = field(default_factory=dict)

    def add_field(self, name: str, value: Union[float, int, str], data_type: str = None):
        """Add a field data entry"""
        if data_type is None:
            if isinstance(value, int):
                data_type = "Int32"
            elif isinstance(value, float):
                data_type = "Float32"
            else:
                data_type = "String"

        self.fields[name] = {"value": value, "type": data_type}


@dataclass
class VTKGeometry:
    """Container for VTK geometry data"""
    positions: np.ndarray
    connectivity: Optional[np.ndarray] = None
    offsets: Optional[np.ndarray] = None
    cell_types: Optional[np.ndarray] = None

    def __post_init__(self):
        self.positions = np.array(self.positions)
        n_points = len(self.positions)

        # Default to vertex cells if no connectivity specified
        if self.connectivity is None:
            self.connectivity = np.arange(n_points)
            self.offsets = np.arange(1, n_points + 1)
            self.cell_types = np.ones(
                n_points, dtype=np.uint8)  # VTK_VERTEX = 1


class VTKDataExtractor(ABC):
    """Abstract base class for extracting simulation data for VTK output"""

    @abstractmethod
    def extract_geometry(self, state: Any) -> VTKGeometry:
        """Extract geometry (positions, connectivity) from simulation state"""
        pass

    @abstractmethod
    def extract_point_data(self, state: Any) -> List[VTKField]:
        """Extract point data fields from simulation state"""
        pass

    @abstractmethod
    def extract_cell_data(self, state: Any) -> List[VTKField]:
        """Extract cell data fields from simulation state"""
        pass

    @abstractmethod
    def extract_field_data(self, state: Any, timestep: int, dt: float, elapsed_time: float) -> VTKFieldData:
        """Extract field data (global metadata) from simulation state"""
        pass


class GenericVTKLogger:
    """
    Generic logger for writing PVTU (parallel VTK Unstructured Grid) files.
    Can be customized for different simulation types through data extractors.
    """

    def __init__(self, output_dir: str, prefix: str = 'simulation_',
                 data_extractor: Optional[VTKDataExtractor] = None):
        self.output_dir = output_dir
        self.prefix = prefix
        self.data_dir = os.path.join(output_dir, 'data')
        self.data_extractor = data_extractor
        self.t_start = time.monotonic_ns()

        # Setup output directories
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir)
        os.makedirs(self.data_dir)

    def get_vtu_filename(self, timestep: int) -> str:
        return f"{self.prefix}{timestep:07d}.vtu"

    def get_pvtu_filename(self, timestep: int) -> str:
        return f"{self.prefix}{timestep:07d}.pvtu"

    def write_timestep_pvtu(self, timestep: int, point_fields: List[VTKField],
                            cell_fields: List[VTKField], field_data: VTKFieldData):
        """Write PVTU file referencing the VTU file for this timestep"""
        pvtu_filename = self.get_pvtu_filename(timestep)
        vtu_filename = self.get_vtu_filename(timestep)
        pvtu_path = os.path.join(self.output_dir, pvtu_filename)

        with open(pvtu_path, 'w') as f:
            f.write('<?xml version="1.0" encoding="UTF-8" standalone="no" ?>\n')
            f.write(
                '<VTKFile byte_order="LittleEndian" type="PUnstructuredGrid" version="0.1">\n')
            f.write('  <PUnstructuredGrid GhostLevel="0">\n')

            # Field data
            if field_data.fields:
                f.write('    <PFieldData>\n')
                for name, info in field_data.fields.items():
                    f.write(f'      <PDataArray Name="{name}" NumberOfComponents="1" '
                            f'format="ascii" type="{info["type"]}"/>\n')
                f.write('    </PFieldData>\n')

            # Point data
            if point_fields:
                f.write('    <PPointData>\n')
                for field in point_fields:
                    f.write(f'      <PDataArray Name="{field.name}" '
                            f'NumberOfComponents="{field.components}" '
                            f'format="ascii" type="{field.data_type}"/>\n')
                f.write('    </PPointData>\n')

            # Cell data
            if cell_fields:
                f.write('    <PCellData>\n')
                for field in cell_fields:
                    f.write(f'      <PDataArray Name="{field.name}" '
                            f'NumberOfComponents="{field.components}" '
                            f'format="ascii" type="{field.data_type}"/>\n')
                f.write('    </PCellData>\n')
            else:
                f.write('    <PCellData/>\n')

            # Points
            f.write('    <PPoints>\n')
            f.write('      <PDataArray Name="positions" NumberOfComponents="3" '
                    'format="ascii" type="Float32"/>\n')
            f.write('    </PPoints>\n')

            # Cells
            f.write('    <PCells>\n')
            f.write('      <PDataArray Name="connectivity" NumberOfComponents="1" '
                    'format="ascii" type="Int32"/>\n')
            f.write('      <PDataArray Name="offsets" NumberOfComponents="1" '
                    'format="ascii" type="Int32"/>\n')
            f.write('      <PDataArray Name="types" NumberOfComponents="1" '
                    'format="ascii" type="UInt8"/>\n')
            f.write('    </PCells>\n')

            f.write(f'    <Piece Source="./data/{vtu_filename}"/>\n')
            f.write('  </PUnstructuredGrid>\n')
            f.write('</VTKFile>\n')

    def add_vtu_file(self, timestep: int, dt: float, geometry: VTKGeometry,
                     point_fields: List[VTKField], cell_fields: List[VTKField],
                     field_data: VTKFieldData):
        """Create a VTU file with simulation data for the given timestep"""
        vtu_filename = self.get_vtu_filename(timestep)
        vtu_path = os.path.join(self.data_dir, vtu_filename)

        n_points = len(geometry.positions)
        n_cells = len(
            geometry.offsets) if geometry.offsets is not None else n_points

        with open(vtu_path, 'w') as f:
            f.write('<?xml version="1.0" encoding="UTF-8" standalone="no" ?>\n')
            f.write(
                '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
            f.write('  <UnstructuredGrid>\n')

            # Field Data
            if field_data.fields:
                f.write('    <FieldData>\n')
                for name, info in field_data.fields.items():
                    f.write(f'      <DataArray type="{info["type"]}" Name="{name}" '
                            f'NumberOfTuples="1">{info["value"]}</DataArray>\n')
                f.write('    </FieldData>\n')

            f.write(
                f'    <Piece NumberOfCells="{n_cells}" NumberOfPoints="{n_points}">\n')

            # Point Data
            if point_fields:
                f.write('      <PointData>\n')
                for field in point_fields:
                    self._write_data_array(f, field, "        ")
                f.write('      </PointData>\n')

            # Cell Data
            if cell_fields:
                f.write('      <CellData>\n')
                for field in cell_fields:
                    self._write_data_array(f, field, "        ")
                f.write('      </CellData>\n')
            else:
                f.write('      <CellData>\n')
                f.write('      </CellData>\n')

            # Points
            f.write('      <Points>\n')
            f.write('        <DataArray type="Float32" NumberOfComponents="3" '
                    'Name="positions" format="ascii">\n')
            for pos in geometry.positions:
                if len(pos) == 2:  # 2D case
                    f.write(f'          {pos[0]} {pos[1]} 0.0\n')
                else:  # 3D case
                    f.write(f'          {pos[0]} {pos[1]} {pos[2]}\n')
            f.write('        </DataArray>\n')
            f.write('      </Points>\n')

            # Cells
            f.write('      <Cells>\n')

            # Connectivity
            f.write(
                '        <DataArray type="Int32" Name="connectivity" format="ascii">\n')
            for conn in geometry.connectivity:
                f.write(f'          {conn}\n')
            f.write('        </DataArray>\n')

            # Offsets
            f.write(
                '        <DataArray type="Int32" Name="offsets" format="ascii">\n')
            for offset in geometry.offsets:
                f.write(f'          {offset}\n')
            f.write('        </DataArray>\n')

            # Cell types
            f.write('        <DataArray type="UInt8" Name="types" format="ascii">\n')
            for cell_type in geometry.cell_types:
                f.write(f'          {cell_type}\n')
            f.write('        </DataArray>\n')

            f.write('      </Cells>\n')
            f.write('    </Piece>\n')
            f.write('  </UnstructuredGrid>\n')
            f.write('</VTKFile>\n')

    def _write_data_array(self, f, field: VTKField, indent: str):
        """Write a data array to the VTU file"""
        f.write(f'{indent}<DataArray type="{field.data_type}" '
                f'NumberOfComponents="{field.components}" '
                f'Name="{field.name}" format="ascii">\n')

        if field.components == 1:
            for value in field.data.flatten():
                f.write(f'{indent}  {value}\n')
        else:
            for row in field.data.reshape(-1, field.components):
                f.write(f'{indent}  {" ".join(map(str, row))}\n')

        f.write(f'{indent}</DataArray>\n')

    def log_timestep(self, timestep: int, dt: float, state: Any):
        """Log a complete timestep using the data extractor"""
        if self.data_extractor is None:
            raise ValueError("No data extractor provided")

        elapsed_time = (time.monotonic_ns() - self.t_start) / 1e9

        geometry = self.data_extractor.extract_geometry(state)
        point_fields = self.data_extractor.extract_point_data(state)
        cell_fields = self.data_extractor.extract_cell_data(state)
        field_data = self.data_extractor.extract_field_data(
            state, timestep, dt, elapsed_time)

        self.add_vtu_file(timestep, dt, geometry,
                          point_fields, cell_fields, field_data)
        self.write_timestep_pvtu(
            timestep, point_fields, cell_fields, field_data)

    def clear_data_folder(self):
        """Clear all VTU files from the data folder"""
        for fname in os.listdir(self.data_dir):
            print(f"Removing {fname}")
            os.remove(os.path.join(self.data_dir, fname))


class GenericSimulationLogger:
    """Generic simulation logger using VTK output"""

    def __init__(self, vtk_logger: GenericVTKLogger, log_every_n_iterations: int = 1):
        self.vtk_logger = vtk_logger
        self.log_every_n_iterations = log_every_n_iterations

    def should_log(self, iteration: int) -> bool:
        """Check if this iteration should be logged"""
        return iteration % self.log_every_n_iterations == 0

    def log_timestep_complete(self, iteration: int, dt: float, state: Any):
        """Log timestep if it should be logged"""
        if self.should_log(iteration):
            self.vtk_logger.log_timestep(iteration, dt, state)


# Example implementation for the original bacteria simulation
class BacteriaDataExtractor(VTKDataExtractor):
    """Data extractor for bacteria particle simulation"""

    def extract_geometry(self, state) -> VTKGeometry:
        """Extract positions from bacteria simulation state"""
        n_particles = len(state.l)
        positions = np.zeros((n_particles, 3))
        for i in range(n_particles):
            positions[i] = state.C[7*i:7*i+3]
        return VTKGeometry(positions)

    def extract_point_data(self, state) -> List[VTKField]:
        """Extract point data from bacteria simulation state"""
        n_particles = len(state.l)

        quaternions = np.zeros((n_particles, 4))
        for i in range(n_particles):
            quaternions[i] = state.C[7*i+3:7*i+7]

        directions = np.array([getDirectionVector(q)
                               for q in quaternions])

        lengths = np.column_stack([
            state.l,
            0.5 * state.l0 * np.ones(n_particles),
            0.5 * state.l0 * np.ones(n_particles)
        ])

        return [
            VTKField("forces", state.forces, 3),
            VTKField("torques", state.torques, 3),
            VTKField("directions", directions, 3),
            VTKField("impedance", state.impedance, 1),
            VTKField("lengths", lengths, 3),
            VTKField("typeIds", np.zeros(n_particles, dtype=int), 1, "Int32"),
            VTKField("ids", np.arange(n_particles, dtype=int), 1, "Int32"),
        ]

    def extract_cell_data(self, state) -> List[VTKField]:
        """No cell data for bacteria simulation"""
        return []

    def extract_field_data(self, state, timestep: int, dt: float, elapsed_time: float) -> VTKFieldData:
        """Extract global metadata from bacteria simulation state"""
        field_data = VTKFieldData()
        field_data.add_field("elapsed_time", elapsed_time)
        field_data.add_field("simulation_time", dt * timestep)
        field_data.add_field("avg_bbpgd_iterations",
                             state.avg_bbpgd_iterations)
        field_data.add_field("constraint_iterations",
                             state.constraint_iterations)
        if len(state.constraints) != 0:
            pass
        field_data.add_field("max_overlap", state.max_overlap)
        field_data.add_field("num_particles", len(state.l))
        return field_data


class ConstraintDataExtractor(VTKDataExtractor):
    """Data extractor for constraint simulation"""

    def extract_geometry(self, state) -> VTKGeometry:
        """Extract positions from constraint simulation state"""
        n_constraints = len(state.constraints)
        positions = np.zeros((n_constraints, 3))
        for i in range(n_constraints):
            constraint = state.constraints[i]
            constraint_center = (constraint.labI + constraint.labJ) / 2
            positions[i] = constraint_center
        return VTKGeometry(positions)

    def extract_point_data(self, state) -> List[VTKField]:
        """Extract point data from constraint simulation state"""
        n_constraints = len(state.constraints)
        normals = np.zeros((n_constraints, 3))
        deltas = np.zeros((n_constraints, 1))
        phases = np.zeros((n_constraints, 1))
        for i in range(n_constraints):
            constraint = state.constraints[i]
            normals[i] = constraint.normI
            deltas[i] = constraint.delta0
            phases[i] = constraint.phase
        return [
            VTKField("normals", normals, 3),
            VTKField("deltas", deltas, 1),
            VTKField("phases", phases, 1),
        ]

    def extract_cell_data(self, state) -> List[VTKField]:
        """Extract cell data from constraint simulation state"""
        return []

    def extract_field_data(self, state, timestep: int, dt: float, elapsed_time: float) -> VTKFieldData:
        """Extract global metadata from constraint simulation state"""
        field_data = VTKFieldData()
        field_data.add_field("num_constraints", len(state.constraints))
        return field_data


# Usage example:
def create_bacteria_logger(output_dir: str) -> GenericSimulationLogger:
    """Create a logger configured for bacteria simulation"""
    extractor = BacteriaDataExtractor()
    vtk_logger = GenericVTKLogger(
        output_dir, 'Bacteria_Particles_', extractor)
    return GenericSimulationLogger(vtk_logger)


def create_constraint_logger(output_dir: str) -> GenericSimulationLogger:
    """Create a logger configured for constraint simulation"""
    extractor = ConstraintDataExtractor()
    vtk_logger = GenericVTKLogger(
        output_dir, 'Constraint_Particles_', extractor)
    return GenericSimulationLogger(vtk_logger)
