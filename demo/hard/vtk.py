import os
import shutil
from dataclasses import dataclass

import numpy as np
from quaternion import getDirectionVector


class VTKLogger:
    """
    Logger for writing PVTU (parallel VTK Unstructured Grid) files and managing VTU file references per timestep.
    Each timestep produces a .pvtu file referencing a single .vtu file in the data folder.
    """

    def __init__(self, output_dir: str, prefix: str = 'Bacteria_Particles_0_'):
        self.output_dir = output_dir
        self.prefix = prefix
        self.data_dir = os.path.join(output_dir, 'data')

        # completely delete the output directory
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir)
        os.makedirs(self.data_dir)

    def get_vtu_filename(self, timestep: int) -> str:
        return f"{self.prefix}{timestep:07d}.vtu"

    def get_pvtu_filename(self, timestep: int) -> str:
        return f"{self.prefix}{timestep:07d}.pvtu"

    def write_timestep_pvtu(self, timestep: int):
        pvtu_filename = self.get_pvtu_filename(timestep)
        vtu_filename = self.get_vtu_filename(timestep)
        pvtu_path = os.path.join(self.output_dir, pvtu_filename)
        with open(pvtu_path, 'w') as f:
            f.write('<?xml version="1.0" encoding="UTF-8" standalone="no" ?>\n')
            f.write(
                '<VTKFile byte_order="LittleEndian" type="PUnstructuredGrid" version="0.1">\n')
            f.write('  <PUnstructuredGrid GhostLevel="0">\n')

            # Field data for global timestep information
            f.write('    <PFieldData>\n')
            f.write(
                '      <PDataArray Name="timestep" NumberOfComponents="1" format="ascii" type="Int32"/>\n')
            f.write(
                '      <PDataArray Name="avg_bbpgd_iterations" NumberOfComponents="1" format="ascii" type="Float32"/>\n')
            f.write(
                '      <PDataArray Name="constraint_iterations" NumberOfComponents="1" format="ascii" type="Int32"/>\n')
            f.write('    </PFieldData>\n')

            # Point data - per particle data
            f.write('    <PPointData>\n')
            f.write(
                '      <PDataArray Name="forces" NumberOfComponents="3" format="ascii" type="Float32"/>\n')
            f.write(
                '      <PDataArray Name="torques" NumberOfComponents="3" format="ascii" type="Float32"/>\n')
            f.write(
                '      <PDataArray Name="directions" NumberOfComponents="3" format="ascii" type="Float32"/>\n')
            f.write(
                '      <PDataArray Name="stresses" NumberOfComponents="1" format="ascii" type="Float32"/>\n')
            f.write(
                '      <PDataArray Name="lengths" NumberOfComponents="3" format="ascii" type="Float32"/>\n')
            f.write(
                '      <PDataArray Name="typeIds" NumberOfComponents="1" format="ascii" type="Int32"/>\n')
            f.write(
                '      <PDataArray Name="ids" NumberOfComponents="1" format="ascii" type="Int32"/>\n')
            f.write('    </PPointData>\n')

            f.write('    <PCellData/>\n')
            f.write('    <PPoints>\n')
            f.write(
                '      <PDataArray Name="positions" NumberOfComponents="3" format="ascii" type="Float32"/>\n')
            f.write('    </PPoints>\n')
            f.write('    <PCells>\n')
            f.write(
                '      <PDataArray Name="connectivity" NumberOfComponents="1" format="ascii" type="Int32"/>\n')
            f.write(
                '      <PDataArray Name="offsets" NumberOfComponents="1" format="ascii" type="Int32"/>\n')
            f.write(
                '      <PDataArray Name="types" NumberOfComponents="1" format="ascii" type="UInt8"/>\n')
            f.write('    </PCells>\n')
            f.write(f'    <Piece Source="./data/{vtu_filename}"/>\n')
            f.write('  </PUnstructuredGrid>\n')
            f.write('</VTKFile>\n')

    def add_vtu_file(self, timestep: int, positions, forces, torques,
                     directions, stresses, lengths, type_ids, ids, constraint_iterations, avg_bbpgd_iterations):
        """
        Create a VTU file with complete particle data for the given timestep.
        """
        vtu_filename = self.get_vtu_filename(timestep)
        vtu_path = os.path.join(self.data_dir, vtu_filename)

        positions = np.array(positions)
        n_particles = len(positions)

        with open(vtu_path, 'w') as f:
            f.write('<?xml version="1.0" encoding="UTF-8" standalone="no" ?>\n')
            f.write(
                '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
            f.write('  <UnstructuredGrid>\n')

            # Field Data at UnstructuredGrid level - global information for this timestep
            f.write('    <FieldData>\n')
            f.write(
                f'      <DataArray type="Int32" Name="timestep" NumberOfTuples="1">{timestep}</DataArray>\n')
            f.write(
                f'      <DataArray type="Float32" Name="avg_bbpgd_iterations" NumberOfTuples="1">{avg_bbpgd_iterations:.6f}</DataArray>\n')
            f.write(
                f'      <DataArray type="Int32" Name="constraint_iterations" NumberOfTuples="1">{constraint_iterations}</DataArray>\n')
            f.write('    </FieldData>\n')

            f.write(
                f'    <Piece NumberOfCells="{n_particles}" NumberOfPoints="{n_particles}">\n')

            # Point Data - per particle data
            f.write('      <PointData>\n')

            # Forces
            f.write(
                '        <DataArray type="Float32" NumberOfComponents="3" Name="forces" format="ascii">\n')
            for force in forces:
                f.write(
                    f'          {force[0]:.6f} {force[1]:.6f} {force[2]:.6f}\n')
            f.write('        </DataArray>\n')

            # Torques
            f.write(
                '        <DataArray type="Float32" NumberOfComponents="3" Name="torques" format="ascii">\n')
            for torque in torques:
                f.write(
                    f'          {torque[0]:.6f} {torque[1]:.6f} {torque[2]:.6f}\n')
            f.write('        </DataArray>\n')

            # Directions
            f.write(
                '        <DataArray type="Float32" NumberOfComponents="3" Name="directions" format="ascii">\n')
            for direction in directions:
                f.write(
                    f'          {direction[0]:.6f} {direction[1]:.6f} {direction[2]:.6f}\n')
            f.write('        </DataArray>\n')

            # Stresses
            f.write(
                '        <DataArray type="Float32" NumberOfComponents="1" Name="stresses" format="ascii">\n')
            for stress in stresses.flatten():
                f.write(f'          {stress:.6f}\n')
            f.write('        </DataArray>\n')

            # Lengths
            f.write(
                '        <DataArray type="Float32" NumberOfComponents="3" Name="lengths" format="ascii">\n')
            for length in lengths:
                f.write(
                    f'          {length[0]:.6f} {length[1]:.6f} {length[2]:.6f}\n')
            f.write('        </DataArray>\n')

            # Type IDs
            f.write(
                '        <DataArray type="Int32" NumberOfComponents="1" Name="typeIds" format="ascii">\n')
            for type_id in type_ids:
                f.write(f'          {type_id}\n')
            f.write('        </DataArray>\n')

            # IDs
            f.write(
                '        <DataArray type="Int32" NumberOfComponents="1" Name="ids" format="ascii">\n')
            for particle_id in ids:
                f.write(f'          {particle_id}\n')
            f.write('        </DataArray>\n')

            f.write('      </PointData>\n')

            # Cell Data (empty for vertex cells)
            f.write('      <CellData>\n')
            f.write('      </CellData>\n')

            # Points - particle positions
            f.write('      <Points>\n')
            f.write(
                '        <DataArray type="Float32" NumberOfComponents="3" Name="positions" format="ascii">\n')
            for pos in positions:
                f.write(f'          {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n')
            f.write('        </DataArray>\n')
            f.write('      </Points>\n')

            # Cells - vertex cells for particles
            f.write('      <Cells>\n')

            # Connectivity - each vertex cell points to one point
            f.write(
                '        <DataArray type="Int32" Name="connectivity" format="ascii">\n')
            for i in range(n_particles):
                f.write(f'          {i}\n')
            f.write('        </DataArray>\n')

            # Offsets - cumulative count of connectivity entries
            f.write(
                '        <DataArray type="Int32" Name="offsets" format="ascii">\n')
            for i in range(1, n_particles + 1):
                f.write(f'          {i}\n')
            f.write('        </DataArray>\n')

            # Types - VTK_VERTEX = 1
            f.write('        <DataArray type="UInt8" Name="types" format="ascii">\n')
            for i in range(n_particles):
                f.write('          1\n')
            f.write('        </DataArray>\n')

            f.write('      </Cells>\n')

            f.write('    </Piece>\n')
            f.write('  </UnstructuredGrid>\n')
            f.write('</VTKFile>\n')

    def clear_data_folder(self):
        for fname in os.listdir(self.data_dir):
            print(f"Removing {fname}")
            os.remove(os.path.join(self.data_dir, fname))


@dataclass
class SimulationState:
    """Container for simulation state at a given iteration/timestep"""

    C: np.ndarray
    l: np.ndarray
    max_overlap: float
    forces: np.ndarray
    torques: np.ndarray
    stresses: np.ndarray
    constraint_iterations: int
    avg_bbpgd_iterations: int


class VTKSimulationLogger:
    """VTK-based simulation logger"""

    def __init__(self, vtk_logger, log_every_n_iterations: int = 1, log_iterations: bool = False):
        self.vtk_logger = vtk_logger
        self.log_every_n_iterations = log_every_n_iterations
        self.log_iterations = log_iterations

    def log_timestep_complete(self, iteration: int, final_state: SimulationState):
        """Log final timestep data"""
        n_particles = len(final_state.l)
        positions = self._extract_positions(final_state.C, n_particles)

        # Extract quaternions
        quaternions = self._extract_quaternions(final_state.C, n_particles)

        # Log complete timestep data
        self.vtk_logger.add_vtu_file(
            iteration,
            positions,
            final_state.forces,
            final_state.torques,
            np.array([getDirectionVector(q) for q in quaternions]),
            final_state.stresses,
            np.column_stack(
                [final_state.l, 0.5 * np.ones(n_particles), 0.5 * np.ones(n_particles)]),
            np.zeros(n_particles, dtype=int),
            np.arange(n_particles, dtype=int),
            final_state.constraint_iterations,
            final_state.avg_bbpgd_iterations
        )
        self.vtk_logger.write_timestep_pvtu(iteration)

    def _extract_positions(self, C: np.ndarray, n_particles: int) -> np.ndarray:
        """Extract 3D positions from configuration vector"""
        positions = np.zeros((n_particles, 3))
        for i in range(n_particles):
            positions[i] = C[7*i:7*i+3]
        return positions

    def _extract_quaternions(self, C: np.ndarray, n_particles: int) -> np.ndarray:
        """Extract quaternions from configuration vector"""
        quaternions = np.zeros((n_particles, 4))
        for i in range(n_particles):
            quaternions[i] = C[7*i+3:7*i+7]
        return quaternions
