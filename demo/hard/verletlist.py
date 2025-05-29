import numpy as np
from collections import defaultdict
from itertools import product


class LinkedCellList:
    def __init__(self, cutoff_distance, skin_distance=0.5):
        """
        Linked-cell list for neighbor finding with capsules (3D particles).

        Args:
            cutoff_distance: Interaction cutoff distance between capsule surfaces.
            skin_distance: Optional buffer to reduce update frequency.
        """
        self.cutoff = cutoff_distance
        self.skin = skin_distance
        self.cell_size = self.cutoff + self.skin
        self.grid = defaultdict(list)

        self.last_positions = None
        self.last_lengths = None
        self.cell_indices = None

    def position_to_cell_index(self, pos):
        return tuple(np.floor(pos / self.cell_size).astype(int))

    def needs_update(self, positions, lengths):
        if self.last_positions is None or self.last_lengths is None:
            return True
        
        if len(positions) != len(self.last_positions) or len(lengths) != len(self.last_lengths):
            return True

        max_disp_sq = np.max(
            np.sum((positions - self.last_positions) ** 2, axis=1))
        max_growth = np.max(np.abs(lengths - self.last_lengths))

        disp_thresh = (self.skin / 3.0) ** 2
        growth_thresh = self.skin / 3.0

        return max_disp_sq > disp_thresh or max_growth > growth_thresh

    def build(self, positions, lengths):
        """
        Build the linked cell structure.
        Args:
            positions: (N, 3) array of capsule center positions.
            lengths: (N,) array of capsule lengths.
        """
        self.grid.clear()
        self.cell_indices = []

        for i, pos in enumerate(positions):
            idx = self.position_to_cell_index(pos)
            self.grid[idx].append(i)
            self.cell_indices.append(idx)

        self.last_positions = positions.copy()
        self.last_lengths = lengths.copy()

    def get_close_pairs(self, positions, lengths, actual_cutoff=None):
        """
        Return pairs of indices whose capsule surfaces are within the actual cutoff.
        Args:
            positions: (N, 3) array of capsule positions.
            lengths: (N,) array of capsule lengths.
            actual_cutoff: Optional override for interaction cutoff.
        Returns:
            List of (i, j) index pairs.
        """
        if actual_cutoff is None:
            actual_cutoff = self.cutoff

        visited_pairs = set()
        close_pairs = []

        for i, pos_i in enumerate(positions):
            len_i = lengths[i]
            cell = self.cell_indices[i]

            for offset in product([-1, 0, 1], repeat=3):
                neighbor_cell = tuple(cell[d] + offset[d] for d in range(3))

                for j in self.grid.get(neighbor_cell, []):
                    if j <= i:
                        continue
                    pair = (i, j)
                    if pair in visited_pairs:
                        continue
                    visited_pairs.add(pair)

                    pos_j = positions[j]
                    len_j = lengths[j]

                    center_dist = np.linalg.norm(pos_i - pos_j)
                    surface_dist = center_dist - (len_i + len_j) / 2.0

                    if surface_dist <= actual_cutoff:
                        close_pairs.append(pair)

        return close_pairs
