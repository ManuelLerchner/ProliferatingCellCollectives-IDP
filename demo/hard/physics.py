import numpy as np
from scipy.linalg import block_diag


def make_particle_mobility_matrix(L):
    """
    Create a mobility matrix for n particles.
    Parameters:
    L (array-like): Length vector
    i (int): Index of the particle
    Returns:
    np.ndarray: The mobility matrix
    """
    # Create the mobility matrix
    def makeM_particle(i):
        M = np.zeros((6, 6))

        M[0:3, 0:3] = np.eye(3) * L[i]
        M[3:6, 3:6] = np.eye(3) * 12 / (L[i]**3)
        return M

    M = block_diag(*[makeM_particle(i) for i in range(len(L))])
    return M



def make_map(C):
    """
    Constructs the G map for a system of N particles.

    Parameters:
    positions_and_quaternions: Array containing position and quaternion data
                              for each particle, where each particle has:
                              [x, y, z, s, w_x, w_y, w_z]

    Returns:
    G: The generalized mobility matrix
    """
    N = len(C) // 7  # Assuming 7 values per particle

    def create_Xi(s, w):
        """Creates the Xi matrix for a particle with quaternion [s, w]"""
        W_matrix = np.array([
            [0, -w[2], w[1]],
            [w[2], 0, -w[0]],
            [-w[1], w[0], 0]
        ])
        A= 0.5 * np.block([
            [-w.reshape(1, 3)],  # -w^T as a row vector
            [s*np.eye(3) - W_matrix]  # sI - W
        ])
        return A

    def create_Gn(particle_idx):
        """Creates the G_n matrix for particle n"""
        particle_data = C[particle_idx*7:(
            particle_idx+1)*7]
        s = particle_data[3]
        w = particle_data[4:7]
        Xi_matrix = create_Xi(s, w)
        return block_diag(np.eye(3), Xi_matrix)

    # Create G as a block diagonal matrix of all G_n matrices
    G_blocks = [create_Gn(n) for n in range(N)]
    return block_diag(*G_blocks)
