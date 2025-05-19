import numpy as np


def collide_stress(dir_i, dir_j, center_i, center_j, h_i, h_j, r_i, r_j, rho, p_loc, q_loc):
    """
    Compute collision stress for a pair of sylinders

    Parameters:
    -----------
    dir_i : numpy.ndarray
        Direction of sylinder I (unit vector)
    dir_j : numpy.ndarray
        Direction of sylinder J (unit vector)
    center_i : numpy.ndarray
        Center location of sylinder I (lab frame)
    center_j : numpy.ndarray
        Center location of sylinder J (lab frame)
    h_i : float
        Length (cylindrical part) of sylinder I
    h_j : float
        Length (cylindrical part) of sylinder J
    r_i : float
        Radius of sylinder I
    r_j : float
        Radius of sylinder J
    rho : float
        Mass density (typically set to 1)
    p_loc : numpy.ndarray
        Location of force on sylinder I (lab frame)
    q_loc : numpy.ndarray
        Location of force on sylinder J (lab frame)

    Returns:
    --------
    numpy.ndarray
        Pairwise stress tensor if gamma = 1
    """
    def initialize_sy_n(r, h, rho):
        """Initialize the tensor integral N for sylinder aligned with z axis"""
        n_i = np.zeros((3, 3))
        beta = h / (2.0 * r)
        n_i[0, 0] = 1.0 / 30.0 * (15.0 * beta + 8)
        n_i[1, 1] = n_i[0, 0]
        n_i[2, 2] = 1.0 / 15.0 * \
            (10.0 * beta**3 + 20.0 * beta**2 + 15.0 * beta + 4.0)
        n_i = n_i * rho * r**5 * np.pi
        return n_i

    def initialize_sy_ga(r, h, rho):
        """Initialize the tensor integral GAMMA for sylinder aligned with z axis"""
        gamma_i = np.zeros((3, 3))
        beta = h / (2.0 * r)
        gamma_i[0, 0] = 1.0 / 30.0 * \
            (20.0 * beta**3 + 40.0 * beta**2 + 45.0 * beta + 16.0)
        gamma_i[1, 1] = gamma_i[0, 0]
        gamma_i[2, 2] = 1.0 / 15.0 * (15 * beta + 8)
        gamma_i = gamma_i * np.pi * r**5 * rho
        return gamma_i

    # Initialize tensors
    n_i_base = initialize_sy_n(r_i, h_i, rho)
    gamma_i_base = initialize_sy_ga(r_i, h_i, rho)
    n_j_base = initialize_sy_n(r_j, h_j, rho)
    gamma_j_base = initialize_sy_ga(r_j, h_j, rho)

    # Extract diagonal values
    a_i = n_i_base[0, 0]
    b_i = n_i_base[2, 2]
    a_j = n_j_base[0, 0]
    b_j = n_j_base[2, 2]

    # Create directional tensors
    dir_i = np.array(dir_i)
    dir_j = np.array(dir_j)
    n_i = a_i * np.eye(3) + (b_i - a_i) * np.outer(dir_i, dir_i)
    n_j = a_j * np.eye(3) + (b_j - a_j) * np.outer(dir_j, dir_j)

    # Inverse gamma values
    a_i = 1.0 / gamma_i_base[0, 0]
    b_i = 1.0 / gamma_i_base[2, 2]
    a_j = 1.0 / gamma_j_base[0, 0]
    b_j = 1.0 / gamma_j_base[2, 2]

    inv_gamma_i = a_i * np.eye(3) + (b_i - a_i) * np.outer(dir_i, dir_i)
    inv_gamma_j = a_j * np.eye(3) + (b_j - a_j) * np.outer(dir_j, dir_j)

    # Force calculation
    # Normalized force direction
    f1 = (q_loc - p_loc) / np.linalg.norm(q_loc - p_loc)
    r_if = np.outer(center_i, -f1)  # Newton's law
    r_jf = np.outer(center_j, f1)

    x_icf = np.cross(p_loc - center_i, -f1)  # Newton's law
    x_jcf = np.cross(q_loc - center_j, f1)

    # Levi-Civita symbol epsilon (3D anti-symmetric tensor)
    # This is a function to compute the Levi-civita tensor product with a vector
    def levi_civita_product(tensor, vector):
        """Compute the product of a tensor with a vector using the Levi-Civita symbol

        This implements the tensor contraction: tensor[i,l] * epsilon[j,k,l] * inv_gamma[k,r] * vector[r]
        where epsilon is the Levi-Civita symbol.
        """
        result = np.zeros((3, 3))

        # Define the Levi-Civita symbol (antisymmetric tensor)
        epsilon = np.zeros((3, 3, 3))
        epsilon[0, 1, 2] = epsilon[1, 2, 0] = epsilon[2, 0, 1] = 1
        epsilon[0, 2, 1] = epsilon[2, 1, 0] = epsilon[1, 0, 2] = -1

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        for r in range(3):
                            # For sylinder I
                            if tensor is n_i:
                                result[i, j] += tensor[i, l] * epsilon[j,
                                                                       k, l] * inv_gamma_i[k, r] * vector[r]
                            # For sylinder J
                            else:
                                result[i, j] += tensor[i, l] * epsilon[j,
                                                                       k, l] * inv_gamma_j[k, r] * vector[r]
        return result

    # Calculate stress tensors SG_I and SG_J using the levi_civita_product function
    sg_i = levi_civita_product(n_i, x_icf)
    sg_j = levi_civita_product(n_j, x_jcf)

    # Final stress tensor
    stress_ij = r_if + r_jf + sg_i + sg_j
    return stress_ij
