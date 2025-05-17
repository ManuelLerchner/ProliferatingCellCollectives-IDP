import numpy as np


def make_particle_position(x, s, w):
    """
    Create a particle representation in 4D space.

    Parameters:
    x (array-like): Center of mass coordinates
    s (float): Scalar component
    w (array-like): Vector component

    Returns:
    np.ndarray: A 4D array representing the particle
    """
    # Ensure x and w are numpy arrays
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)

    # Check dimensions
    if x.shape != (3,) or w.shape != (3,):
        raise ValueError("x and w must be 3D vectors")

    # Create the particle representation
    C = np.concatenate((x, [s], w))

    return C


def make_particle_length(l):
    """
    Create a particle representation with length.

    Parameters:
    l (float): Length of the particle

    Returns:
    np.ndarray: A 1D array representing the length
    """
    # Ensure l is a float
    l = float(l)

    # Create the length representation
    L = np.array([l])

    return L


def make_particle_forces(f, t):
    """
    Create a particle representation with forces.
    Parameters:
    f (array-like): Force vector
    t (array-like): Torque vector
    Returns:
    np.ndarray: A 6D array representing the forces and torques
    """
    # Ensure f and t are numpy arrays
    f = np.asarray(f, dtype=float)
    t = np.asarray(t, dtype=float)

    # Check dimensions
    if f.shape != (3,) or t.shape != (3,):
        raise ValueError("f and t must be 3D vectors")

    # Create the forces and torques representation
    F = np.concatenate((f, t))

    return F


def make_particle_velocity(v, w):
    """
    Create a particle representation with velocity.
    Parameters:
    v (array-like): Linear velocity vector
    w (array-like): Angular velocity vector
    Returns:
    np.ndarray: A 6D array representing the velocities
    """
    # Ensure v and w are numpy arrays
    v = np.asarray(v, dtype=float)
    w = np.asarray(w, dtype=float)

    # Check dimensions
    if v.shape != (3,) or w.shape != (3,):
        raise ValueError("v and w must be 3D vectors")

    # Create the velocity representation
    V = np.concatenate((v, w))

    return V
