import numpy as np


def rotateVectorOfPositions(q, vs):
    """
    Rotate a vector of positions using a quaternion.

    Parameters:
    q (array-like): Quaternion [s, v1, v2, v3]
    v (array-like): Vector of positions

    Returns:
    np.ndarray: Rotated vector of positions
    """
    s, v1, v2, v3 = q
    ww = s**2
    wx = s * v1
    wy = s * v2
    wz = s * v3
    xx = v1**2
    xy = v1 * v2
    xz = v1 * v3
    yy = v2**2
    yz = v2 * v3
    zz = v3**2

    r00 = ww + xx - yy - zz
    r01 = 2 * (xy - wz)
    r02 = 2 * (xz + wy)
    r10 = 2 * (xy + wz)
    r11 = ww - xx + yy - zz
    r12 = 2 * (yz - wx)
    r20 = 2 * (xz - wy)
    r21 = 2 * (yz + wx)
    r22 = ww - xx - yy + zz

    rotated_positions = np.zeros_like(vs)

    for i in range(len(vs)):
        rotated_positions[i][0] = r00 * vs[i][0] + r01 * vs[i][1] + r02 * vs[i][2]
        rotated_positions[i][1] = r10 * vs[i][0] + r11 * vs[i][1] + r12 * vs[i][2]
        rotated_positions[i][2] = r20 * vs[i][0] + r21 * vs[i][1] + r22 * vs[i][2]

    return rotated_positions
