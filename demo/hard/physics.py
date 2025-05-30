import numpy as np
from scipy.sparse import block_diag
from scipy.sparse import csr_matrix


def make_particle_mobility_matrix(l, xi):
    """
    Optimized: Create a sparse block diagonal mobility matrix for n particles.
    """
    n = len(l)
    blocks = []

    for i in range(n):
        l_i = l[i]
        diag = 1/xi*np.array([1/l_i]*3 + [12 / (l_i**3)]*3)
        blocks.append(csr_matrix((diag, (range(6), range(6))), shape=(6, 6)))

    return block_diag(blocks, format='csr')


def make_map(C):
    """
    Optimized: Constructs the sparse G matrix for N particles more efficiently.
    """
    N = len(C) // 7
    blocks = []

    for n in range(N):
        data = C[n*7:(n+1)*7]
        s, w = data[3], data[4:7]

        # Xi matrix (4x3)
        wx, wy, wz = w
        Xi = np.array([
            [-wx, -wy, -wz],
            [s, -wz,  wy],
            [wz,  s, -wx],
            [-wy,  wx,  s]
        ]) * 0.5

        # Create block diagonal [I3, Xi]
        I3 = csr_matrix(np.eye(3))
        Xi_sparse = csr_matrix(Xi)
        G_n = block_diag((I3, Xi_sparse), format='csr')
        blocks.append(G_n)

    return block_diag(blocks, format='csr')
