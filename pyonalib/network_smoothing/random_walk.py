"""
Module for network smoothing by "random walk with restarts" (RWWR) process [1].

The process takes an [N x N] adjacency matrix defining a graph to do smoothing on, and a [M x N] data matrix to be smoothed.

The smoothing is defined by the iterative process

    F_{t+1} = \alpha F_{t}A + (1-\alpha)F_0

whereby values from F_0 (the initial data matrix) are propagated through the graph defined by the adjacency matrix A,
by a random walk process where a_{i,j} is the probability of walking from node i to node j, and (1-alpha) is the restart probability.

The iterative process is guaranteed to converge so long as the rows of A is row-normalized.

[1]: random walk network smoothing paper...
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from scipy import io, linalg, sparse

def smooth_by_iterations(adj_matrix, expr_matrix, alpha, tol=1e-9, max_iter=100):
    """
    "Brute force" calculation of RWWR process.

        adj_matrix:    [N x N] matrix. Will be row-normalized.
        expr_matrix:   [M x N] DataFrame to be smoothed.
        alpha:         1 - the restart probability
        tol, max_iter: iterative process will stop when change is < tol or max_iter is reached
    """
    Anorm = np.matrix(normalize(adj_matrix, axis=1, norm='l1'))
    Fs = [expr_matrix]
    for i in range(1, max_iter+1):
        Fs.append(alpha * np.dot(Fs[i-1], Anorm) + (1-alpha) * Fs[0])
        if linalg.norm(Fs[-1] - Fs[-2]) < tol:
            print("Converged in {} iterations".format(i))
            break
    return pd.DataFrame(Fs[-1], index=expr_matrix.index, columns=expr_matrix.columns)

def compute_smoothing_kernel(adj_matrix, alpha):
    """
    The influence matrix K = k_{i,j} where k_{i,j} is the resulting value of node j following a RWWR process starting at node i with a value of 1.
    The influence matrix is the effective smoothing kernel of the closed-form solution of RWWR. The closed form solution is:

        F_\infty = F_0 (1-\alpha) (I - \alpha A)^{-1}

    which can be written as

        F_\infty = F_0 * K

    where K is the smoothing kernel.

    This function computes K using matrix inversion.

        adj_matrix: [N x N] matrix. will be row-normalized
        alpha:      1 - the restart probability
    """
    Anorm = normalize(adj_matrix, axis=1, norm='l1')
    I = np.eye(Anorm.shape[0])
    influence_matrix = (1-alpha) * linalg.inv(I-alpha*Anorm)
    return influence_matrix

def smooth_by_mtx_inv(adj_matrix, expr_matrix, alpha):
    """
    Performs network smoothing of expr_matrix on the graph defined by adj_matrix using the closed form solution.
    See `compute_smoothing_kernel`.

        adj_matrix:  [N x N] adjacency matrix defining the graph to smooth on. Will be row-normalized.
        expr_matrix: [M x N] the data matrix to be smoothed.
        alpha:       1 - the restart probability.
    """
    K = compute_smoothing_kernel(adj_matrix, alpha)
    return pd.DataFrame(np.dot(expr_matrix, K), index=expr_matrix.index,
                       columns=expr_matrix.columns)

def smooth_by_linalg_solve(adj_matrix, expr_matrix, alpha):
    """
    Performs network smoothing by RWWR by solving the matrix equation

        F_\infty \left(I - \alpha A\right) = F_0 (1-\alpha)
    
    using `scipy.linalg.solve`. This is equivalent to the closed form solution in `compute_smoothing_kernel`,
    but avoids inversions of a large matrix and is potentially both faster and more numerically stable,
    though different cases present different performance.

        adj_matrix:  [N x N] adjacency matrix defining the graph to smooth on. Will be row-normalized.
        expr_matrix: [M x N] the data matrix to be smoothed.
        alpha:       1 - the restart probability.
    """
    Anorm = normalize(adj_matrix, axis=1, norm='l1')
    I = np.eye(Anorm.shape[0])
    solution = linalg.solve((I-alpha*Anorm).T, expr_matrix.T*(1-alpha)).T
    return pd.DataFrame(solution, index=expr_matrix.index, columns=expr_matrix.columns)
