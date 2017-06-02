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

def _need_transpose(expr_matrix, adj_matrix):
    """
    The smoothing process is defined on a matrix of data rows. This checks the dimensions of E and A to see if E should be transposed.
    """
    return expr_matrix.shape[1] != adj_matrix.shape[0]

def smooth_by_iterations(expr_matrix, adj_matrix, alpha, tol=1e-9, max_iter=100, transpose='auto'):
    """
    "Brute force" calculation of RWWR process.

        adj_matrix:    [N x N] matrix. Will be row-normalized.
        expr_matrix:   [M x N] DataFrame to be smoothed.
        alpha:         1 - the restart probability
        tol, max_iter: Iterative process will stop when change is < tol or max_iter is reached
        transpose:     If True, `expr_matrix` is [N x M] and will be transposed prior to smoothing and before returning. If 'auto', guess from dimensions.
    """
    transpose = _need_transpose(expr_matrix, adj_matrix) if transpose=='auto' else transpose
    if transpose:
        expr_matrix = expr_matrix.T
    Anorm = np.matrix(normalize(adj_matrix, axis=1, norm='l1'))
    Fs = [expr_matrix]
    for i in range(1, max_iter+1):
        Fs.append(alpha * np.dot(Fs[i-1], Anorm) + (1-alpha) * Fs[0])
        if linalg.norm(Fs[-1] - Fs[-2]) < tol:
            print("Converged in {} iterations".format(i))
            break
    ret = pd.DataFrame(Fs[-1], index=expr_matrix.index, columns=expr_matrix.columns)
    return ret.T if transpose else ret

def compute_smoothing_kernel(adj_matrix, alpha):
    """
    The influence matrix K = k_{i,j} where k_{i,j} is the resulting value of node j following a RWWR process starting at node i with a value of 1.
    The influence matrix is the effective smoothing kernel of the closed-form solution of RWWR. The closed form solution is:

        F_\infty = F_0 (1-\alpha) (I - \alpha A)^{-1}

    which can be written as

        F_\infty = F_0 * K

    where K is the [N x N] smoothing kernel and F_0 is the [M x N] data matrix to be network-smoothed.

    This function computes K using matrix inversion.

        adj_matrix: [N x N] matrix. will be row-normalized
        alpha:      1 - the restart probability
    """
    Anorm = normalize(adj_matrix, axis=1, norm='l1')
    I = np.eye(Anorm.shape[0])
    influence_matrix = (1-alpha) * linalg.inv(I-alpha*Anorm)
    return influence_matrix

def smooth_with_kernel(expr_matrix, kernel, transpose='auto'):
    """
    Perform smoothing of `expr_matrix` using provided kernel.

            expr_matrix: [M x N] the data matrix to be smoothed.
            kernel:      [N x N] smoothing kernel.
            transpose:   If True, `expr_matrix` is [N x M] and will be transposed prior to smoothing and before returning. If 'auto', guess from dimensions.
    """
    transpose = _need_transpose(expr_matrix, kernel) if transpose=='auto' else transpose

    if transpose:
        return pd.DataFrame(np.dot(expr_matrix.T, K), index=expr_matrix.index,
                           columns=expr_matrix.columns).T
    else:
        return pd.DataFrame(np.dot(expr_matrix, K), index=expr_matrix.index,
                           columns=expr_matrix.columns)

def smooth_by_mtx_inv(expr_matrix, adj_matrix, alpha, transpose='auto'):
    """
    Performs network smoothing of expr_matrix on the graph defined by adj_matrix using the closed form solution.
    See `compute_smoothing_kernel`.

        expr_matrix: [M x N] the data matrix to be smoothed.
        adj_matrix:  [N x N] adjacency matrix defining the graph to smooth on. Will be row-normalized.
        alpha:       1 - the restart probability.
        transpose:   If True, `expr_matrix` is [N x M] and will be transposed prior to smoothing and before returning. If 'auto', guess from dimensions.
    """
    K = compute_smoothing_kernel(adj_matrix, alpha)
    return smooth_with_kernel(expr_matrix, K, transpose=transpose)

def smooth_by_linalg_solve(expr_matrix, adj_matrix, alpha, transpose='auto'):
    """
    Performs network smoothing by RWWR by solving the matrix equation

        F_\infty \left(I - \alpha A\right) = F_0 (1-\alpha)
    
    using `scipy.linalg.solve`. This is equivalent to the closed form solution in `compute_smoothing_kernel`,
    but avoids inversions of a large matrix and is potentially both faster and more numerically stable,
    though different cases present different performance.

        expr_matrix: [M x N] the data matrix to be smoothed.
        adj_matrix:  [N x N] adjacency matrix defining the graph to smooth on. Will be row-normalized.
        alpha:       1 - the restart probability.
        transpose:   If True, `expr_matrix` is [N x M] and will be transposed prior to smoothing and before returning. If 'auto', guess from dimensions.
    """
    transpose = _need_transpose(expr_matrix, adj_matrix) if transpose=='auto' else transpose
    if transpose:
        expr_matrix = expr_matrix.T
    Anorm = normalize(adj_matrix, axis=1, norm='l1')
    I = np.eye(Anorm.shape[0])
    solution = linalg.solve((I-alpha*Anorm).T, expr_matrix.T*(1-alpha)).T

    ret = pd.DataFrame(solution, index=expr_matrix.index, columns=expr_matrix.columns)
    return ret.T if transpose else ret
