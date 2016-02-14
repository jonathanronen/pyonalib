import numpy as np
from warnings import warn
from scipy import sparse, stats


def quantile_normalize(F):
    """
    Quantile normalizes column-data matrix (features X samples matrix).
    Each row in the input matrix represents a feature, and each column represents a sample.
    The quantile-normalization makes sure every sample has the same *distribution* of feature weights.
    Algorithm from wikipedia or something.
    """
    if sparse.issparse(F):
        warn("Calling todense() on sparse matrix. This might crash your system.")
        F = F.todense()
    ranks = np.empty(F.shape)
    for colidx in range(ranks.shape[1]):
        ranks[:,colidx] = stats.rankdata(F[:,colidx], method='min')-1
    Fsorted = np.sort(F, axis=0)
    rowmeans = np.mean(Fsorted, axis=1)
    Fnorm = F.copy()
    for rowidx in range(ranks.shape[0]):
        Fnorm[ranks==rowidx] = float(rowmeans[rowidx])
    return Fnorm