"""
Module with clustering-related functions

@jona
"""
import numpy as np
from warnings import warn
from joblib import Parallel, delayed
from statsmodels.distributions.empirical_distribution import ECDF

try:
    import matplotlib.pyplot as plt
except:
    warn("Error importing matplotlib. Plotting functions will not work.")

class ConsensusClustering:
    """
    Performs consensus clustering
    """
    def __init__(self, ):
        pass


def _subsample_cluster(clustering_method, data, col_sample, row_sample, random_seed=None):
    np.random.seed(random_seed)
    ridx = np.random.choice(data.shape[0], int(np.ceil(row_sample * data.shape[0])), replace=False)
    cidx = np.random.choice(data.shape[1], int(np.ceil(col_sample * data.shape[1])), replace=False)
    data_sub = data[ridx,:][:,cidx]

    yhat = np.empty(data.shape[0])
    yhat[:] = np.nan
    yhat[ridx] = clustering_method(data_sub)
    return yhat


def consensus_matrix(data, clustering_method, col_sample=.8, row_sample=.8,
        n_runs=1000, n_jobs=1, parallel_backend='multiprocessing'):
    """
    Perform consensus clustering on data using cluster_method, return co-clustering matrix
    n_runs times, sample without replacement a col_sample of the columns and row_sample of the rows,
    use clustering_method to cluster subsample of data
    and retutn in the end a co-clustering matrix where C[i,j] is the number of times sample i and
    sample j (in original data matrix) were clustered together, when both were sampled.

    Example:
    --------
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from pyonalib.clustering import consensus_matrix

    # generate random data from 2 gaussians
    cl_data = np.vstack([2.5*np.random.randn(70,2)+7, np.random.randn(30,2)+1])
    cl_yhat = clustering_method(cl_data)

    # plot cluster data
    plt.scatter(cl_data[:,0], cl_data[:,1])

    # generate and visualize consensus matrix, using KMeans with 3 clusters in the inner loop
    def clustering_method(data):
        cl = KMeans(3)
        return cl.fit_predict(data)
    cl_CCM = consensus_matrix(cl_data, clustering_method, n_runs=10)
    sns.clustermap(cl_CCM)
    # as expected, the consensus matrix is not clean. Try it with 2 clusters in the inner loop.
    """
    # precalculate random seeds for daughter jobs
    seeds = [np.random.randint(np.iinfo(np.int32).max) for i in range(n_runs)]

    # run n_runs runs
    p = Parallel(n_jobs=n_jobs, backend=parallel_backend)
    yhats = p(delayed(_subsample_cluster)(clustering_method, data, col_sample, row_sample, random_seed=s) for s in seeds)
    
    # count co-clustering
    CCM = yhats2ccm(yhats)
    return CCM

def yhats2ccm(yhats):
    """
    Given a list of cluster assignment vectors, of the form
    yhats = [[1,0,0], [1,1,0], ...]
    where each element of yhats is a vector of length N with the ith element
    reperesenting the ith sample's cluster assignment,
    computes co-clustering matrix CCM
    where CCM[i,j] is the proportion of times the ith sample and the jth sample
    are assigned the same cluster, when they both have cluster assignments.

    Accepts yhat vectors where not all samples are assigned, with np.nan marking
    a missing assignment.
    """
    norm_factor = np.zeros((len(yhats[0]), len(yhats[0])))
    CCM = np.zeros((len(yhats[0]), len(yhats[0])))
    for yhat in yhats:
        for i in range(len(yhat)):
            if np.isnan(yhat[i]):
                continue
            for j in range(i, len(yhat)):
                if np.isnan(yhat[j]):
                    continue
                norm_factor[i,j] += 1
                norm_factor[j,i] += 1
                if yhat[i] == yhat[j]:
                    CCM[i,j] += 1
                    CCM[j,i] += 1
            
    return CCM/norm_factor


def consensus_index_cdf(CCM):
    return ECDF(CCM[np.tril_indices_from(CCM, -1)])

def _set_kwargs_for_plot(kwargs):
    if "ax" not in kwargs:
        kwargs['ax'] = plt.figure().add_subplot(111)
    kwargs['color'] = kwargs.get('c') or kwargs.get('color') or next(kwargs['ax']._get_lines.prop_cycler)['color']

def pac_metric(ccm, lower=.1, upper=.9):
    """
    Computes Percentage Ambiguous Clustering from a consensus-clustering generated
    co-clustering matrix `ccm`.
    The PAC is the volume of the consensus index distribution which lies between `lower` and `upper`
    (default .1 and .9), symbolizing samples which ambiguously belong to the same cluster.
    Sample pairs with a consensus index of <.1 or >.9 unambiguously don't, or do cluster together.
    """
    ci = consensus_index_cdf(ccm)
    return ci(upper)-ci(lower)

def pac_plot(ci, label='co-clustering index', x=np.arange(0,1.01,.01), ci_top=.9, ci_bottom=.1, **kwargs):
    """
    Make a PAC plot (ref..)

    Example:
    --------

    """
    _set_kwargs_for_plot(kwargs)
    ax = kwargs.pop('ax')
    pac = ci(ci_top) - ci(ci_bottom)
    label = "{}, PAC={:.3f}".format(label, pac)
    ax.plot(x, ci(x), label=label, **kwargs)
    ax.plot([ci_bottom, ci_bottom], [0, 1], 'k--')
    ax.plot([ci_top, ci_top], [0, 1], 'k--')
    plt.xlim(0,1)
    plt.ylim(0,1)
    ax.legend(loc='best')
    return ax
