"""
Module with clustering-related functions

@jona
"""
import numpy as np
from joblib import Parallel, delayed

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
    norm_factor = np.zeros((data.shape[0], data.shape[0]))
    CCM = np.zeros((data.shape[0], data.shape[0]))
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
    pass