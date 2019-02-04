import numpy as np
from scipy.spatial.distance import squareform, pdist

def max_min_method(X, n_clusters):
    dists = squareform(pdist(X))
    furthest_ixs = np.unravel_index(np.argmax(dists), dists.shape)
    res = [X[i] for i in furthest_ixs]
    while len(res) < n_clusters:
        ix = np.argmax((np.linalg.norm(X[:, np.newaxis] - np.array(res), axis=2)).min(axis=1))
        res.append(X[ix])
    return np.array(res)


