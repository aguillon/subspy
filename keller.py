import numpy as np
#import misc
from config import Verbosity, verbose_function, logging

import cluster


@verbose_function
def weighted_fcm(x, n_clusters, max_iter=300, tol=1e-4, centers=None,
        memberships=None, weights=None,  **kwargs):
    n, d = x.shape
    centers = cluster._init_centers(centers, x, n_clusters, **kwargs)
    memberships = cluster._init_memberships(memberships, centers, x, n_clusters, **kwargs)
    weights = _init_weights(weights, x, n_clusters)
    for i in range(max_iter):
        new_memberships = cluster._update_memberships(x, n_clusters, weights, centers, **kwargs)
        new_centers = cluster._update_centers(x, n_clusters, weights, new_memberships, **kwargs)
        new_weights = _new_update_weights(x, n_clusters, new_memberships,
                new_centers, **kwargs)
        if np.linalg.norm(new_centers - centers) < tol:
            break
        memberships = new_memberships
        centers = new_centers
        weights = new_weights
    return (memberships, centers, weights)

def _new_update_weights(x, n_clusters, memberships, centers, v = 2., m = 2.,
        **kwargs):
    n,d = x.shape
    p = 1. / (1. - v)
    m_memberships = memberships ** m
    T = (x[:,None] - centers) ** 2
    S = np.einsum("ij,jil->il",m_memberships, T) ** p
    weights = S / (np.sum(S,axis = 1)).reshape((-1,1))
    return weights

def _update_weights(x, n_clusters, memberships, centers, v = 2., m = 2.,
        **kwargs):
    n,d = x.shape
    weights = np.zeros((n_clusters, d))
    consts = [0 for _ in range(d)]
    expo = 1. / (1. - v)
    for r in range(n_clusters):
        tot = 0.0
        for j in range(d):
            consts[j] = sum(memberships[r,i] ** m * (x[i,j] - centers[r,j])**2 for i in range(n))
            tot += consts[j] ** expo
        for j in range(d):
            weights[r,j] = consts[j] ** expo / tot
    return weights


class Keller(cluster.FCMeans):
    def __init__(self, n_clusters, max_iter = 300, tol = 1e-4, verbose =
            Verbosity.NONE, centers = None, memberships = None, weights = None,
            m = 2., v = 2.):
        cluster.FCMeans.__init__(self, n_clusters, max_iter, tol, verbose,
                centers, memberships, weights, m = m, v = v)

    def fit(self, X, y=None):
        self._costs = []     # store inertia values
        self.X = X
        self.memberships, self.centers, self.weights = self._alternate_descent()
        self._inertia = self._compute_inertia(self.X, self.centers, self.memberships,
                self.weights)

    def _update_weights(self, X, n_clusters, memberships, centers):
        return _new_update_weights(X, n_clusters, memberships, centers, verbose = self.verbose)

    def _alternate_descent(self):
        n, d = self.X.shape
        centers = cluster._init_centers(self.centers, self.X, self.n_clusters)
        memberships = cluster._init_memberships(self.memberships, centers, self.X, self.n_clusters)
        weights = cluster._init_weights(self.weights, self.X, self.n_clusters)
        for i in range(self.max_iter):
            if self.verbose & Verbosity.COST_FUNCTION:
                self._log_cost(centers, memberships, weights)
            new_memberships = self._update_memberships(self.X, self.n_clusters, weights, centers)
            new_centers = self._update_centers(self.X, self.n_clusters, weights, new_memberships)
            new_weights = self._update_weights(self.X, self.n_clusters, new_memberships, new_centers)
            if np.linalg.norm(new_centers - centers) < self.tol and i > 1:
                break
            memberships = new_memberships
            centers = new_centers
            weights = new_weights
        if self.verbose & Verbosity.COST_FUNCTION:
            self._log_cost(centers, memberships, weights)
        return memberships, centers, weights


