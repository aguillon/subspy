import numpy as np
import scipy.sparse as sp
import random
import misc
import cluster
from config import Verbosity

def entropy_algorithm(x, n_clusters, max_iter=300, tol=1e-4, verbose=False, centers=None, memberships=None, gamma = 0.5):
    n, d = x.shape
    weights = np.zeros((n_clusters, d))
    for j in range(d):
        for r in range(n_clusters):
            weights[r,j] = 1
    if sp.issparse(x):
        weights = sp.csc_matrix(weights)
    centers = clustering._init_centers(centers, x, n_clusters)
    if memberships is None:
        memberships = np.zeros((n_clusters, n))
        for i in range(n):
            r = random.randint(0, n_clusters - 1)
            memberships[r,i] = 1
    for i in range(max_iter):
        # update_memberships squares the weights
        weights = np.sqrt(weights)
        new_memberships = clustering._update_memberships(x, n_clusters, weights, centers)
        new_centers = clustering._update_centers(x, n_clusters, weights, new_memberships)
        new_weights = _update_weights(x, n_clusters, new_memberships, new_centers, gamma = gamma)
        if np.linalg.norm(new_centers - centers) + np.linalg.norm(new_weights - weights) < tol:
            break
        memberships = new_memberships
        centers = new_centers
        weights = new_weights
    return memberships, centers, weights

def _update_weights(x, n_clusters, memberships, centers, gamma = 0.5, m = 2.):
    n,d = x.shape
    weights = np.zeros((n_clusters, d))
    neg_dispersions = np.zeros((d,))
    for r in range(n_clusters):
        # argument of the exponentials
        neg_dispersions = -np.sum(memberships[r,i] ** m *
                np.asarray((x[i] - centers[r])).reshape((-1))**2
                for i in range(n))/gamma
        # in the paper: tot += np.exp(-dispersions[j]/gamma)
        # log sum trick
        smallest_disp = max(neg_dispersions) # smallest dispersion
        log_total = np.log(np.sum(np.exp(neg_dispersions - smallest_disp)))
        log_weight = neg_dispersions - smallest_disp - log_total
        weights[r] = np.exp(log_weight)
    return weights

class EWKM(cluster.FCMeans):
    def __init__(self, n_clusters, weights_gamma, max_iter = 300, tol = 1e-4,
            verbose = Verbosity.NONE, centers = None, memberships = None,
            weights = None, m = 2., v = 2.):
        cluster.FCMeans.__init__(self, n_clusters, max_iter, tol, verbose,
                centers, memberships, weights, m = m, v = v)
        self.weights_gamma = weights_gamma

    def fit(self, X, y=None):
        self._costs = []     # store inertia values
        self.X = X
        self.memberships, self.centers, self.weights = self._alternate_descent()
        self._inertia = self._compute_inertia(self.X, self.centers, self.memberships,
                self.weights)

    def _update_weights(self, X, n_clusters, memberships, centers):
        return _update_weights(X, n_clusters, memberships, centers,
                gamma = self.weights_gamma)

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
            if np.linalg.norm(new_centers - centers) < self.tol:
                break
            memberships = new_memberships
            centers = new_centers
            weights = new_weights
        if self.verbose & Verbosity.COST_FUNCTION:
            self._log_cost(centers, memberships, weights)
        return memberships, centers, weights


