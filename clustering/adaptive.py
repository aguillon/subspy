import numpy as np
import cluster
import clustering.possibilistic
import clustering.initialization as initialization
from config import Verbosity


def _init_eta(X, centers, beta):
    diss = np.linalg.norm(centers[:, np.newaxis] - centers, axis=2)
    np.fill_diagonal(diss, np.inf)
    dmin = diss.min(axis=1)
    return dmin/(2 * (-np.log(beta)))

class AdaptivePCM(clustering.possibilistic.PCMeans):
    def __init__(self, n_clusters, beta,
            max_iter=300, tol=1e-4, verbose=Verbosity.NONE, centers=None,
            memberships=None, weights=None):
        super().__init__(n_clusters, eta_parameter = beta, tol=tol,
                verbose=verbose, centers=centers, memberships=memberships,
                weights=weights)
        self.beta = beta

    def _update_clusters(self, X, memberships, centers, weights):
        n_clusters,n = memberships.shape
        _,d = X.shape
        labels = np.sort(np.unique(memberships.argmax(axis=0)))
        new_n_clusters = len(labels)
        new_memberships = np.zeros((new_n_clusters, n))
        new_centers = np.zeros((new_n_clusters, d))
        new_weights = np.zeros((new_n_clusters, d))
        for new_r,old_r in enumerate(labels):
            new_memberships[new_r] = memberships[old_r]
            new_centers[new_r] = centers[old_r]
            new_weights[new_r] = weights[old_r]
        return new_n_clusters, new_memberships, new_centers, new_weights

    def _update_eta(self, X, memberships, centers):
        c,_ = memberships.shape
        max_u = memberships.argmax(axis=0)
        etas = np.zeros((c,))
        for r in range(c):
            X_r = X[max_u == r]
            n_r, _ = X_r.shape
            center = X_r.sum(axis=0)/n_r
            s = ((X_r[:,None] - center)**2).sum()
            etas[r] = 1/n_r * s
        return etas

    def _alternate_descent(self):
        n, d = self.X.shape
        self._init_weights()
        n_clusters = self.n_clusters
        centers = initialization.max_min_method(self.X, n_clusters)
        memberships = cluster._init_memberships(self.memberships, centers, self.X, n_clusters)
        eta = _init_eta(self.X, centers, self.beta)
        weights = self.weights
        for i in range(self.max_iter):
            if self.verbose & Verbosity.COST_FUNCTION:
                self._log_cost(centers, memberships, weights)
            new_memberships = self._update_memberships(self.X, n_clusters, weights, centers, eta)
            new_centers = self._update_centers(self.X, n_clusters, weights, new_memberships)
            if np.linalg.norm(new_centers - centers) < self.tol:
                break
            n_clusters, new_memberships, new_centers, new_weights = self._update_clusters(self.X, new_memberships, new_centers, weights)
            eta = self._update_eta(self.X, new_memberships, new_centers)
            memberships = new_memberships
            centers = new_centers
            weights = new_weights
        if self.verbose & Verbosity.COST_FUNCTION:
            self._log_cost(centers, memberships, self.weights)
        self.eta = eta
        return memberships, centers


