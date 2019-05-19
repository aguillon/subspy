import numpy as np
import cluster
from config import Verbosity
from scipy.optimize import LinearConstraint, Bounds, minimize

class TSKMeans(cluster.FCMeans):
    def __init__(self, n_clusters, alpha, max_iter = 300, tol = 1e-4, verbose =
            Verbosity.NONE, centers = None, memberships = None, weights = None):
        super().__init__(n_clusters, max_iter, tol, verbose, centers, memberships, weights)
        self.alpha = alpha

    def _update_memberships(self, X, weights, centers):
        T = (weights * (X[:,None] - centers) ** 2).sum(axis=2)
        u = np.zeros((self.n_clusters, X.shape[0]))
        u[T.argmin(axis=1),np.arange(len(X))] = 1
        return u

    def _update_centers(self, X, weights, memberships):
        return super()._update_centers(self.X, self.n_clusters, weights,
                memberships)

    def _update_weights(self, X, memberships, centers, old_weights):
        n_clusters,d = old_weights.shape
        w = np.zeros((n_clusters,d))
        bounds = Bounds(np.zeros(d), np.ones(d))
        ones = np.array(1)
        constraint = LinearConstraint(np.ones(d).T, ones, ones)
        for p in range(n_clusters):
            res = minimize(lambda weights: self._compute_inertia(X, centers,
                memberships, weights.reshape((-1,1)).T),
                old_weights[p,:].T,
                method="SLSQP",
                bounds=bounds,
                constraints=[constraint])
            w[p,:] = res.x
            print(res.x)
        return w

    def _alternate_descent(self):
        n, d = self.X.shape
        self._init_weights()
        centers = cluster._init_centers(self.centers, self.X, self.n_clusters)
        memberships = cluster._init_memberships(self.memberships, centers, self.X, self.n_clusters)
        for i in range(self.max_iter):
            if self.verbose & Verbosity.COST_FUNCTION:
                self._log_cost(centers, memberships, self.weights)
            new_memberships = self._update_memberships(self.X, self.weights, centers)
            new_centers = self._update_centers(self.X, self.weights, new_memberships)
#            print(new_memberships)
#            print("centers")
#            print(new_centers)
#            print("weights")
#            print(self.weights)
            print(self._compute_inertia(self.X, centers, memberships,
                self.weights, _mode="debug"))
            new_weights = self._update_weights(self.X, new_memberships,
                    new_centers, self.weights)
            print(self._compute_inertia(self.X, centers, memberships,
                new_weights, _mode="debug"))
            if np.linalg.norm(new_centers - centers) < self.tol:
                break
            memberships = new_memberships
            centers = new_centers
            self.weights = new_weights
        if self.verbose & Verbosity.COST_FUNCTION:
            self._log_cost(centers, memberships, self.weights)
        return memberships, centers

    def _compute_inertia(self, X, centers, memberships, weights, _mode="normal"):
        T = (weights * (X[:,None] - centers) ** 2).sum(axis=2)
        first_term = (memberships * T.T).sum()
        w_sum = ((weights[:,1:] - weights[:,:-1])**2).sum()
        second_term = 1/2*self.alpha*w_sum
        if _mode == "debug":
            print(weights[0])
            print("first_term: ", first_term)
            print("second term: ", second_term)
        return first_term + second_term



