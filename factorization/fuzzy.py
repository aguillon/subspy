import numpy as np
import cluster
import factorization.nmf as nmf
import factorization.commons as commons

from scipy.optimize import LinearConstraint, Bounds, minimize


# TODO: - implement accelerated version and other initialization methods
#       - generalize projected gradient descent to enforce summation constraint?
class FactorizedFCM(nmf.NMF):
    def __init__(self, n_components, max_iter=300, tol=1e-4):
        super().__init__(n_components, descent_method="cd", max_iter=max_iter,tol=tol)

    def _init_H(self):
        all_indices = commons.pca_hull_initialization(self.X)
        print(all_indices)
        indices = np.random.choice(all_indices, self.n_components, replace=False)
        self.H = self.X[indices]
        print(indices)
        print(self.H)
        return self.H

# REMOVE ME: doesn't seem to work?  Test again when constraint on W is enforced
#    def _init_H(self):
#        if self.W is None:
#            self._init_W()
#        self.H = cluster._init_centers(self.H, self.X, self.n_components)
#        return self.H

    def _init_W(self):
        if self.W is None:
            memberships = cluster._init_memberships(None, self.H, self.X,
                    self.n_components)
            self.W = memberships.T
        return self.W

    def _update_H(self, W, H):
        return cluster._update_centers(self.X, self.n_components, None, W.T, m=1)

    def _update_W(self, W, H):
        n,n_clusters = W.shape
        W3 = np.zeros_like(W)
        for i in range(n):
            bounds = Bounds(np.zeros(n_clusters), np.ones(n_clusters))
            ones = np.array(1)
            constraint = LinearConstraint(np.ones(n_clusters).T, ones, ones)
            res = minimize(lambda memberships: commons._least_squares_cost(self.X[i], H, memberships),
                W[i].T,
                method="SLSQP",
                bounds=bounds,
                constraints=[constraint])
            W3[i] = res.x
        return W3

    def get_memberships(self):
        return self.W.T

    def get_centers(self):
        return self.H

