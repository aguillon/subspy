import numpy as np
import cluster
import factorization.nmf as nmf
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
from itertools import combinations

from scipy.optimize import LinearConstraint, Bounds, minimize


def pca_hull_initialization(X):
    """Initialization algorithm from Bauckhage & Thurau (2009)"""
    pca = PCA(n_components=0.95) # must explain 95% of the variance
    X -= X.mean()
    Xpca = pca.fit_transform(X)
    indices = set()
    for (i,j) in combinations(range(pca.n_components_),2):
        Xproj = Xpca[:,(i,j)]
        hull = ConvexHull(Xproj)
        indices.update(hull.vertices)
    return np.array(sorted(indices))

def _least_squares_cost(X, centers, memberships):
    return np.linalg.norm(X - memberships.dot(centers))**2

# TODO: - implement accelerated version and other initialization methods
#       - generalize projected gradient descent to enforce summation constraint?
class FactorizedFCM(nmf.NMF):
    def __init__(self, n_components, max_iter=300, tol=1e-4):
        super().__init__(n_components, descent_method="cd", max_iter=max_iter,tol=tol)


    def _init_H(self):
        all_indices = pca_hull_initialization(self.X)
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
        print(H)
        W2 = np.zeros_like(W)
        for i in range(W.shape[0]):
            bounds = Bounds(np.zeros_like(W[i]), np.ones_like(W[i]))
            ones = np.array(1)
            constraint = LinearConstraint(np.ones(self.W.shape[1]).T, ones, ones)
            res = minimize(lambda memberships: _least_squares_cost(self.X[i], H, memberships),
                W[i].T,
                method="trust-constr",
                bounds=bounds,
                constraints=[constraint])
            print(res.x)
            W2[i] = res.x
        return W2

    def get_memberships(self):
        return self.W.T

    def get_centers(self):
        return self.H

