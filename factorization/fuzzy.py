import numpy as np
import cluster
import factorization.nmf as nmf


class FactorizedFCM(nmf.NMF):
    def __init__(self, n_components, max_iter=300, tol=1e-4):
        super().__init__(n_components, method="cd", max_iter=max_iter,tol=tol)

    def _update_H(self, W, H):
        return cluster._update_centers(self.X, self.n_components, None, W.T, m=1)

    def get_memberships(self):
        return self.W.T

    def get_centers(self):
        return self.H

