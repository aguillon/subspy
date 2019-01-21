import numpy as np
import cluster
from proximal_subspace import _pfscm_prox_gradient, _prox_abs, Prosecco, _sparsity_prox_op
import proximal_possibilistic
from config import Verbosity


def _prox_op(u, l):
    c,n = u.shape
    z = np.zeros((c,n))
    for i in range(n):
        som = sum(u[:,i])
        for r in range(c):
            z[r,i] = u[r,i] + 1./c * (1 + _prox_abs(som - 1, c*l) - som)
    return z


class SquarePenaltyPCM(cluster.FCMeans):
    def __init__(self, n_clusters, memberships_gamma, sparsity_gamma,
            max_iter=300, tol=1e-4, verbose=Verbosity.NONE, centers=None,
            memberships=None, weights=None):
        cluster.FCMeans.__init__(self, n_clusters, max_iter=max_iter, tol=tol,
                verbose=verbose, centers=centers, memberships=memberships,
                weights=weights, m=2., v=2.)
        self.memberships_gamma = memberships_gamma
        self.sparsity_gamma = sparsity_gamma

    def _memberships_gradient(self, memberships, x, centroids):
        c,n = memberships.shape
        n,d = x.shape
        T = ((x - centroids[:,None]) ** 2).sum(axis=2)
        grad = self.m * memberships ** (self.m - 1) * T + 2 * self.sparsity_gamma * memberships
        return grad

    def _alternate_descent(self):
        n, d = self.X.shape
        self.centers = cluster._init_centers(self.centers, self.X, self.n_clusters)
        self.memberships = cluster._init_memberships(self.memberships, self.centers, self.X, self.n_clusters)
        self.weights = cluster._init_weights(self.weights, self.X, self.n_clusters)
        self.memberships, self.centers = super()._alternate_descent()
        for niter in range(self.max_iter):
            if self.verbose & Verbosity.COST_FUNCTION:
                self._log_cost(self.centers, self.memberships, self.weights)
            cst = np.min(1/self._memberships_hessian(self.memberships,
                    self.centers, n, d, self.n_clusters))
            new_memberships = _pfscm_prox_gradient(self.memberships, self._memberships_gradient,
                    _prox_op, cst, prox_arg = cst * self.memberships_gamma,
                    max_iter = 10, tol = self.tol,
                    grad_args = (self.X, self.centers))
            new_centers = self._update_centers(self.X, self.n_clusters,
                    self.weights, new_memberships)
            gain = np.linalg.norm(new_centers - self.centers)
            gain += np.linalg.norm(new_memberships - self.memberships)
            if gain < self.tol:
                break
            self.centers = new_centers
            self.memberships = new_memberships
        return self.memberships, self.centers


    def _memberships_hessian(self, new_memberships, new_centers, n, d, n_clusters):
        T = ((self.X - new_centers[:,None]) ** 2).sum(axis=2)
        return self.m * (self.m - 1) * T + 2 * self.sparsity_gamma


def _sparse_cost_function(u_zero, new_u, l):
    (c,) = u_zero.shape
    t = np.count_nonzero(new_u)
    return l*t + np.abs(np.sum(new_u) - 1) + 1/2 * np.linalg.norm(u_zero - new_u)**2

class SparsePCM(proximal_possibilistic.PossibilisticCMeans):
    def __init__(self, n_clusters, memberships_gamma, sparsity_gamma,
            max_iter=300, tol=1e-4, verbose=Verbosity.NONE, centers=None,
            memberships=None, weights=None):
        super().__init__(n_clusters, memberships_gamma, max_iter=max_iter, tol=tol,
                verbose=verbose, centers=centers, memberships=memberships,
                weights=weights)
        self.sparsity_gamma = sparsity_gamma

    def _sparse_prox_op(self, u, l):
        print(u)
        assert np.all(u >= 0)
        c,n = u.shape
        z = proximal_possibilistic._prox_op(u, l)
        for i in range(n):
            best_vect = z[:,i]
            best_cost = _sparse_cost_function(z[:,i], best_vect,self.sparsity_gamma)
            for k in range(c):
                ixs = np.argpartition(best_vect, c-k-1)[:c-k-1]
                ui = z[:,i].copy()    # FIXME il manque un morceau de renormalisation
                ui[ixs] = 0
                cost = _sparse_cost_function(z[:,i], ui, self.sparsity_gamma)
                if cost <= best_cost:
                    best_cost = cost
                    best_vect = ui.copy()
            z[:,i] = best_vect
        return z

    def _alternate_descent(self):
        n, d = self.X.shape
        self.centers = cluster._init_centers(self.centers, self.X, self.n_clusters)
        self.memberships = cluster._init_memberships(self.memberships, self.centers, self.X, self.n_clusters)
        self.weights = cluster._init_weights(self.weights, self.X, self.n_clusters)
        self.memberships, self.centers = cluster.FCMeans._alternate_descent(self)
        for niter in range(self.max_iter):
            if self.verbose & Verbosity.COST_FUNCTION:
                self._log_cost(self.centers, self.memberships, self.weights)
            cst = 1/2*np.min(1/self._memberships_hessian(self.memberships,
                    self.centers))
            #print("Cst is {}".format(cst))
            #print("Cst is now {}".format(cst))
            new_memberships = _pfscm_prox_gradient(self.memberships, self._memberships_gradient,
                    self._sparse_prox_op, cst, prox_arg = cst * self.memberships_gamma,
                    max_iter = 10, tol = self.tol,
                    grad_args = (self.X, self.centers))
            new_centers = self._update_centers(self.X, self.n_clusters,
                    self.weights, new_memberships)
            gain = np.linalg.norm(new_centers - self.centers)
            if gain < self.tol:
                break
            self.centers = new_centers
            self.memberships = new_memberships
        return self.memberships, self.centers



