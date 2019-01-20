import numpy as np
import cluster
import keller
from proximal_subspace import _pfscm_prox_gradient, _prox_abs, Prosecco, _sparsity_prox_op
from config import Verbosity
from entropy import EWKM
from borgelt import Borgelt

def _prox_op(u, l):
    c,n = u.shape
    z = np.zeros((c,n))
    for i in range(n):
        som = sum(u[:,i])
        for r in range(c):
            z[r,i] = u[r,i] + 1./c * (1 + _prox_abs(som - 1, c*l) - som)
    return z


# TODO change the name
class PossibilisticCMeans(cluster.FCMeans):
    def __init__(self, n_clusters, memberships_gamma, max_iter = 300, tol = 1e-4, verbose =
            Verbosity.NONE, centers = None, memberships = None, weights = None):
        cluster.FCMeans.__init__(self, n_clusters, max_iter = max_iter, tol = tol, verbose = verbose,
                centers = centers, memberships = memberships, weights = weights, m = 2., v = 2.)
        self.memberships_gamma = memberships_gamma

    def _memberships_gradient(self, memberships, x, centroids):
        c,n = memberships.shape
        n,d = x.shape
        T = ((x - centroids[:,None]) ** 2).sum(axis=2)
        memberships_grad = self.m * memberships ** (self.m - 1) * T
        return memberships_grad

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
                    self.centers))
            print("Cst is {}".format(cst))
            #print("Cst is now {}".format(cst))
            new_memberships = _pfscm_prox_gradient(self.memberships, self._memberships_gradient,
                    _prox_op, cst, prox_arg = cst * self.memberships_gamma,
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


    def _memberships_hessian(self, new_memberships, new_centers):
        # TODO: tensor T is computed here and in gradient descent
        T = ((self.X - new_centers[:,None]) ** 2).sum(axis=2)
        return self.m * (self.m - 1) * T

class PossibilisticSubspace(keller.Keller):
    def __init__(self, n_clusters, memberships_gamma, max_iter = 300, tol = 1e-4, verbose =
            Verbosity.NONE, centers = None, memberships = None, weights = None):
        keller.Keller.__init__(self, n_clusters, max_iter = max_iter, tol = tol, verbose = verbose,
                centers = centers, memberships = memberships, weights = weights, m = 2., v = 2.)
        self.memberships_gamma = memberships_gamma

    def _memberships_gradient(self, memberships, x, centers, weights):
        v_weights = weights ** self.v
        T = ((v_weights * (x[:,None] - centers) ** 2).sum(axis=2)).T
        gradient = self.m * memberships ** (self.m-1) * T
        return gradient

    def _alternate_descent(self):
        n, d = self.X.shape
        self.centers = cluster._init_centers(self.centers, self.X, self.n_clusters)
        self.memberships = cluster._init_memberships(self.memberships, self.centers, self.X, self.n_clusters)
        self.weights = cluster._init_weights(self.weights, self.X, self.n_clusters)
        self.memberships, self.centers = cluster.FCMeans._alternate_descent(self)
        for niter in range(self.max_iter):
            if self.verbose & Verbosity.COST_FUNCTION:
                self._log_cost(self.centers, self.memberships, self.weights)
            cst = np.min(1/self._memberships_hessian(self.memberships,
                    self.centers, self.weights))
            new_memberships = _pfscm_prox_gradient(self.memberships, self._memberships_gradient,
                    _prox_op, cst, prox_arg = cst * self.memberships_gamma,
                    max_iter = 10, tol = self.tol,
                    grad_args = (self.X, self.centers, self.weights))
            new_centers = self._update_centers(self.X, self.n_clusters,
                    self.weights, new_memberships)
            new_weights = self._update_weights(self.X, self.n_clusters, new_memberships,
                    new_centers)
            gain = np.linalg.norm(new_centers - self.centers)
            if gain < self.tol:
                break
            self.centers = new_centers
            self.memberships = new_memberships
            self.weights = new_weights
        return self.memberships, self.centers, self.weights

    def _memberships_hessian(self, _memberships, centers, weights):
        v_weights = weights ** self.v
        T = ((v_weights * (self.X[:,None] - centers) ** 2).sum(axis=2)).T
        compact_hessian = self.m * T
        return compact_hessian

class PossibilisticProsecco(Prosecco, PossibilisticSubspace):
    def __init__(self, n_clusters, memberships_gamma, weights_gamma, max_iter = 300, tol = 1e-4, verbose =
            Verbosity.NONE, centers = None, memberships = None, weights = None):
        Prosecco.__init__(self, n_clusters, weights_gamma, max_iter, tol, verbose,
                centers, memberships, weights)
        PossibilisticSubspace.__init__(self, n_clusters, memberships_gamma, max_iter, tol, verbose,
                centers, memberships, weights)

    def _alternate_descent(self):
        n, d = self.X.shape
        self.centers = cluster._init_centers(self.centers, self.X, self.n_clusters)
        self.memberships = cluster._init_memberships(self.memberships, self.centers, self.X, self.n_clusters)
        self.weights = cluster._init_weights(self.weights, self.X, self.n_clusters)
        self.memberships, self.centers = cluster.FCMeans._alternate_descent(self)
        for niter in range(self.max_iter):
            if self.verbose & Verbosity.COST_FUNCTION:
                self._log_cost(self.centers, self.memberships, self.weights)
            memberships_cst = np.min(1/self._memberships_hessian(self.memberships,
                self.centers, self.weights))
            new_memberships = _pfscm_prox_gradient(self.memberships, self._memberships_gradient,
                    _prox_op, memberships_cst, prox_arg = memberships_cst * self.memberships_gamma,
                    max_iter = 10, tol = self.tol,
                    grad_args = (self.X, self.centers, self.weights))
            #print(np.sum(new_memberships))
            new_centers = self._update_centers(self.X, self.n_clusters,
                    self.weights, new_memberships)
            weights_cst = np.min(1/self._weights_hessian(new_memberships,
                    new_centers, n, d, self.n_clusters))
            new_weights = _pfscm_prox_gradient(self.weights, self._weights_gradient,
                    _sparsity_prox_op, weights_cst, prox_arg = weights_cst * self.weights_gamma,
                    max_iter = 10, tol = self.tol, grad_args = (self.X, new_centers, new_memberships))
            gain = np.linalg.norm(new_centers - self.centers)
            #print(self.centers)
            #print(gain)
            if gain < self.tol:
                break
            self.weights = new_weights
            self.centers = new_centers
            self.memberships = new_memberships
        return self.memberships, self.centers, self.weights


class PossibilisticEWKM(EWKM, PossibilisticSubspace):
    def __init__(self, n_clusters, memberships_gamma, weights_gamma, max_iter = 300, tol = 1e-4, verbose =
            Verbosity.NONE, centers = None, memberships = None, weights = None):
        EWKM.__init__(self, n_clusters, weights_gamma, max_iter, tol, verbose,
                centers, memberships, weights)
        PossibilisticSubspace.__init__(self, n_clusters, memberships_gamma, max_iter, tol, verbose,
                centers, memberships, weights)

    def _alternate_descent(self):
        n, d = self.X.shape
        self.centers = cluster._init_centers(self.centers, self.X, self.n_clusters)
        self.memberships = cluster._init_memberships(self.memberships, self.centers, self.X, self.n_clusters)
        self.weights = cluster._init_weights(self.weights, self.X, self.n_clusters)
        self.memberships, self.centers = cluster.FCMeans._alternate_descent(self)
        for niter in range(self.max_iter):
            if self.verbose & Verbosity.COST_FUNCTION:
                self._log_cost(self.centers, self.memberships, self.weights)
            memberships_cst = np.min(1/self._memberships_hessian(self.memberships,
                self.centers, self.weights))
            new_memberships = _pfscm_prox_gradient(self.memberships, self._memberships_gradient,
                    _prox_op, memberships_cst, prox_arg = memberships_cst * self.memberships_gamma,
                    max_iter = 10, tol = self.tol,
                    grad_args = (self.X, self.centers, self.weights))
            #print(np.sum(new_memberships))
            new_centers = self._update_centers(self.X, self.n_clusters,
                    self.weights, new_memberships)
            new_weights = self._update_weights(self.X, self.n_clusters, new_memberships, new_centers)
            gain = np.linalg.norm(new_centers - self.centers)
            #print(self.centers)
            #print(gain)
            if gain < self.tol:
                break
            self.weights = new_weights
            self.centers = new_centers
            self.memberships = new_memberships
        return self.memberships, self.centers, self.weights



class PossibilisticBorgelt(Borgelt, PossibilisticSubspace):
    def __init__(self, n_clusters, memberships_gamma, beta, max_iter = 300, tol = 1e-4, verbose =
            Verbosity.NONE, centers = None, memberships = None, weights = None):
        Borgelt.__init__(self, n_clusters, beta, max_iter, tol, verbose,
                centers, memberships, weights)
        PossibilisticSubspace.__init__(self, n_clusters, memberships_gamma, max_iter, tol, verbose,
                centers, memberships, weights)

    def _alternate_descent(self):
        n, d = self.X.shape
        self.centers = cluster._init_centers(self.centers, self.X, self.n_clusters)
        self.memberships = cluster._init_memberships(self.memberships, self.centers, self.X, self.n_clusters)
        self.weights = cluster._init_weights(self.weights, self.X, self.n_clusters)
        self.memberships, self.centers = cluster.FCMeans._alternate_descent(self)
        for niter in range(self.max_iter):
            if self.verbose & Verbosity.COST_FUNCTION:
                self._log_cost(self.centers, self.memberships, self.weights)
            memberships_cst = np.min(1/self._memberships_hessian(self.memberships,
                self.centers, self.weights))
            new_memberships = _pfscm_prox_gradient(self.memberships, self._memberships_gradient,
                    _prox_op, memberships_cst, prox_arg = memberships_cst * self.memberships_gamma,
                    max_iter = 10, tol = self.tol,
                    grad_args = (self.X, self.centers, self.weights))
            #print(np.sum(new_memberships))
            new_centers = self._update_centers(self.X, self.n_clusters,
                    self.weights, new_memberships)
            new_weights = self._update_weights(self.X, self.n_clusters, new_memberships, new_centers)
            gain = np.linalg.norm(new_centers - self.centers)
            #print(self.centers)
            #print(gain)
            if gain < self.tol:
                break
            self.weights = new_weights
            self.centers = new_centers
            self.memberships = new_memberships
        return self.memberships, self.centers, self.weights

