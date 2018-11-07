import numpy as np
import random
import misc
import cluster
import keller       # initialization with keller algorithm
from config import Verbosity

def borgelt_algorithm(x, n_clusters, max_iter=300, tol=1e-4, verbose=False, centers=None, memberships=None, weights=None, beta=0.5):
    n, d = x.shape
    if weights is None:
        weights = np.zeros((n_clusters, d))
        for j in range(d):
            for r in range(n_clusters):
                weights[r,j] = 1./d
    if centers is None:
        mi = np.array([min(p[i] for p in x) for i in range(d)])
        ma = np.array([max(p[i] for p in x) for i in range(d)])
        centers = np.array([np.random.uniform(mi, ma, size = d) for _ in range(n_clusters)])
    if memberships is None:
        memberships = np.zeros((n_clusters, n))
        for i in range(n):
            r = random.randint(0, n_clusters - 1)
            memberships[r,i] = 1
    for i in range(max_iter):
        new_memberships = clustering._update_memberships(x, n_clusters, weights, centers)
        new_centers = clustering._update_centers(x, n_clusters, weights, new_memberships)
        new_weights = _borgelt_weights(x, n_clusters, new_memberships, new_centers, weights, beta = beta)
        if np.linalg.norm(new_centers - centers) + np.linalg.norm(new_weights - weights) < tol:
            break
        memberships = new_memberships
        centers = new_centers
        weights = new_weights
    return memberships, centers, weights

def _update_weights(x, n_clusters, memberships, centers, beta = 0.5, m = 2.):
    n,d = x.shape
    # we consider axes-parallel subspaces
    # so we need only diagonal values
    diagonal = np.zeros((n_clusters, d))
    consts = np.zeros((d,))
    for r in range(n_clusters):
        tot = 0.0
        for j in range(d):
            # computation of the constants
            # actually consts[j]⁻²
            consts[j] = 1/sum(memberships[r,i] ** m * (x[i,j] - centers[r,j])**2 for i in range(n))
            # if old_weights[r,j] > 0:
            tot += consts[j]
        try:
            assert tot > 0
        except:
            raise
        # sorting and inversing the constants
        consts2 = sorted(consts, reverse=True)
        # computation of m_oplus
        acc = tot = 0.
        m_oplus = d # in case β = 0
        for k in range(1,d+1):
            acc += consts2[k-1]
            term = beta/(1 + beta*(k)) * acc
            if consts2[k-1] <= term:
                # m_oplus = max(1,k-1) # in case we stop at the first iteration
                m_oplus = k-1
                break
            tot = acc     # from Borgelt
        if not tot > 0.:  # from Borgelt
            tot = 1.
        consts2 = None
        for j in range(d):
            # formula from Borgelt p.5 does not work!
            t = 1/(1-beta) * ((1 + beta*(m_oplus - 1))/tot * consts[j] - beta)
            # taking the max is needed according to Klawonn & Höppner
            diagonal[r,j] = max(0,t)
    return diagonal



class Borgelt(cluster.FCMeans):
    def __init__(self, n_clusters, beta, max_iter = 300, tol = 1e-4,
            verbose = Verbosity.NONE, centers = None, memberships = None,
            weights = None, m = 2., v = 2.):
        cluster.FCMeans.__init__(self, n_clusters, max_iter, tol, verbose,
                centers, memberships, weights, m = m, v = v)
        self.beta = beta

    def fit(self, X, y=None):
        self._costs = []     # store inertia values
        self.X = X
        self.memberships, self.centers, self.weights = self._alternate_descent()
        self._inertia = self._compute_inertia(self.X, self.centers, self.memberships,
                self.weights)

    def _update_weights(self, X, n_clusters, memberships, centers):
        return _update_weights(X, n_clusters, memberships, centers,
                beta = self.beta)

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


