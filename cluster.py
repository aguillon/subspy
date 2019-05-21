import numpy as np
import scipy.sparse as sp
import random
import math
from config import Verbosity, verbose_function, logging

EPSILON = np.finfo(np.float32).eps

def labels_to_memberships(labels):
    n = labels.shape[0]
    k = len(np.unique(labels))
    memberships = np.zeros((k, n))
    for i,c in enumerate(labels):
        memberships[c,i] = 1
    return memberships

def memberships_to_labels(memberships):
    return np.argmax(memberships, axis=0)

# (Weighted) FCM algorithm: functional implementation
@verbose_function
def weighted_fcm(x, n_clusters, weights, max_iter=300, tol=1e-4, centers=None, memberships=None,  **kwargs):
    n, d = x.shape
    centers = _init_centers(centers, x, n_clusters, **kwargs)
    memberships = _init_memberships(memberships, centers, x, n_clusters, **kwargs)
    for i in range(max_iter):
        # FIXME : on veut pouvoir \'ecrire les appels \`a update sans expliciter
        # la liste qui va contenir les logs
        # Objet log avec lequel on int\'eragit au d\'ebut de l'algo ?
        # print(_cost_function(x, memberships, centers, weights))
        new_memberships = _update_memberships(x, n_clusters, weights, centers, **kwargs)
        new_centers = _update_centers(x, n_clusters, weights, new_memberships, **kwargs)
        if np.linalg.norm(new_centers - centers) < tol:
            break
        memberships = new_memberships
        centers = new_centers
    return (memberships, centers)

def fuzzy_c_means(x, n_clusters, **kwargs):
    _, d = x.shape
    weights = _init_weights(None, x, n_clusters, **kwargs)
    return weighted_fcm(x, n_clusters, weights, **kwargs)

class Clustering(object):
    def __init__(self, max_iter, tol, verbose):
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.labels_

class FuzzyClustering(Clustering):
    def __init__(self, n_clusters,  centers = None, memberships = None,
            weights = None, max_iter = 300, tol = 1e-4, verbose = Verbosity.NONE):
        Clustering.__init__(self, max_iter, tol, verbose)
        self.n_clusters = n_clusters
        self.centers = centers
        self.memberships = memberships
        self.weights = weights

    def _init_weights(self):
        assert hasattr(self, "X"), "This method should be called after fit()"
        _, d = self.X.shape
        if self.weights is None:
            self.weights = np.full((self.n_clusters, d), 1/d)  # Is there a constant matrix datatype in numpy?

    def fit(self, X, y=None):
        #  FIXME general framework for observations?
        self._costs = []     # store inertia values
        self.X = X
        self.memberships, self.centers = self._alternate_descent()
        self._inertia = self._compute_inertia(self.X, self.centers, self.memberships,
                self.weights)
        self.labels_ = memberships_to_labels(self.memberships)



@logging(Verbosity.MEMBERSHIPS, "log_memberships")
def _new_update_memberships(x, n_clusters, weights, centers, v = 2., m = 2., **kwargs):
    n,d = x.shape
    p = 1. / (1. - m)
    v_weights = weights ** v
    T = (v_weights[:,None] * (x - centers[:,None]) ** 2).sum(axis=2)
    memberships = T**p / (np.sum(T**p, axis=0))
    return memberships

@logging(Verbosity.MEMBERSHIPS, "log_memberships")
def _update_memberships(x, n_clusters, weights, centers, v = 2., m = 2., **kwargs):
    n,d = x.shape
    p = 1. / (1. - m)
    memberships = np.zeros((n_clusters, n))
    #print("Debut memberships")
    if sp.issparse(weights):
        weights = weights.tocsc()   # FIXME: what's the best way to avoid conversions?
    for sample_idx in range(n):
        special_care = []
        #if sample_idx % 10 == 0:
        #    print(sample_idx)
        # Computation of `som` denominator
        som = 0.0
        if sp.issparse(x):
            assert sp.isspmatrix_csr(x), "Only the CSR format is supported (iteration on points)"
            temps = np.zeros((n_clusters,))
            potential_clusters = set()    # used together with special_care
            for jptr in range(x.indptr[sample_idx], x.indptr[sample_idx+1]):
                j = x.indices[jptr]
                if sp.issparse(weights):
                    for r2ptr in range(weights.indptr[j], weights.indptr[j+1]):
                        r2 = weights.indices[r2ptr]
                        potential_clusters.add(weights.indices[r2ptr])
                        temps[r2] += weights.data[r2ptr] ** v * (x.data[jptr] - centers[r2, j])**2
                else:
                    for r2 in range(n_clusters):
                        temps[r2] +=  weights[r2,j] ** v * (x.data[jptr] - centers[r2,j])**2
                        potential_clusters.add(r2)
            som = np.sum(np.power(temps, p, where=(temps != 0)))
        else:
            for r2 in range(n_clusters):
                t = sum(weights[r2,j] ** v * (x[sample_idx,j] - centers[r2,j])**2 for j in range(d))
                if t != 0:
                    som += math.pow(t, p)
        # Computation of `temps` numerator for current sample
        temps = np.zeros((n_clusters,))
        if sp.issparse(x):
            for jptr in range(x.indptr[sample_idx], x.indptr[sample_idx+1]):
                j = x.indices[jptr]
                if sp.issparse(weights):
                    for cidx_ptr in range(weights.indptr[j], weights.indptr[j+1]):
                        center_idx = weights.indices[cidx_ptr]
                        temps[center_idx] += weights.data[cidx_ptr] ** v * (x.data[jptr] - centers[center_idx, j])**2
                else:
                    for center_idx in range(n_clusters):
                        temps[center_idx] += weights[center_idx,j] ** v * (x.data[jptr] - centers[center_idx,j])**2
        else:
            for center_idx in range(n_clusters):
                temps[center_idx] = sum(weights[center_idx,j] ** v * (x[sample_idx,j] - centers[center_idx,j])**2 for j in range(d))

        # FIXME: without out=out_ parameter, np.power seems to misbehave when some fields are 0
        # (observed in np 1.13.3)
        out_ = np.zeros_like(temps)
        numerators = np.power(temps, p, out=out_, where=(temps != 0))
        #print("Sample {}, numerators {}".format(sample_idx, numerators))
        for cluster_idx,numerator in enumerate(numerators):
            if numerator > 0:
                rslt = numerator / som
                memberships[cluster_idx, sample_idx] = rslt
            else:
                special_care.append(cluster_idx)
        t = len(special_care)
        if t > 0:
            for r in range(n_clusters):
                if r in special_care:
                    memberships[r, sample_idx] = 1./t
                    #print("Special : ({},{})".format(r, sample_idx))
                else:
                    memberships[r, sample_idx] = 0
                #print(memberships[:, sample_idx])
    return memberships


@logging(Verbosity.CENTERS, "log_centers")
def _update_centers(x, n_clusters, _weights, memberships, m = 2., **kwargs):
    n,d = x.shape
    t = np.zeros((n_clusters, n))
    clusters = np.zeros((n_clusters, d))
    t = memberships ** m
    if sp.issparse(x):
        assert sp.isspmatrix_csr(x), "Only the CSR format is supported (iteration on points)"
        for r in range(n_clusters):
            num = denom = 0.0
            for i in range(n):
                for jptr in range(x.indptr[i], x.indptr[i+1]):   # TODO rewrite using np operations
                    j = x.indices[jptr]
                    clusters[r,j] += t[r,i] * x.data[jptr]
            denom = np.sum(t[r,:])
            for j in range(d):
                if denom > 0:
                    clusters[r,j] /= denom
                else:
                    clusters[r,j] = 0.0
    else:
        # epsilons = np.random.randn(*clusters.shape)
        num = t.dot(x) # + epsilons
        denom = np.sum(t, axis=1).reshape((-1,1))
        denom[denom == 0] = EPSILON
        clusters = 1./denom * num
#        for r in range(n_clusters):
#            for j in range(d):
#                num = denom = 0.0
#                for i in range(n):  # TODO rewrite using np operations
#                    num += t[r,i] * x[i,j]
#                denom = np.sum(t[r,:])
#                if denom > 0:
#                    clusters[r,j] = num / denom
#                else:
#                    clusters[r,j] = 0.0
    #print(clusters)
    return clusters



# TODO: generate alternate optimization-style algorithms using
# a metaclass

# FCM algorithm: OOP implementation
class FCMeans(FuzzyClustering):
    def __init__(self, n_clusters, max_iter = 300, tol = 1e-4, verbose =
            Verbosity.NONE, centers = None, memberships = None, weights = None,
            m = 2, v = 2):
        super().__init__(n_clusters, centers, memberships, weights, max_iter, tol, verbose)
        self.m = m
        self.v = v

    def _update_memberships(self, X, n_clusters, weights, centers):
        return _new_update_memberships(X, n_clusters, weights, centers, verbose = self.verbose)

    def _update_centers(self, X, n_clusters, weights, memberships):
        return _update_centers(X, n_clusters, weights, memberships, verbose =
                self.verbose, m = self.m)

    def _alternate_descent(self):
        n, d = self.X.shape
        self._init_weights()
        centers = _init_centers(self.centers, self.X, self.n_clusters)
        memberships = _init_memberships(self.memberships, centers, self.X, self.n_clusters)
        for i in range(self.max_iter):
            if self.verbose & Verbosity.COST_FUNCTION:
                self._log_cost(centers, memberships, self.weights)
            # FIXME : on veut pouvoir \'ecrire les appels \`a update sans expliciter
            # la liste qui va contenir les logs
            # Objet log avec lequel on int\'eragit au d\'ebut de l'algo ?
            # print(_cost_function(x, memberships, centers, weights))
            new_memberships = self._update_memberships(self.X, self.n_clusters, self.weights, centers)
            new_centers = self._update_centers(self.X, self.n_clusters, self.weights, new_memberships)
            if np.linalg.norm(new_centers - centers) < self.tol:
                break
            memberships = new_memberships
            centers = new_centers
        if self.verbose & Verbosity.COST_FUNCTION:
            self._log_cost(centers, memberships, self.weights)
        return memberships, centers

    def _compute_inertia(self, X, centers, memberships, weights):
        v_weights = weights ** self.v
        T = (v_weights * (X[:,None] - centers) ** 2).sum(axis=2)
        m_memberships = memberships ** self.m
        return (m_memberships * T.T).sum()

    def _old_inertia(self):
        n,d = self.X.shape
        tot = 0.0
        for r in range(self.n_clusters):
            for i in range(n):
                tot += self.memberships[r,i] ** 2 * sum(self.weights[r,p] ** 2 * (self.X[i,p] - self.centers[r,p]) ** 2 for p in range(d))
        return tot

    def _log_cost(self, centers, memberships, weights):
        assert hasattr(self, "_costs"), ".fit() must have been called before accessing inertia values"
        self._costs.append(self._compute_inertia(self.X, centers, memberships, weights))


@logging(Verbosity.CENTERS, "log_centers")
def _init_centers(centers, points, n_clusters, **kwargs):
    (n,d) = points.shape
    if centers is None:
        if sp.issparse(points):
            assert sp.isspmatrix_csr(points), "Only the CSR format is supported (iteration on points)"
            mi = np.array(np.min(points, axis=0).todense()).flatten()
            ma = np.array(np.max(points, axis=0).todense()).flatten()
        else:
            mi = np.array([min(points[i,j] for i in range(n)) for j in range(d)])
            ma = np.array([max(points[i,j] for i in range(n)) for j in range(d)])
        centers = np.array([np.random.uniform(mi, ma, size = d) for _ in range(n_clusters)])
    return centers


@logging(Verbosity.MEMBERSHIPS, "log_memberships")
def _init_memberships(memberships, centers, points, n_clusters, **kwargs):
    (n, _) = points.shape
    if memberships is None:
        memberships = np.zeros((n_clusters, n))
        for i in range(n):
            r = random.randint(0, n_clusters - 1)
            memberships[r,i] = 1
    if type(memberships) is np.ndarray and len(memberships.shape) == 1:
        # sklearn 'labels' field
        memberships = labels_to_memberships(memberships)
    return memberships


def _init_weights(weights, points, n_clusters, **kwargs):
    (_, d) = points.shape
    if weights is None:
        weights = np.full((n_clusters, d), 1/d)  # Is there a constant matrix datatype in numpy?
    return weights


