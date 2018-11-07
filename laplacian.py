import numpy as np
import scipy.sparse as sp
import random
import misc
import keller
import math
import cluster
from config import Verbosity, verbose_function, logging


def gaussian_kernel(pt, pt2, sigma):
    return np.exp(-np.linalg.norm(pt - pt2)**2/(2*sigma**2))

def _old_neighb_matrix(points, k = 5):
    neigh_dist = np.zeros((len(points)))
    for (i,pt) in enumerate(points):
        neighbors = []
        for pt2 in points:
            d = np.linalg.norm(pt - pt2)
            if len(neighbors) < k:
                neighbors.append(d)
                neighbors.sort()
            elif d < neighbors[-1]:
                neighbors.append(d)
                neighbors.sort()
                if len(neighbors) > k:
                    neighbors.pop()
        neigh_dist[i] = neighbors[-1]
    print(neigh_dist)
    sim = []
    for (i,pt) in enumerate(points):
        sim.append([])
        sig1 = neigh_dist[i]
        for (j,pt2) in enumerate(points):
            sig2 = neigh_dist[j]
            if np.linalg.norm(pt - pt2) > math.sqrt(sig1 * sig2):
                sim[-1].append(0)
            else:
                sim[-1].append(gaussian_kernel(pt, pt2, math.sqrt(sig1 * sig2)))
    return np.array(sim)

def neighb_matrix(points, k = 5):
    dists = ((points[None, :] - points[:, None])**2).sum(axis=2)
    indices = np.argsort(dists, axis=1)
    ind = indices[:,1:k]
    idx = np.ogrid[tuple(map(slice, ind.shape))]
    idx[1] = ind
    k_dists = np.zeros(dists.shape)
    k_dists[tuple(idx)] = dists[tuple(idx)]
    sigmas = np.max(np.sqrt(k_dists), axis=1)
    sigmass = sigmas[None,:] * sigmas[:,None]
    dists2 = dists/sigmass
    resultat = np.exp(-dists2/2)
    resultat[dists2 >= 1] = 0
    return resultat


def sparse_neighb_matrix(rows, k = 5):
    mat = neighb_matrix(rows, k = k)
    return sp.lil_matrix(mat)

def gaussian_affinity(x, sigma = 1., epsilon = 0.01):
    (n, d) = x.shape
    x = x.todense()
    aff = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            aff[i,j] = np.exp(-np.linalg.norm(x[i] - x[j])**2/(2*sigma**2))
            aff[j,i] = aff[i,j]
    return aff

@verbose_function
def laplacian_weighted_fcm(x, n_clusters, weights, affinity, gamma = 1., max_iter=300, tol=1e-4, centers=None, memberships=None, **kwargs):
    n, d = x.shape
    centers = cluster._init_centers(centers, x, n_clusters, **kwargs)
    memberships = cluster._init_memberships(memberships, centers, x, n_clusters, **kwargs)
    for i in range(max_iter):
        new_memberships = _new_laplacian_memberships(x, n_clusters, weights, centers, affinity, memberships, gamma = gamma, **kwargs)
        new_centers = cluster._update_centers(x, n_clusters, weights, new_memberships, **kwargs)
        if np.linalg.norm(new_centers - centers) + np.linalg.norm(new_memberships - memberships) < tol:
            break
        memberships = new_memberships
        centers = new_centers
    return memberships, centers

def laplacian_fcm(x, n_clusters, affinity, **kwargs):
    _, d = x.shape
    weights = np.full((n_clusters, d), 1/d)
    return laplacian_weighted_fcm(x, n_clusters, weights, affinity, **kwargs)

@logging(Verbosity.MEMBERSHIPS, "log_memberships")
def _laplacian_memberships(x, n_clusters, weights, centers, affinity, previous_memberships, gamma = 1., v = 2., m = 2.):
    n,d = x.shape
    U = previous_memberships
    memberships = np.zeros((n_clusters, n))
    if sp.issparse(weights):
        weights = weights.tocsc()   # FIXME: what's the best way to avoid conversions?
    for i in range(n):
        prod_u_s = np.zeros((n_clusters,))
        denom = np.zeros((n_clusters,))
        inv = 0.0
        for r in range(n_clusters):
            d2_ri = np.sum(weights[r, :] ** v * np.asarray(x[i, :] - centers[r, :]).reshape(-1)**2)
            sum_s = gamma*sum(affinity[i, :])
            denom[r] = 2*d2_ri + 4*sum_s
            prod_u_s[r] = gamma*np.sum(4*(U[r,:]*affinity[i,:]))
            inv += 1./denom[r]
        inv = 1./inv
        lambda_i = inv * (sum(prod_u_s[r]/denom[r] for r in range(n_clusters)) - 1)
        for r in range(n_clusters):
            memberships[r,i] = (prod_u_s[r] - lambda_i)/denom[r]
    return memberships

def _new_laplacian_memberships(x, n_clusters, weights, centers, affinity,
        previous_memberships, gamma = 1., v = 2.):
    n,d = x.shape
    v_weights = weights ** v
    sum_s = 4*gamma*affinity.sum(axis=1)
    prod = (4*gamma*np.einsum("rj,ij->ri", previous_memberships, affinity)).T
    T = (v_weights * (x[:,None] - centers) ** 2).sum(axis=2)
    E = 2*T + sum_s.reshape((-1,1))
    lambd = 1/((1/E).sum(axis=1)) * ((prod / E).sum(axis=1) - 1)
    return ((prod - lambd.reshape((-1,1)))/E).T


class WLFC(keller.Keller):
    def __init__(self, n_clusters, sim_gamma, max_iter = 300, tol = 1e-4,
            verbose = Verbosity.NONE, centers = None, memberships = None,
            weights = None, m = 2., v = 2., neigh_constant = None):
        if neigh_constant is None:
            neigh_constant = 7
        self.neigh_constant = neigh_constant
        keller.Keller.__init__(self, n_clusters, max_iter, tol, verbose,
                centers, memberships, weights, m = m, v = v)
        self.sim_gamma = sim_gamma

    def fit(self, X, y=None, affinity=None):
        if affinity is not None:
            self.affinity = affinity
        else:
            self.affinity = neighb_matrix(X, k = self.neigh_constant)
        super().fit(X, y)

    def _update_memberships(self, X, n_clusters, weights, centers, previous_memberships):
        return _new_laplacian_memberships(X, n_clusters, weights, centers,
                self.affinity, previous_memberships, gamma = self.sim_gamma)

    def _alternate_descent(self):
        n, d = self.X.shape
        self.centers = cluster._init_centers(self.centers, self.X, self.n_clusters)
        self.memberships = cluster._init_memberships(self.memberships, self.centers, self.X, self.n_clusters)
        self.weights = cluster._init_weights(self.weights, self.X, self.n_clusters)
        fcm_init = cluster.FCMeans(self.n_clusters)
        fcm_init.fit(self.X)
        # self.memberships, self.centers = fcm_init.memberships, fcm_init.centers
        for i in range(self.max_iter):
            if self.verbose & Verbosity.COST_FUNCTION:
                self._log_cost(self.centers, self.memberships, self.weights)
            new_memberships = self._update_memberships(self.X, self.n_clusters, self.weights, self.centers, self.memberships)
            new_centers = self._update_centers(self.X, self.n_clusters, self.weights, new_memberships)
            new_weights = self._update_weights(self.X, self.n_clusters, new_memberships, new_centers)
            if np.linalg.norm(new_centers - self.centers) < self.tol:
                break
            self.memberships = new_memberships
            self.centers = new_centers
            self.weights = new_weights
        if self.verbose & Verbosity.COST_FUNCTION:
            self._log_cost(self.centers, self.memberships, self.weights)
        self._fcm_init = fcm_init
        return self.memberships, self.centers, self.weights

    def _compute_inertia(self, X, centers, memberships, weights):
        v_weights = weights ** self.v
        T = (v_weights * (X[:,None] - centers) ** 2).sum(axis=2)
        m_memberships = memberships ** self.m
        terme_1 = (m_memberships * T.T).sum()
        return terme_1
        #terme_2 =

