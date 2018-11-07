import numpy as np
import scipy.sparse as sp
import cluster
import keller
import random
import copy
from config import Verbosity, verbose_function
from sklearn.preprocessing import normalize

DEBUG = []

def __contains_neg(mat):
    (a,b) = mat.shape
    for row in range(a):
        for col in range(b):
            if mat[row,col] < 0:
                return True
    return False

def _pfscm_prox_gradient(mat, grad, prox, lip_const, prox_arg, max_iter = 10, tol=1e-4, grad_args = None):
    diff = tol+1.
    debug = copy.deepcopy(mat)
    for _ in range(max_iter):
        if grad_args is None:
            mat2 = grad(mat)
        else:
            mat2 = grad(mat, *grad_args)
        #print("Fin gradient")
        #print(mat2)
        aux = (-lip_const) * mat2 + mat
        new_mat = prox((-lip_const) * mat2 + mat, l=prox_arg)
#        assert not __contains_neg(new_mat)      # TODO: following informations should be
        if __contains_neg(new_mat):
            print("!!!!")                # in final version of the implementation (meta-infos?)
#            print(debug)                 # Base class with the Hessian + prox stuff?
            print(mat)
            print(mat2)
            print(aux)
            print(new_mat)
            print(lip_const)
#            DEBUG.append((mat, grad_args, lip_const))
#            try:
#                print(aux.todense())
#            except:
#                print(aux)
        #print("Fin proximal")
        if sp.issparse(mat):
            new_diff = sp.linalg.norm(new_mat - mat)
        else:
            new_diff = np.linalg.norm(new_mat - mat)
        if abs(diff - new_diff) < tol:
            break
        diff = new_diff
        mat = new_mat
    return mat

def _prox_abs(x, l):
    return np.sign(x) * (abs(x) -l) if abs(x) > l else 0

def _pfscm_sparse_prox_op(weights, l):
    c,d = weights.shape
    z = sp.dok_matrix((c,d))
    for r in range(c):
        som = 0.0
        for jptr in range(weights.indptr[r], weights.indptr[r+1]):
            som += weights.data[jptr]
        for jptr in range(weights.indptr[r], weights.indptr[r+1]):
            j = weights.indices[jptr]
            z[r,j] = weights.data[jptr] + 1./d * (1 + _prox_abs(som - 1, d*l) - som)
    return z.tocsr()

def _pfscm_prox_op(w, l):
    if sp.issparse(w):
        return _pfscm_sparse_prox_op(w, l)
    c,d = w.shape
    z = np.zeros((c,d))
    for r in range(c):
        #print(w)
        som = np.sum(w[r,:])
        #print("som", som)
        for j in range(d):
            #print("wrj:", w[r,j], "\t exp:", 1./d * (1 + _prox_abs(som - 1, d*l) - som))
            z[r,j] = w[r,j] + 1./d * (1 + _prox_abs(som - 1, d*l) - som)
            #print("prox_abs", _prox_abs(som - 1, d*l))
            #print("prox", z[r,j])
    return z

# FIXME? is it worth to produce a sparse matrix?
def _pfscm_grad(weights, x, centroids, memberships, m = 2., v = 2.):
    c,d = weights.shape
    n,d2 = x.shape
    assert(d == d2)
    if sp.issparse(x):
        weights = weights.tocsc()
        assert sp.isspmatrix_csc(x), "Only the CSC format is supported (iteration on dimensions)"
        # Supports `weights` being sparse or not but produces a new sparse matrix
        # Do we have to use a d*c loop however?
        # Can we "fusion" this with the prox operator in order to remove useless dims?..
        w = sp.dok_matrix((c,d))
        for j in range(d):
            #for r in range(c):
                temps = np.zeros((c,))
                for iptr in range(x.indptr[j], x.indptr[j+1]):
                    i = x.indices[iptr]
                    if sp.issparse(weights):
                        for rptr in range(weights.indptr[j], weights.indptr[j+1]):
                            r = weights.indices[rptr]
                            temps[r] += v * memberships[r, i] ** m * weights.data[rptr] * (np.linalg.norm(x.data[iptr] - centroids[r, j]))**2
                    else:
                        for r in range(c):
                            temps[r] = v * memberships[r,i] ** m * weights[r,j] * (x.data[iptr] - centroids[r, j])**2
#                        for jptr in range(weights.indptr[r], weights.indptr[r+1]):
#                            # Or should it be the other way around:
#                            #  * iteration on the weights to get pointers to the clusters
#                            #  * loop first on the weights then on the data X
#                            j2 = weights.indices[jptr]
#                            if j2 == j:
#                                t += v * memberships[r, i] ** m * weights.data[jptr] * (x.data[iptr] - centroids[r, j])**2
#                                break
#                    else:
#                        t += v * memberships[r,i] ** m * weights[r,j] * (x.data[iptr] - centroids[r, j])**2
                    for r2 in range(c):
                        w[r2,j] = temps[r2]
        return w.tocsr()
    else:
        w = np.zeros((c,d))
        for j in range(d):
            for r in range(c):
                w[r,j] = v * sum(memberships[r,i] ** m * weights[r,j] * (x[i,j] - centroids[r,j])**2 for i in range(n))
        return w


# TODO support "smart" initial weight matrices
# e.g. "full" (constant) matrix (constant size!) or pre-filled matrices from TF IDF
def pfscm(x, n_clusters, max_iter = 300, tol=1e-4, verbose=False, gamma_const = 1000, centers = None, memberships = None):
    (n,d) = x.shape
    v = 2.
    m = 2.
    centers = clustering._init_centers(centers, x, n_clusters)
    if memberships is None:
        memberships = np.zeros((n_clusters, n))
    for i in range(n):
        r = random.randint(0, n_clusters - 1)
        memberships[r,i] = 1
    weights = np.ones((n_clusters, d))
    if sp.issparse(x):
        weights = sp.csr_matrix(weights)
        xcsc = x.tocsc()     # Used in gradient descent
    for i in range(max_iter):
        clus = clustering.weighted_fcm(x, n_clusters, weights, max_iter = max_iter, tol = tol, centers = centers, memberships = memberships)
        new_centers = clus.centers
        new_memberships = clus.memberships
        # computation of an ad-hoc descent step...
        h_norm = 0
        if sp.issparse(x):
            for r in range(n_clusters):
                for i in range(n):
                    for jptr in range(x.indptr[i], x.indptr[i+1]):
                        j = x.indices[jptr]
                        h_norm += v * new_memberships[r,i] ** m * np.linalg.norm(x.data[jptr] - new_centers[r,j])**2
            cst = 1./h_norm
            new_weights = _pfscm_prox_gradient(weights, _pfscm_grad, _pfscm_prox_op, cst, prox_arg = cst * gamma_const,
                    max_iter = max_iter, tol = tol, grad_args = (xcsc, new_centers, new_memberships))
            weight_gain = sp.linalg.norm(new_weights - weights)
        else:
            for j in range(d):
                for r in range(n_clusters):
                        h_norm += v * sum(new_memberships[r,i] ** m * (x[i,j] - new_centers[r,j])**2 for i in range(n))
            cst = 1./h_norm
            new_weights = _pfscm_prox_gradient(weights, _pfscm_grad, _pfscm_prox_op, cst, prox_arg = cst * gamma_const,
                    max_iter = max_iter, tol = tol, grad_args = (x, new_centers, new_memberships))
            weight_gain = np.linalg.norm(new_weights - weights)
        gain = np.linalg.norm(new_centers - centers) + weight_gain + np.linalg.norm(new_memberships - memberships)
        if gain < tol:
            print("Itération", i)
            break
        else:
            #print("gain", gain, "itération", i)
            #print(centers)
            #print(new_centers)
            #print(": new_weights :")
            #print(new_weights)
            #print(np.linalg.norm(new_memberships - memberships), np.linalg.norm(new_centers - centers))
            pass
            #print(np.linalg.norm(new_centers - centers)) # + np.linalg.norm(new_weights - weights))
            #print(centers)
            #print(new_centers)
        weights = new_weights
        centers = new_centers
        memberships = new_memberships

    return SubspaceClustering(x, memberships, centers, weights)


### Sparsity inducing proximal descent for subspace clustering

def _simplex_proj_omitting(w_r, to_zero):
    (d,) = w_r.shape
    ds = d - len(to_zero)
    w2 = np.zeros((d,))
    som = sum(w_r[j] for j in range(d) if j not in to_zero)
    for j in range(d):
        if j not in to_zero:
            w2[j] = w_r[j] + 1/ds * (1 - som)
    return w2

def _simplex_alt(w_r):
    K = np.full(w_r.shape, 1)
    return w_r + 1./d*K.T * (1 - K.dot(w_r))

def _is_zero(x):
    return x == 0

def _count_zeros(wr):
    t = 0
    for x in wr:
        if _is_zero(x):
            t += 1
    return t

def _sparsity_cost_function(initial_weights_row, new_weights_row, constant):
    (d,) = initial_weights_row.shape
    t = _count_zeros(new_weights_row)
    return constant*(d - t) + 1/2 * np.linalg.norm(initial_weights_row - new_weights_row)**2

def _sparsity_sparse_cost_function(initial_row, new_weights_row, constant):
    lzero = new_weights_row.nnz
    return constant*lzero + 1/2 * sp.linalg.norm(initial_row - new_weights_row)**2

# projects non-zero entries to the simplex
# doesn't behave as _simplex_proj_omitting
def _sparse_matrix_projection(w, r):
    (c,d) = w.shape
    som = np.sum(w.data[w.indptr[r]:w.indptr[r+1]])
#    if omitting is not None:
#        ds = w.indptr[r+1] - w.indptr[r] - len(omitting)
#        w.data[w.indptr[r]:w.indptr[r+1]] += 1./ds * (1 - som)
#        w.data[np.array(omitting)] = 0.
#    else:
    ds = w.indptr[r+1] - w.indptr[r]
    w.data[w.indptr[r]:w.indptr[r+1]] += 1./ds * (1 - som)
    return d - (w.indptr[r+1] - w.indptr[r])

# this one does
def _sparse_matrix_initial_projection(w):
    (c,d) = w.shape
    w2 = np.zeros((c,d))
    for r in range(c):
        som = np.sum(w.data[w.indptr[r]:w.indptr[r+1]])
        w2[r,:] = 1/d * (1 - som)
        for jptr in range(w.indptr[r], w.indptr[r+1]):
            j = w.indices[jptr]
            w2[r,j] += w.data[jptr]
    return sp.csr_matrix(w2)

# TODO: more in place updates
def _sparsity_prox_op(w, l):
    (c,d) = w.shape
    if sp.issparse(w):
        w = w.tocsr()
        # FIXME why is this still necessary?
        # w.data[w.data < 0] = 0
        sols = []
        indptr_list = [0]
        data_list = []
        indices_list = []
        #print("input")
        #print(w)
        w = _sparse_matrix_initial_projection(w)
        #print("projection")
        #print(w)
        for r in range(c):
            # Initial projection on the simplex
            initial_row = w[r,:]
            best_row = initial_row
            n_zeros = d - initial_row.nnz
            #print("initial row")
            #print(initial_row)
            #print("initial zeros")
            #print(n_zeros)
            cost = _sparsity_sparse_cost_function(initial_row, initial_row, l)
            # Descent towards the best sparse solution
            while n_zeros < d-1:
                imin = np.argmin(w.data[w.indptr[r]:w.indptr[r+1]])
                #print("imin")
                #print(imin)
                w.data[imin+w.indptr[r]] = 0.0   # FIXME we should not modify the matrix before knowing its the best solution
                w.eliminate_zeros()  # FIXME inefficient?
                n_zeros += 1
                _sparse_matrix_projection(w, r)
                #print(w)
                row = w[r,:]
                new_cost = _sparsity_sparse_cost_function(initial_row, row, l)
                #print("cost", cost)
                #print("new_cost", new_cost)
                if new_cost > cost:
                    break
                else:
                    best_row = row
            sols.append(best_row)
            lastptr = indptr_list[-1]
            indptr_list.extend([x + lastptr for x in best_row.indptr[1:]])
            data_list.extend(best_row.data)
            indices_list.extend(best_row.indices)
            #print("solution so far:")
            #print(data_list)
        result = sp.csr_matrix(       # FIXME produce CSC matrix instead
                (np.array(data_list),
                 np.array(indices_list),
                 np.array(indptr_list)),
                shape = (c,d))
        DEBUG.append(result.copy())
        #print("result:")
        #print(result)
        return result
    else:
        #print("input")
        #print(w)
        for r in range(c):
            w[r,:] = _simplex_proj_omitting(w[r,:], set())
        #print("projection")
        #print(w)
        w2 = copy.copy(w)
        for r in range(c):
            ##print("Starting\t", w2[r,:])
            row = w2[r,:] # does not copy w2
            initial_row = copy.copy(w2[r,:])
            cost = _sparsity_cost_function(initial_row, row, l)
            ##print("Cost\t\t", cost)
            n_zeros = _count_zeros(row)
            zeros = set()
            while n_zeros < d-1:
                smallest_i = min((i for i in range(d) if not _is_zero(row[i])), key = lambda i: row[i])
                n_zeros += 1
                row[smallest_i] = 0.0
                zeros.add(smallest_i)
                # reshape is needed by normalize for sklearn > 0.19
                # row = ... is kept for clarity
                # row = normalize(row.reshape(1,-1), norm='l1', copy=False).reshape((d,))
                row = _simplex_proj_omitting(row, zeros)
                ##print("Row:\t\t", row)
                new_cost = _sparsity_cost_function(initial_row, row, l)
                ##print("Cost:\t\t", new_cost)
                if new_cost > cost:
                    break
                else:
                    cost = new_cost
                    w[r,:] = row # copies back the row into w
        #print("result:")
        #print(w)
        return w

def _cost_function(x, centers, memberships, weights, gamma):
    m = v = 2.
    (n,d) = x.shape
    (c,d) = centers.shape
    tot = 0.0
    for r in range(c):
        if sp.issparse(weights):
            assert sp.isspmatrix_csc(x)
            for jptr in range(weights.indptr[r], weights.indptr[r+1]):
                j = weights.indices[jptr]
                for iptr in range(x.indptr[j], x.indptr[j+1]):
                    i = x.indices[iptr]
                    tot += memberships[r,i] ** m * weights.data[jptr] ** v * np.linalg.norm(centers[r,j] - x.data[iptr])**2
        else:
            for j in range(d):
                for i in range(n):
                    tot += memberships[r,i] ** m * weights[r,j] ** v * np.linalg.norm(centers[r,j] - x[i,j])**2
    tot += gamma * len(np.argwhere(weights != 0))
    return tot

@verbose_function
def sparsity_pfscm(x, n_clusters, max_iter = 100, tol=1e-4, gamma_const = 0.1, verbose = Verbosity.NONE, **kwargs):
    if verbose is False or verbose is None:
        verbose = Verbosity.NONE
    if verbose is True:
        verbose = Verbosity.ALL
    DEBUG = []
    (n,d) = x.shape
    m = v = 2.
    centers = cluster._init_centers(None, x, n_clusters)
    #print(centers)
    memberships = np.zeros((n_clusters, n))
    for i in range(n):
        r = random.randint(0, n_clusters - 1)
        memberships[r,i] = 1
    weights = np.ones((n_clusters, d))
    weights *= 1./d
    if sp.issparse(x):
        weights = sp.csc_matrix(weights)
        xcsc = x.tocsc()     # Used in gradient descent
        #print("Fin CSC")
    for niter in range(max_iter):
        if verbose & Verbosity.COST_FUNCTION:
            xs = xcsc if sp.issparse(x) else x
            kwargs["cost_function"].append(_cost_function(xs, centers,
                memberships, weights, gamma_const))
        #print("Iteration #{}".format(niter))
        clus = cluster.weighted_fcm(x, n_clusters, weights, max_iter = max_iter, tol = tol, centers = centers, memberships = memberships, verbose=verbose, **kwargs)
        #print("Fin clustering")
        new_centers = clus[1]
        #print(new_centers)
        new_memberships = clus[0]
        #print(weights)
        # computation of an ad-hoc descent step...
        h_norm = 0
        if sp.issparse(x):
            for r in range(n_clusters):
                for i in range(n):
                    for jptr in range(x.indptr[i], x.indptr[i+1]):
                        j = x.indices[jptr]
                        h_norm += v * new_memberships[r,i] ** m * np.linalg.norm(x.data[jptr] - new_centers[r,j])**2
            cst = 1./(h_norm)
            new_weights = _pfscm_prox_gradient(weights, _pfscm_grad, _sparsity_prox_op, cst, prox_arg = cst * gamma_const,
                    max_iter = 10, tol = tol, grad_args = (xcsc, new_centers, new_memberships))
            weight_gain = sp.linalg.norm(new_weights - weights)
        else:
            for j in range(d):
                for r in range(n_clusters):
                        h_norm += v * sum(new_memberships[r,i] ** m * (x[i,j] - new_centers[r,j])**2 for i in range(n))
            cst = 1./h_norm
            new_weights = _pfscm_prox_gradient(weights, _pfscm_grad, _sparsity_prox_op, cst, prox_arg = cst * gamma_const,
                    max_iter = 10, tol = tol, grad_args = (x, new_centers, new_memberships))
            weight_gain = np.linalg.norm(new_weights - weights)
        #print(cst)
        gain = np.linalg.norm(new_centers - centers) + weight_gain + np.linalg.norm(new_memberships - memberships)
        #print(gain)
        if gain < tol:
            #print("Itération", niter)
            break
        else:
            #print("gain", gain, "itération", i)
            #print(centers)
            #print(new_centers)
            pass
            #print(np.linalg.norm(new_centers - centers)) # + np.linalg.norm(new_weights - weights))
            #print(centers)
            #print(new_centers)
            #print(np.linalg.norm(new_weights - weights))
            #print(new_weights)
            #print(np.linalg.norm(new_memberships - memberships))
        weights = new_weights
        centers = new_centers
        memberships = new_memberships

    clus = cluster.weighted_fcm(x, n_clusters, weights, max_iter = max_iter, tol = tol, centers = centers, memberships = memberships)
    memberships = clus[0]
    centers = clus[1]

    if verbose & Verbosity.COST_FUNCTION:
        xs = xcsc if sp.issparse(x) else x
        kwargs["cost_function"].append(_cost_function(xs, centers,
            memberships, weights, gamma_const, distance_f = distance_f))
    return memberships, centers, weights


def extend(new_points, clus):
    (n_clusters, _) = clus.centers.shape
    new_memberships = clustering._update_memberships(new_points, n_clusters,
            clus.weights, clus.centers)
    new_points = np.concatenate((clus.points, new_points), axis = 0)
    new_memberships = np.concatenate((clus.memberships, new_memberships),
            axis = 1)
    return SubspaceClustering(new_points, new_memberships, clus.centers,
            clus.weights)



class PFSCM(cluster.FCMeans):
    def __init__(self, n_clusters, weights_gamma, max_iter = 300, tol = 1e-4, verbose =
            Verbosity.NONE, centers = None, memberships = None, weights = None):
        cluster.FCMeans.__init__(self, n_clusters, max_iter = max_iter, tol = tol, verbose = verbose,
                centers = centers, memberships = memberships, weights = weights, m = 2., v = 2.)
        self.weights_gamma = weights_gamma

    def fit(self, X, y=None):
        #  FIXME general framework for observations?
        self._costs = []     # store inertia values
        self.X = X
        self.memberships, self.centers, self.weights = self._alternate_descent()
        self._inertia = self._compute_inertia(self.X, self.centers, self.memberships,
                self.weights)

    def _alternate_descent(self):
        n, d = self.X.shape
        self.centers = cluster._init_centers(self.centers, self.X, self.n_clusters)
        self.memberships = cluster._init_memberships(self.memberships, self.centers, self.X, self.n_clusters)
        self.weights = cluster._init_weights(self.weights, self.X, self.n_clusters)
        for niter in range(self.max_iter):
            if self.verbose & Verbosity.COST_FUNCTION:
                self._log_cost(self.centers, self.memberships, self.weights)
            new_memberships, new_centers = super()._alternate_descent()
            #cst = 1/self._compute_step(new_memberships, new_centers, n, d, self.n_clusters)
            #print("Cst is {}".format(cst))
            #cst = np.sqrt(cst)
            #print("Cst is now {}".format(cst))
            cst = np.min(1/self._weights_hessian(new_memberships,
                    new_centers, n, d, self.n_clusters))
            new_weights = _pfscm_prox_gradient(self.weights, self._weights_gradient,
                    _pfscm_prox_op, cst, prox_arg = cst * self.weights_gamma,
                    max_iter = 10, tol = self.tol, grad_args = (self.X, new_centers, new_memberships))
            gain = np.linalg.norm(new_centers - self.centers)
            if gain < self.tol:
                break
            self.weights = new_weights
            self.centers = new_centers
            self.memberships = new_memberships
        return self.memberships, self.centers, self.weights

    def _weights_hessian(self, new_memberships, new_centers, n, d,
            n_clusters):
        z = np.zeros_like(self.weights)
        for j in range(d):
            for r in range(n_clusters):
                z[r,j] = self.v * sum(new_memberships[r,i] ** self.m * (self.X[i,j] -
                    new_centers[r,j])**2 for i in range(n))
        return z

    def _weights_gradient(self, weights, x, centers, memberships):
        p = 1. / (1 - self.v)
        m_memberships = memberships ** self.m
        T = (x[:,None] - centers) ** 2
        S = np.einsum("ij,jil->il",m_memberships, T)    # FIXME: notations
        gradient = self.v * weights ** (self.v - 1) * S
        return gradient

    def _compute_step(self, new_memberships, new_centers, n, d, n_clusters):
        h_norm = 0
        for j in range(d):
            for r in range(n_clusters):
                h_norm += self.v * sum(new_memberships[r,i] ** self.m * (self.X[i,j] -
                    new_centers[r,j])**2 for i in range(n))
        return h_norm

class Prosecco(PFSCM):
    def __init__(self, n_clusters, weights_gamma, max_iter = 300, tol = 1e-4, verbose =
            Verbosity.NONE, centers = None, memberships = None, weights = None):
        PFSCM.__init__(self, n_clusters, weights_gamma, max_iter, tol, verbose,
                centers, memberships, weights)

    def _alternate_descent(self):
        n, d = self.X.shape
        self.centers = cluster._init_centers(self.centers, self.X, self.n_clusters)
        self.memberships = cluster._init_memberships(self.memberships, self.centers, self.X, self.n_clusters)
        self.weights = cluster._init_weights(self.weights, self.X, self.n_clusters)
        for niter in range(self.max_iter):
            if self.verbose & Verbosity.COST_FUNCTION:
                self._log_cost(self.centers, self.memberships, self.weights)
            new_memberships, new_centers = cluster.FCMeans._alternate_descent(self) # oh well
            #new_memberships, new_centers = cluster.weighted_fcm(self.X, self.n_clusters, self.weights, centers = self.centers, memberships = self.memberships)
            cst = 1/2*np.min(1/self._weights_hessian(new_memberships,
                    new_centers, n, d, self.n_clusters))
            #cst = 1./(np.sum(self._weights_hessian(new_memberships,
            #        new_centers, n, d, self.n_clusters)))
            #print(cst)
            #print(self.weights)
            #print(new_centers)
            new_weights = _pfscm_prox_gradient(self.weights, self._weights_gradient,
                    _sparsity_prox_op, cst, prox_arg = cst * self.weights_gamma,
                    max_iter = 10, tol = self.tol, grad_args = (self.X, new_centers, new_memberships))
            #print(new_weights)
            gain = np.linalg.norm(new_centers - self.centers) + np.linalg.norm(new_weights - self.weights) + np.linalg.norm(new_memberships - self.memberships)
            if gain < self.tol:
                break
            self.weights = new_weights
            self.centers = new_centers
            self.memberships = new_memberships
        return self.memberships, self.centers, self.weights

