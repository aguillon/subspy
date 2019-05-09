import numpy as np
import cluster
import factorization.commons as commons

from scipy.optimize import LinearConstraint, Bounds, minimize


class ArchetypalAnalysis():
    def __init__(self, n_archetypes, max_iter=300, tol=1e-4):
        self.n_archetypes = n_archetypes
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, X):
        self.X = X
        self._alternate_descent()

    def _init_A(self):
        memberships = cluster._init_memberships(None, None, self.X, self.n_archetypes)
        return memberships.T # different conventions between NMF and FCM

    def _init_Z(self):
        all_indices = commons.pca_hull_initialization(self.X)
        indices = np.random.choice(all_indices, self.n_archetypes, replace=False)
        self.beta = np.zeros((self.n_archetypes, self.X.shape[0]))
        self.beta[np.arange(self.n_archetypes), indices] = 1
        return self.X[indices]

    def _update_A(self, A, Z):
        # FIXME: factorize with F-FCM and other method below
        n,n_archetypes = A.shape
        A2 = np.zeros_like(A)
        for i in range(n):
            bounds = Bounds(np.zeros(n_archetypes), np.ones(n_archetypes))
            ones = np.array(1)
            constraint = LinearConstraint(np.ones(n_archetypes).T, ones, ones)
            res = minimize(lambda memberships:
                    commons._least_squares_cost(self.X[i], Z, memberships),
                A[i].T,
                method="SLSQP",
                bounds=bounds,
                constraints=[constraint])
            A2[i] = res.x
        return A2

    def _update_Z(self, A, Z):
        n,n_archetypes = A.shape
        # build the vbar matrix used in the CLS problem
        mask = np.eye(n_archetypes, dtype=bool)
        before_dot_products = A[:,:,None] * Z
        V = self.X[:,:,None] - np.tensordot(before_dot_products, ~mask, axes=[1,0])
        #V = self.X - A[:,None][~mask].dot(Z)
        diagonal = A.dot(mask)
        diagonal[diagonal == 0] = commons.EPSILON
        V /= diagonal[:,None,:]
        A2 = A**2
        vbar = (np.einsum("ir,ipr->pr", A2, V)/A2.sum(axis=0)).T
        # solve a CLS problem for archetypes coefficients
        beta = np.zeros_like(A.T)
        new_Z = np.zeros_like(Z)
        for l in range(n_archetypes):
            bounds = Bounds(np.zeros(n), np.ones(n))
            ones = np.array(1)
            constraint = LinearConstraint(np.ones(n).T, ones, ones)
            res = minimize(lambda beta:
                    commons._least_squares_cost(vbar[l], self.X, beta),
                self.beta[l],
                method="SLSQP",
                bounds=bounds,
                constraints=[constraint])
            beta[l] = res.x
            new_Z[l] = beta[l].dot(self.X)
        return beta, new_Z


    def _alternate_descent(self):
        self.A = self._init_A()
        self.Z = self._init_Z()
        for i in range(self.max_iter):
            new_A = self._update_A(self.A, self.Z)
            new_beta, new_Z = self._update_Z(new_A, self.Z)
            error = np.linalg.norm(self.Z - new_Z) + np.linalg.norm(self.A - new_A)
            if error < self.tol:
                print("%d itérations" % i)
                break
            #print(self.compute_inertia(self.X, new_W, new_H))
            print(new_A)
            self.A = new_A
            self.Z = new_Z
            self.beta = new_beta
        return self.A, self.Z

    def get_memberships(self):
        return self.A.T

    def get_centers(self):
        return self.Z



