import numpy as np
import cluster
import factorization.commons as commons

def _mu_update_W(X, W, H):
    W2 = W.copy()
    W2 = W2 * X.dot(H.T)
    denom = W.dot(H.dot(H.T))
    denom[denom == 0] = commons.EPSILON
    W2 = W2 / denom
    return W2

def _mu_update_H(X, H, W):
    H2 = H.copy()
    H2 *= W.T.dot(X)
    denom = W.T.dot(W.dot(H))
    denom[denom == 0] = commons.EPSILON
    H2 /= denom
    return H2

# Special case of projected gradient descent
def _positive_gradient(mat, grad_f, other_square, grad_args, beta_const=0.8,
        max_iter=50, tol=1e-4):
    for i in range(max_iter):
        grad = grad_f(mat, *grad_args)
        # inner loop to find the right descent step (Armijo rule)
        for t in range(1,100):
            alpha = beta_const**t
            # TODO: restart from last alpha and go up or down from it
            new_mat = np.maximum(0, mat - alpha*grad)
            diff = new_mat - mat
            # See "Projected Gradient Methods for NMF" by Lin
            quadratic_armijo = 0.5 * (grad * diff).sum() # 'sigma' cst not important?
            #print(diff.shape)
            #print(other_square.shape)
            #print(other_square.dot(diff.T).shape)
            quadratic_armijo += 0.5 * (diff.T *
                    (other_square.dot(diff.T))).sum()
            #print(quadratic_armijo)
            if quadratic_armijo <= 0: # We found the right alpha, we keep new_mat
                break
        if np.linalg.norm(new_mat - mat) < tol:
            break
        mat = new_mat
    return new_mat

class NMF:
    """Simple NMF implementation for experimental purpose. You should probably
    use sklearn.decomposition.nmf instead."""
    def __init__(self, n_components, descent_method="mu", max_iter=300, tol=1e-4):
        self.n_components = n_components
        self.W = None
        self.H = None
        self.descent_method = descent_method
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        self.X = X
        self.W, self.H = self._descent()
        self.inertia_ = self.compute_inertia(self.X, self.W, self.H)

    def _init_H(self):
        avg = np.sqrt(self.X.mean() / self.n_components)
        self.H = avg * np.random.randn(self.n_components, self.X.shape[1])
        self.H = np.abs(self.H)
        return self.H

    def _init_W(self):
        avg = np.sqrt(self.X.mean() / self.n_components)
        self.W = avg * np.random.randn(self.X.shape[0], self.n_components)
        self.W = np.abs(self.W)
        # FCM standard initialization method is not adapted to NMF
        #self.W = cluster._init_memberships(self.W, None, self.X, self.n_components)
        #self.W = self.W.T # FCM uses a different convention
        return self.W

    def _gradient_W(self, W, H):
        return (W.dot(H) - self.X).dot(H.T)

    def _gradient_Ht(self, Ht, W):
        # Descent algorithm is easier to implement if we work with Ht
        # (allows to share the code with W)
        return (W.T.dot(W.dot(Ht.T) - self.X)).T

    def _update_W(self, W, H):
        if self.descent_method == "mu":
            return _mu_update_W(self.X, W, H)
        else:
            return _positive_gradient(W, self._gradient_W, H.dot(H.T), [H])

    def _update_H(self, W, H):
        if self.descent_method == "mu":
            return _mu_update_H(self.X, H, W)
        else:
            Ht = _positive_gradient(H.T, self._gradient_Ht, W.T.dot(W), [W])
            return Ht.T

    def _descent(self):
        n, d = self.X.shape
        self.H = self._init_H()
        self.W = self._init_W()
        for i in range(self.max_iter):
            new_W = self._update_W(self.W, self.H)
            new_H = self._update_H(new_W, self.H)
            error = np.linalg.norm(self.H - new_H) + np.linalg.norm(self.W - new_W)
            if error < self.tol:
                print("%d itérations" % i)
                break
            # FIXME: implement logging protocol
            print(self.compute_inertia(self.X, new_W, new_H))
            self.W = new_W
            self.H = new_H
        return self.W, self.H

    def compute_inertia(self, X, W, H):
        return np.linalg.norm(X - W.dot(H))**2

    def get_memberships(self):
        return self.W.T
