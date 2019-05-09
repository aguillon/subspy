import numpy as np

EPSILON = np.finfo(np.float32).eps

from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
from itertools import combinations

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
    """Cost function for various convex problems"""
    return np.linalg.norm(X - memberships.dot(centers))**2



