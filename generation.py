import numpy as np
import misc

# ls elements have should be immutable
def make_diagonal(ls):
    r = []
    n = len(ls)
    for i in range(n):
        l = [0 for j in range(n)]
        l[i] = ls[i]
        r.append(l)
    return r

def identity_mat(n, normalize=False):
    if normalize:
        return make_diagonal([math.sqrt(n) for i in range(n)])
    else:
        return make_diagonal([1 for i in range(n)])

# TODO? Take optional context object in order to ease up the
# writing of scripts
def gaussian(center, size, cov = None, seed = None):
    if seed:
        np.random.seed(seed)
    if not cov:
        cov = identity_mat(len(center))
    elif type(cov) == list and type(cov[0]) != list:
        cov = make_diagonal(cov)
    elif type(cov) != list:
        raise ValueError("cov", cov)
    t = np.random.multivariate_normal(center, cov, size)
    return t, [0] * size

def gaussians(centers, sizes, covs):
    r = []
    for i,c in enumerate(centers):
        r.extend(gaussian(c, sizes[i], covs[i])[0])
    colors = [[i] * sizes[i] for i in range(len(sizes))]
    colors = misc.flatten(colors)
    r = np.array(r)
    return r,colors

# TODO: points could be (X,y)
def add_noise(points, r=0.1):
    n,d = points.shape
    size = int(n * r)
    noise = np.random.uniform(np.min(points, axis=0), np.max(points, axis=0),
            size = (size,d))
    return np.concatenate((points, noise))
