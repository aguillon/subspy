import numpy as np
import misc

from numpy.random import randint

def make_diagonal(ls):
    return np.diagflat(ls)

def identity_mat(n, normalize=False):
    if normalize:
        return make_diagonal([math.sqrt(n) for i in range(n)])
    else:
        return make_diagonal([1 for i in range(n)])

# TODO? Take optional context object in order to ease up the
# writing of scripts (e.g. should we return the classes
# or not)
def gaussian(center, size, cov = None, seed = None, with_y = None):
    if seed:
        np.random.seed(seed)
    if not cov:
        cov = identity_mat(len(center))
    if type(cov) == list:
        flatten_me = False
        if(type(cov[0])) == np.ndarray:
            flatten_me = True
        cov = np.array(cov)
        if flatten_me:
            cov = np.hstack(cov)
    assert type(cov) == np.ndarray, "generation.gaussian now accepts arrays"
    if len(cov.shape) == 1:
        cov = make_diagonal(cov)
    t = np.random.multivariate_normal(center, cov, size)
    if with_y is None:
        return t
    else:
        return t, np.full(size, with_y)

def gaussians(centers, sizes, covs):
    r = []
    for i,c in enumerate(centers):
        r.extend(gaussian(c, sizes[i], covs[i]))
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

def time_series(n, d, n_clusters, l_segments=(3,6), offsets=None):
    """Simple generator of real-valued time series, with some distinctive
    portion for each cluster."""
    if type(l_segments) == int:
        l_segments = (l_segments, l_segments+1)
    if offsets is None:
        offsets = np.zeros((n_clusters, d))
    X = []
    y = []
    for r in range(n_clusters):
        if r < n_clusters - 1:
            size = int(n/n_clusters)
        else:
            size = n - len(X)
        pattern_length = randint(*l_segments)
        pattern_start = randint(0, d-pattern_length)
        # building the variance vector including the pattern
        print(pattern_start)
        variances = list()
        variances.append(randint(10, 20, size=pattern_start)/10)
        variances.append(randint(1, 5, size=pattern_length)/30)
        variances.append(randint(10, 20, size=d-(pattern_length+pattern_start))/10)
        X1, y1 = gaussian(offsets[r], size, variances, with_y = r)
        X.extend(X1)
        y.extend(y1)
    return np.array(X), np.array(y)


