import numpy as np
import scipy.sparse as sp
import itertools
from sklearn.metrics import accuracy_score

def flatten(ls):
    l = []
    for l2 in ls:
        for x in l2:
            l.append(x)
    return l

def unzip(ls):
    l1 = []
    l2 = []
    for (a,b) in ls:
        l1.append(a)
        l2.append(b)
    return (l1, l2)

def majority_indice(mat):
    (d,) = mat.shape
    ix = 0
    for i in range(d):
        if mat[ix] < mat[i]:
            ix = i
    return ix

colors = ['#002366','#908090','#4682dd','#ff0000', '#427800', '#129805']

def indices(*indicess):
    t = []
    for indices in indicess:
        t.extend(indices)
    return t

def euclidean_distance(x, y):
    return np.linalg.norm(x - y)


def rename(map, list):
    return [map[1][map[0].index(x)] for x in list]

def best_accuracy(truth, labels):
    truths = set(truth)
    labelss = list(set(labels))
    all_renamings = zip(itertools.repeat(labelss), itertools.permutations(truths))
    return max(accuracy_score(truth, rename(map, labels)) for map in all_renamings)


def iter_mat(mat):
    if sp.issparse(mat):
        for ptr in range(mat.indptr[0], mat.indptr[1]):
            yield (mat.indices[ptr], mat.data[ptr])
    else:
        (_, d) = mat.shape
        for r in range(d):
            yield (r, mat[0,r])
