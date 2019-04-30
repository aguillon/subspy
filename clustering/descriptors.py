import numpy as np
import scipy.sparse as sp
import sklearn as sk
import itertools
import re
from copy import deepcopy
from functools import partial

_hashtags_re = re.compile(r"\B#\w*[a-zA-Z]+\w*")

def hashtags_from_tweets(tweets):
    all_tags = set()
    for tweet in tweets:
        tags = re.findall(_hashtags_re, tweet)
        all_tags.update(tags)
    return all_tags

def hashtags_from_vectorizer(vectorizer):
    return [word for word in vectorizer.get_feature_names()
            if re.match(_hashtags_re, word)]

def _max(l):
    if len(l) == 0:
        return 0
    return max(l)

def _count_overlap(lists):
    c = np.zeros((max(_max(l) for l in lists)+1,), dtype=int)
    for list in lists:
        c[list] += 1
    c = c - 1
    c[c < 0] = 0
    return c.sum()

# max_overlap is supposed to be a relaxation; can we enumerate partitions of
# tags more efficiently (and add overlap if we're close to a solution)?
# Other idea: explore tags that maximize information gain first
def k_partitions(liste, k, max_overlap, max_size):
    def __too_long(l):
        return len(l) > max_size

    def _k_partitions(i):
        if len(liste) == i:
            yield [[] for _ in range(k)]
        else:
            parts = _k_partitions(i+1)
            for part in parts:
                if any(map(__too_long, part)):
                    continue
                print(part)
                overlap = _count_overlap(part)
                #yield deepcopy(part)
                for j in range(max_overlap - overlap + 1):
                    # Includes the "empty" position so we just yield
                    # the partition formed without the current element
                    for positions in itertools.combinations(range(k), j):
                        for p in positions:
                            part[p].append(liste[i])
                        yield deepcopy(part)
                        for p in positions:
                            part[p].pop()

    return filter(all, _k_partitions(0))

def _check_matrix(X, labels, cluster_id, tags):
    # Get all non-zero indices for a given cluster and a set of tags
    #print("check matrix %s %s %s" % (labels, cluster_id, tags))
    cluster = X[labels == cluster_id]
    mat = X[labels == cluster_id][:, tags] != 0
    # There is no sp.all(), we use sp.find instead: check that each
    # row is covered by at least one tag
    #print(mat)
    n,_ = cluster.shape
    non_zero_rows = sp.find(mat)[0]
    return len(np.unique(non_zero_rows)) == n

# TODO: - allow only a fraction of each cluster to be covered
#       - support tags constraints
#       - look for tags in an non-finite way! (online discovery)
class ConstrainedDescriptor:

    def __init__(self, X, cluster_labels, alpha, beta, vectorizer = None):
        self.X = X
        self.cluster_labels = np.array(cluster_labels)
        self.alpha = alpha
        self.beta = beta
        self.k = len(np.unique(cluster_labels))
        # If X is an occurrence matrix (e.g. TF-IDF) then it should be
        # sparse and we need a vectorizer matrix
        if sp.issparse(X):
            assert vectorizer is not None, "Word representation matrix needed"
            self.vectorizer = vectorizer

    def fit_descriptors(self, tags,
            together_constraints = None, apart_constraints = None):
        if together_constraints is not None or apart_constraints is not None:
            raise NotImplementedError
        tag_indices = self.vectorizer.transform(tags).nonzero()[1]
        #print(tag_indices)
        # Generate all admissible descriptors, regarding alpha
        # and the constraints
        for all_descs in k_partitions(tag_indices, self.k, self.beta, self.alpha):
            # check that each set of tag describes the corresponding cluster
            bad_tags = False
#            # DEBUG
#            for d in all_descs:
#                print(d)
#                for x in d:
#                    x = np.array(x)
#                    print(self.vectorizer.get_feature_names()[x], end=" ")
#                print()
            for cluster_r in range(self.k):
                if not _check_matrix(self.X, self.cluster_labels, cluster_r,
                        all_descs[cluster_r]):
                    # Current cluster was not covered
                    bad_tags = True
                    break
            if bad_tags:
                continue
            else: # We found a solution
                return all_descs
        return False
