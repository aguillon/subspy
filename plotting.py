import numpy as np
import matplotlib.pyplot as plt
import colorsys
import matplotlib.colors as colors_module
from scipy.spatial import ConvexHull

import misc
import cluster

def plot_majority(clustering, threshold = -1.0, fig = None, colors = misc.colors,
        weak_color = None, _axis = True, gca_obj = None, depthshade = True):
    if hasattr(clustering, "memberships"):
        memberships = clustering.memberships
    elif hasattr(clustering, "get_memberships"):
        memberships = clustering.get_memberships()
    else:
        raise ValueError("No 'memberships' field or 'get_memberships' method")
    if fig is None:
        fig = plt.figure()
    (n,d) = clustering.X.shape
    indices = np.argmax(memberships, axis=0)
    colors = np.array(misc.colors)[indices]
    mask = memberships[indices,np.arange(n)] > threshold
    points = clustering.X[mask,:]
    other_points = clustering.X[~mask,:]
    if d == 2:
        ax = fig.gca()
        ax.scatter(points[:,0], points[:,1], c = colors[mask], marker="o")
        if weak_color is not None:
            ax.scatter(other_points[:,0], other_points[:,1], c = weak_color, marker="+")
    else:
        ax = fig.gca(projection='3d')
        ax.scatter(points[:,0], points[:,1], points[:,2], c = colors[mask], marker="p", depthshade = depthshade)
        if weak_color is not None:
            ax.scatter(other_points[:,0], other_points[:,1], other_points[:,2], c = weak_color, marker="+", depthshade = depthshade)
    if gca_obj:
        ax.view_init(gca_obj.elev, gca_obj.azim)
    elif _axis:
        ax.axis("equal")
    return fig

def plot_centers(clustering, fig = None, colors = misc.colors, gca_obj = None,
        text = None, plot_hull = False, _axis=True):
    if hasattr(clustering, "centers"):
        centers = clustering.centers
    elif hasattr(clustering, "get_centers"):
        centers = clustering.get_centers()
    else:
        raise ValueError("No 'centers' field or 'get_centers' method")
    if fig is None:
        fig = plt.figure()
    (c,d) = centers.shape
    if d == 2:
        ax = fig.gca()
        colors = np.array(misc.colors)[:c]
        colors = [colorsys.rgb_to_hls(*colors_module.to_rgb(co)) for co in colors]
        colors = [colorsys.hls_to_rgb(co[0], 1 - 0.5 * (1 - co[1]), co[2]) for co in colors]
        if text is None:
            # for scatter, size is given as a surface...
            ax.scatter(centers[:,0], centers[:,1], c=colors, marker="p", s=280)
        else:
            ax.text(centers[:,0], centers[:,1], text,
                    color = colors)
        if plot_hull:
            color = "red"
            hull = ConvexHull(centers)
            hull = np.hstack((hull.vertices, hull.vertices)) # completing the loop
            ax.plot(centers[hull,0], centers[hull,1], c=color)
    if gca_obj:
        ax.view_init(gca_obj.elev, gca_obj.azim)
    elif _axis:
        ax.axis("equal")
    return fig


def plot_series(data, label_source, y_val=-1, fig=None, weights=None):
    if fig is None:
        fig = plt.figure()
    if hasattr(label_source, "labels_"):
        labels = label_source.labels_
    elif hasattr(label_source, "memberships"):
        labels = cluster.memberships_to_labels(label_source.memberships)
    else:
        assert type(label_source) == np.ndarray
        labels = label_source
    ax = fig.gca()
    _, d = data.shape
    indices = np.arange(d)
    for row in data[labels != y_val]:
        ax.plot(indices, row, "b--")
    for row in data[labels == y_val]:
        ax.plot(indices, row, "r-")


# TODO
# - support for plotting other stuff than distances (e.g. memberships)
#    (btw computed distances should be given as arguments)
# - support for more than 2 clusters
# - support for pre-labeled data
def plot_distances(points, centers, invisible_frame=True, labels = None, separation=None, dist=misc.euclidean_distance):
    c,d = centers.shape
    n,d2 = points.shape
    fig = plt.figure()
    ax = fig.gca()
    if invisible_frame:
        fig.patch.set_visible(False)
        ax.axis('off')
    graph_width = 4
    ax.set_xlim([-0.1, graph_width+0.1])
    ax.set_ylim([-0.1, graph_width+0.1])
    ax.arrow(0, 0, graph_width+0.1, 0., width=0.005, fc='k', ec='k', clip_on = False)
    ax.arrow(0, 0, 0., graph_width+0.1, width=0.005, fc='k', ec='k', clip_on = False)
    indices = np.arange(1,n+1)/(n/graph_width)
    ax.plot(np.arange(graph_width+1))
    for i,pt in enumerate(points):
        d1 = dist(pt, centers[0])
        d2 = dist(pt, centers[1])
        t = indices[i]
        if labels is not None:
            color = misc.colors[labels[i]]
        else:
            color = "b"
        ax.scatter(t, d1/d2 + t - 1, color=color)
    if separation is not None:
        separation /= (n/graph_width)
        ax.plot([separation,separation], [0,graph_width],color="r")
    plt.show()
    return fig
