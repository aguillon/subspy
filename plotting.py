import numpy as np
import matplotlib.pyplot as plt

import misc

def plot_majority(clustering, threshold = 0.0, fig = None, colors = misc.colors,
        weak_color = None, _axis = True, gca_obj = None, depthshade = True):
    if fig is None:
        fig = plt.figure()
    (n,d) = clustering.X.shape
    if d == 2:
        ax = fig.gca()
        for i in range(n):
            rmax = misc.majority_indice(clustering.memberships[:,i])
            color = misc.colors[rmax]
            if clustering.memberships[rmax,i] >= threshold:
                ax.plot(clustering.X[i,0], clustering.X[i,1],
                marker="o", color = color)
            elif weak_color is not None:
                ax.plot(clustering.X[i,0], clustering.X[i,1],
                marker="+", color = weak_color)
    else:
        ax = fig.gca(projection='3d')
        indices = np.argmax(clustering.memberships, axis=0)
        colors = np.array(misc.colors)[indices]
        mask = clustering.memberships[indices,np.arange(n)] > threshold
        points = clustering.X[mask,:]
        ax.scatter(points[:,0], points[:,1], points[:,2], c = colors[mask], marker="p", depthshade = depthshade)
        if weak_color is not None:
            points = clustering.X[~mask,:]
            ax.scatter(points[:,0], points[:,1], points[:,2], c = weak_color, marker="+", depthshade = depthshade)
    if gca_obj:
        ax.view_init(gca_obj.elev, gca_obj.azim)
    elif _axis:
        ax.axis("equal")

    return fig

def plot_centers(clustering, fig = None, colors = misc.colors, text=None):
    if fig is None:
        fig = plt.figure()
    (n,d) = clustering.points.shape
    (c,_) = clustering.centers.shape
    if d == 2:
        ax = fig.gca()
        for r in range(c):
            color = colors[r]
            if text is None:
                ax.plot(clustering.centers[r,0], clustering.centers[r,1], color =
                        color, marker = "p", markersize=24)
            else:
                ax.text(clustering.centers[r,0], clustering.centers[r,1], text,
                        color = color)
        ax.axis("equal")
    return fig


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