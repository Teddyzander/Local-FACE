import numpy
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn import manifold, preprocessing
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
from scipy import spatial


def creat_custom(num=300, seed=10, noise=0.15):
    np.random.seed(seed + 1)
    X1, y1 = make_blobs(n_samples=int(num / 4), n_features=2, centers=[(0.8, -0.1), (-0.1, 0.45)],
                        cluster_std=noise * 2)
    X2, y2 = make_moons(n_samples=num, noise=noise)
    #X3, y3 = make_blobs(n_samples=int(num / 50), n_features=2, centers=[(-0.3, 0), (-0.3, 0)], cluster_std=noise * 3)

    X = np.concatenate((X1, X2)) #, X3))
    y = np.concatenate((y1, y2)) #, y3))

    X = preprocessing.minmax_scale(X)

    df = pd.DataFrame(preprocessing.minmax_scale(X), columns=["x1", "x2"])
    df["y"] = y
    return df


def create_two_moons(num=300, seed=10, noise=0.15):
    np.random.seed(seed + 1)
    X, y = make_moons(n_samples=num, noise=noise)
    X = preprocessing.minmax_scale(X)

    df = pd.DataFrame(preprocessing.minmax_scale(X), columns=["x1", "x2"])
    df["y"] = y
    return df


def create_circles(num=300, seed=10, noise=0.15):
    np.random.seed(seed + 1)
    X, y = make_circles(n_samples=num, noise=noise)
    X = preprocessing.minmax_scale(X)

    df = pd.DataFrame(preprocessing.minmax_scale(X), columns=["x1", "x2"])
    df["y"] = y
    return df


def create_blobs(num=300, seed=10, noise=0.15):
    np.random.seed(seed + 1)
    X, y = make_blobs(n_samples=num, n_features=2, centers=[(1, -1), (-1, 1)], cluster_std=1 + noise)
    X = preprocessing.minmax_scale(X)

    df = pd.DataFrame(preprocessing.minmax_scale(X), columns=["x1", "x2"])
    df["y"] = y
    return df


def plot_dataset(ax, df):
    # Plot the dataset.
    # dots_color_mapping = mpl.colors.ListedColormap(["#7B90D2", "#FAD689"])
    dots_color_mapping = mpl.colors.ListedColormap(["#ff0040", "#0000cc"])
    # dots_color_mapping =["b"]*num_points_top + ["r"]*num_points_left + ["#FAD689"]*num_points_middle + ["sienna"]*num_points_right

    ax.scatter(df.x1, df.x2, c=df.y,
               cmap=dots_color_mapping, s=20,
               #    edgecolors = 'black',
               zorder=1)

    # ax.grid(color="grey", linestyle="--", linewidth=0.5, alpha=0.75)

    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_ylabel(r"$x_2$", fontsize=14)
    ax.set_xlabel(r"$x_1$", fontsize=14)
    # ax.set_ylabel(r"$x_2~\longleftrightarrow$", fontsize=24)

    return ax


def plot_decision_boundary(ax, X_scaled, predictor, color_bar=False, levels=20):
    h = 0.01
    x1_min, x2_min = np.min(X_scaled, axis=0)
    x1_max, x2_max = np.max(X_scaled, axis=0)

    x1_cords, x2_cords = np.meshgrid(
        np.arange(x1_min, x1_max + h, h),
        np.arange(x2_min, x2_max + h, h)
    )
    new_X = np.c_[x1_cords.ravel(), x2_cords.ravel()]
    new_X_df = pd.DataFrame(new_X, columns=["x1", "x2"])

    def predict_func(X):
        return predictor.predict_proba(X)[:, 1]

    height_values = predict_func(new_X_df)
    height_values = height_values.reshape(x1_cords.shape)

    contour = ax.contourf(
        x1_cords,
        x2_cords,
        height_values,
        levels=20,
        cmap=plt.cm.RdBu,
        alpha=0.8,
        zorder=0
    )
    if color_bar:
        cbar = plt.colorbar(contour, ax=ax, fraction=0.1)
        cbar.ax.tick_params(labelsize=20)
    return ax


def find_cf(x0, data, classifier, k=10, thresh=0.6):
    """
    Find a valid counterfactual by searching through nearby data points
    Args:
        x0: starting point n array
        data: n by m array of all data in the field
        classifier: the trained classifier
        k: positive integer of how many neighbours to consider
        thresh: minimum value of probability classifier to terminate algorithm
    Returns:
        steps: n by p array of p steps to get from x0 to a valid counterfactual
        cf: valid counterfactual (last entry in steps)
    """
    steps = np.zeros((1, 2))
    # set up tree for k nearest neighbours
    tree = spatial.KDTree(list(zip(data[:, 0], data[:, 1])))

    # find closes k points to x0
    close = tree.query(x0, k=k, p=2)[1]

    # find probabilities of closest points
    vals = classifier.predict_proba(tree.data[close])[:, 1]

    # save best move and delete from tree and rebuild
    indx = np.argmax(vals)
    x_hat = tree.data[close[indx]]
    steps[0] = np.array(tree.data[close[indx]])
    cf = steps[0]
    temp = np.delete(tree.data, close[indx], 0)
    tree = spatial.KDTree(list(zip(temp[:, 0], temp[:, 1])))

    # repeat until valid counterfactual is found
    i = 0
    while classifier.predict_proba([x_hat])[0, 1] < thresh:
        # find closes k points to x0
        nei = tree.query(steps[i], k=k, p=2)
        close = nei[1]

        # find weighted probabilities of closest points
        vals = (1 / (1 + nei[0])) * classifier.predict_proba(tree.data[close])[:, 1]

        # save best move and delete from tree and rebuild
        indx = np.argmax(vals)
        x_hat = tree.data[close[indx]]
        best_step = np.array(tree.data[close[indx]])
        steps = np.append(steps, [best_step], axis=0)
        temp = np.delete(tree.data, close[indx], 0)
        tree = spatial.KDTree(list(zip(temp[:, 0], temp[:, 1])))

        cf = best_step
        i += 1
    return steps, cf


def find_cf_mom(x0, data, classifier, k=10, thresh=0.6, mom=0, alpha=0.05):
    """
    Find a valid counterfactual by searching through nearby data points using momentum
    Args:
        x0: starting point n array
        data: n by m array of all data in the field
        classifier: the trained classifier
        k: positive integer of how many neighbours to consider
        thresh: minimum value of probability classifier to terminate algorithm
        mom: positive int of number of last steps used to build momentum
        alpha: positive float of maximum step size when using momentum
    Returns:
        steps: n by p array of p steps to get from x0 to a valid counterfactual
        cf: valid counterfactual (last entry in steps)
    """
    steps = np.zeros((1, 2))
    # set up tree for k nearest neighbours
    tree = spatial.KDTree(list(zip(data[:, 0], data[:, 1])))

    # find closes k points to x0
    close = tree.query(x0, k=k, p=2)[1]

    # find probabilities of closest points
    vals = classifier.predict_proba(tree.data[close])[:, 1]

    # save best move and delete from tree and rebuild
    indx = np.argmax(vals)
    x_hat = tree.data[close[indx]]
    steps[0] = np.array(tree.data[close[indx]])
    cf = steps[0]
    temp = np.delete(tree.data, close[indx], 0)
    tree2 = spatial.KDTree(data)

    # repeat until valid counterfactual is found
    i = 0
    while classifier.predict_proba([x_hat])[0, 1] < thresh:
        # find closes k points to x0
        nei = tree.query(steps[i], k=k, p=2)
        close = nei[1]

        # find weighted probabilities of closest points
        vals = (1 / (1 + nei[0])) * classifier.predict_proba(tree.data[close])[:, 1]

        # save best move and delete from tree and rebuild
        indx = np.argmax(vals)
        x_hat = tree.data[close[indx]]
        best_step = np.array(tree.data[close[indx]])

        if mom > 0:
            if i > mom:
                mom_dir = np.zeros(2)
                for j in range(i - mom, mom):
                    mom_dir += steps[j] - steps[j - 1]
            else:
                mom_dir = np.zeros(2)
                for j in range(i):
                    mom_dir += steps[j] - steps[j - 1]
            mom_dir = mom_dir / mom
            best_step = 0.5 * (steps[i] - best_step) + 0.5 * mom_dir
            best_step_len = np.linalg.norm(best_step, 2)
            if best_step_len > alpha:
                best_step = (best_step / best_step_len) * alpha
            best_step = x_hat + best_step

        steps = np.append(steps, [best_step], axis=0)
        temp = np.delete(tree.data, close[indx], 0)
        tree = spatial.KDTree(list(zip(temp[:, 0], temp[:, 1])))

        cf = best_step
        x_hat = best_step
        i += 1
    return steps, cf


def best_path(x0, cf, data, dist, classifier, kde, thresh, early):
    """
    Find best path through data from x0 to counterfactual via query balls of radius dist
    Args:
        x0: n array starting point
        cf: n array counterfactual point
        data: all tranversible data n by m array
        dist: maximum distance we can move in a single step

    Returns: n by p array of p steps to get from x0 to a valid counterfactual
    """
    xt = x0
    steps = np.zeros((1, 2))
    steps[0] = x0
    # set up tree for k nearest neighbours
    tree = spatial.KDTree(list(zip(data[:, 0], data[:, 1])))
    i = 1
    while not numpy.array_equiv(xt, cf):
        # check if current point actually meets criteria
        if classifier.predict_proba([xt])[0, 1] >= thresh and early:
            print('Better solution located en route')
            break
        # get vector of the best direction of travel
        dir = xt - cf
        dir_len = np.linalg.norm(dir, ord=2)
        # find points within dist of x0
        indx = tree.query_ball_point(xt, dist, p=2)

        # find viable point that is along the path of best direction
        dot = -np.inf
        for j in indx:
            xi = tree.data[j]
            v = xt - xi
            v_len = np.linalg.norm(v, ord=2)
            vdir_len = np.linalg.norm(cf - xi, ord=2)
            if v_len != 0:
                temp = (((1 + (np.dot(dir, v) / (dir_len * v_len))) / 2) * kde.score([(xi + xt) / 2])) / dir_len
                if temp > dot:
                    dot = temp
                    best = j
                    """prob = kde.score([xi])
                    if prob > thresh_p:
                        dot = temp
                        best = j"""

        # if we have nowhere to go and we are at the beginning, terminate
        if len(indx) == 0 and numpy.array_equiv(xt, x0):
            print('No CF path found for sphere size {}'.format(dist))
            break

        if len(indx) == 0:
            xt = x0
            steps = np.zeros((1, 2))
            steps[0] = x0

        # edit tree and save step
        else:
            xt = tree.data[best]
            best_step = np.array(tree.data[best])
            steps = np.append(steps, [best_step], axis=0)
            temp = np.delete(tree.data, best, 0)
            tree = spatial.KDTree(list(zip(temp[:, 0], temp[:, 1])))

    return steps


def create_graph(X, kde, tol=0, density=10):
    G = nx.Graph()
    for i in range(len(X)):
        G.add_node(i)
    for i in range(len(X)):
        for j in range(len(X)):
            if np.linalg.norm(X[i] - X[j]) > 0:
                samples = np.array([numpy.linspace(X[i][0], X[j][0], density + 1),
                           numpy.linspace(X[i][1], X[j][1], density + 1)]).T
                score = kde.score_samples(samples)
                if all(k >= tol for k in score):
                    G.add_edge(i, j, weight=np.linalg.norm(X[i] - X[j], ord=2) * score / (density + 1))
    return G


def shortest_path(G):
    path = nx.shortest_path(G, source=0, target=int(G.number_of_nodes() - 1))

    return path
