import numpy
from sklearn.datasets import make_moons
from sklearn import manifold, preprocessing
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import spatial

seed = 10

def create_two_moons():
    K = 10
    n_samples = 300
    np.random.seed(seed+1)
    X, y = make_moons(n_samples=n_samples, noise=0.15)
    X = preprocessing.minmax_scale(X)

    # X = np.concatenate((X, np.array([[1.70, 0.45],
    #                                 [-0.55, 0.55],
    #                                 [0.50, 0.40]])))
    # y = np.concatenate((y, np.array([1, 1, 1])))
    df = pd.DataFrame(preprocessing.minmax_scale(X), columns=["x1", "x2"])
    df["y"] = y
    return df


def plot_dataset(ax, df):
    # Plot the dataset.
    # dots_color_mapping = mpl.colors.ListedColormap(["#7B90D2", "#FAD689"])
    dots_color_mapping = mpl.colors.ListedColormap(["#ff0040", "#0000cc"])
    # dots_color_mapping =["b"]*num_points_top + ["r"]*num_points_left + ["#FAD689"]*num_points_middle + ["sienna"]*num_points_right

    ax.scatter(df.x1, df.x2, c=df.y,
               cmap=dots_color_mapping, s=25,
               #    edgecolors = 'black',
               zorder=1)

    ax.grid(color="grey", linestyle="--", linewidth=0.5, alpha=0.75)

    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_ylabel(r"$x_2~\longrightarrow$", fontsize=24)
    ax.set_xlabel(r"$x_1~\longleftrightarrow$", fontsize=24)
    # ax.set_ylabel(r"$x_2~\longleftrightarrow$", fontsize=24)

    return ax

def plot_decision_boundary(ax, X_scaled, predictor, color_bar=True):
    h = 0.01
    x1_min, x2_min = np.min(X_scaled, axis=0)
    x1_max, x2_max = np.max(X_scaled, axis=0)

    x1_cords, x2_cords = np.meshgrid(
        np.arange(x1_min, x1_max, h),
        np.arange(x2_min, x2_max, h)
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
    steps = np.zeros((1,2))
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

    #repeat until valid counterfactual is found
    i = 0
    while classifier.predict_proba([x_hat])[0, 1] < thresh:
        # find closes k points to x0
        nei = tree.query(steps[i], k=k, p=2)
        close = nei[1]

        # find probabilities of closest points
        vals = (1/nei[0]) * classifier.predict_proba(tree.data[close])[:, 1]

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

def best_path(x0, cf, data, dist, classifier, thresh):
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
        if classifier.predict_proba([xt])[0, 1] >= thresh:
            print('Better solution located en route')
            break
        # get vector of the best direction of travel
        dir = xt - cf
        dir_len = np.linalg.norm(dir, ord=2)
        # find points within dist of x0
        indx = tree.query_ball_point(xt, dist, p=2)

        # find viable point that is along the path of best direction
        dot = -2
        for j in indx:
            xi = tree.data[j]
            v = xt - xi
            v_len = np.linalg.norm(v, ord=2)
            if v_len != 0:
                temp = np.dot(dir, v) / (dir_len * v_len)
                if temp > dot:
                    dot = temp
                    best = j


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