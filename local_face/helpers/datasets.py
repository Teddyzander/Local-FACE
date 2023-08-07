import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.datasets import make_moons, make_circles, make_blobs


def creat_custom(num=300, seed=10, noise=0.15):
    np.random.seed(seed + 1)
    X1, y1 = make_blobs(n_samples=int(num / 4), n_features=2, centers=[(0.8, -0.2)],
                        cluster_std=2 * noise)
    X2, y2 = make_moons(n_samples=num, noise=noise)
    # X3, y3 = make_blobs(n_samples=int(num / 50), n_features=2, centers=[(-0.3, 0), (-0.3, 0)], cluster_std=noise * 3)

    X = np.concatenate((X1, X2))  # , X3))
    y = np.concatenate((y1, y2))  # , y3))

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


def constrain_search(data, cons):
    """
    return the query data set only containing accessible data
    Args:
        data: original data set
        con: conditions for dropping a row in each feature

    Returns: accessible data
    """
    con_data = data.copy()
    if len(cons) != data.shape[1]:
        raise ValueError("Must pass a constraint for each feature")
    for i, con1 in enumerate(cons):
        if con1 is not None:
            for con2 in con1:
                condition = "con_data.drop(con_data[con_data.iloc[:, i]" + con2 + "].index, inplace=True)"
                exec(condition)

    return con_data
