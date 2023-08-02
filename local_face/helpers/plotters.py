import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_dataset(ax, df, size=20):
    """
    plot the data set
    Args:
        ax: axis object to plot on
        df: data

    Returns:
        axis with the data plotted
    """
    dots_color_mapping = mpl.colors.ListedColormap(["#ff0040", "#0000cc"])

    ax.scatter(df.x1, df.x2, c=df.y,
               cmap=dots_color_mapping, s=size,
               #    edgecolors = 'black',
               zorder=1000)

    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_ylabel(r"$x_2$", fontsize=14)
    ax.set_xlabel(r"$x_1$", fontsize=14)

    return ax


def plot_decision_boundary(ax, X_scaled, predictor, color_bar=False, levels=20, alpha=0.8):
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
        levels=levels,
        cmap=plt.cm.RdBu,
        alpha=alpha,
        zorder=0
    )
    if color_bar:
        cbar = plt.colorbar(contour, ax=ax, fraction=0.1)
        cbar.ax.tick_params(labelsize=20)
    return ax


def plot_density(ax, X_scaled, dense, color_bar=False, levels=20, alpha=0.8, over=False):
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
        score = dense.score_samples(new_X_df)
        score = score.reshape(x1_cords.shape)
        return score

    height_values = predict_func(new_X_df)
    height_values = height_values.reshape(x1_cords.shape)
    if over:
        contour = ax.contourf(
            x1_cords,
            x2_cords,
            height_values,
            levels=levels,
            cmap=plt.cm.gray,
            alpha=alpha,
            zorder=0,
        )
        ax.contour(
            x1_cords,
            x2_cords,
            height_values,
            levels=levels,
            zorder=0
        )
    else:
        contour = ax.contourf(
            x1_cords,
            x2_cords,
            height_values,
            levels=levels,
            cmap=plt.cm.summer,
            alpha=alpha,
            zorder=0
        )
    if color_bar:
        cbar = plt.colorbar(contour, ax=ax, fraction=0.1)
        cbar.ax.tick_params(labelsize=20)
    return ax


def plot_graph(ax, data, model, steps, G, shortest):
    nodes = G.number_of_nodes()
    for i in range(0, nodes):
        edges = G.edges(i)
        for j in edges:
            w = G[i][j[1]]["weight"]
            ax.plot([steps[i, 0], steps[j[1], 0]],
                    [steps[i, 1], steps[j[1], 1]], 'ko', linestyle="--", linewidth = 0.1, alpha=0.25)

    ax.plot(steps[shortest, 0], steps[shortest, 1], '-g', label='Local-FACE', linewidth=3, alpha=1)
    return ax
