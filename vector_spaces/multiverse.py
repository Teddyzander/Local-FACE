import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patheffects as pe
import numpy as np
import func.data_gen as func


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    np.random.seed(1234)
    x = np.linspace(-4, 4, 500)
    y = np.linspace(-4, 4, 500)
    Z = np.zeros((len(x), len(y)))
    X, Y = np.meshgrid(x, y)
    for i in range(0, len(x)):
        for j in range(0, len(y)):
            Z[i, j] = func.surface_1(X[0, i], Y[j, 0])

    origin = 'lower'

    C1 = 1.4
    C2 = 0.2
    font_size = 25

    fig1, ax2 = plt.subplots(layout='constrained')
    CS = ax2.contourf(X, Y, Z, 5, cmap=mpl.colormaps['seismic'], origin=origin, alpha=0.6, antialiased=True)
    ax2.contour(X, Y, Z, levels=[0.7], linestyles='dashed', colors='red', linewidths=1.5, alpha=1)
    # CS2 = ax2.contour(CS, levels=[0], colors='k', origin=origin, linewidths=0.5, linestyles='dashed', alpha=0.5)
    ax2.xaxis.set_tick_params(labelbottom=False)
    ax2.yaxis.set_tick_params(labelleft=False)

    # Hide X and Y axes tick marks
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Add the contour line levels to the colorbar

    plt.annotate("", xy=(2.25, 0.05), xytext=(C1, C2),
                 arrowprops=dict(arrowstyle="->", lw=1.5, color='black', alpha=.7))
    plt.text(2.25 + 0.15, 0.05, 'B', va='center', ha='center',
             rotation='horizontal', fontsize=font_size, color='white', alpha=.7,
             path_effects=[pe.withStroke(linewidth=4, foreground="black")])
    plt.annotate("", xy=(2.45, -0.25), xytext=(C1, C2),
                 arrowprops=dict(arrowstyle="->", lw=1.5, color='black', alpha=.7))
    plt.text(2.45 + 0.15, -0.25, 'A', va='center', ha='center',
             rotation='horizontal', fontsize=font_size, color='white', alpha=.7,
             path_effects=[pe.withStroke(linewidth=4, foreground="black")])
    plt.annotate("", xy=(C1+0.05, 1.3), xytext=(C1, C2),
                 arrowprops=dict(arrowstyle="->", lw=1.5, color='black', alpha=.7))
    plt.annotate("", xy=(2.45, 1.35), xytext=(C1+0.05, 1.3),
                 arrowprops=dict(arrowstyle="->", lw=1.5, color='black', alpha=.7))
    plt.text(2.45 + 0.15, 1.35, 'C', va='center', ha='center',
             rotation='horizontal', fontsize=font_size, color='white', alpha=.7,
             path_effects=[pe.withStroke(linewidth=4, foreground="black")])
    plt.annotate("", xy=(0.8, -3.4), xytext=(C1, C2),
                 arrowprops=dict(arrowstyle="->", lw=1.5, color='black', alpha=.7))
    plt.text(0.8, -3.6, 'H', va='center', ha='center',
             rotation='horizontal', fontsize=font_size, color='white', alpha=.7,
             path_effects=[pe.withStroke(linewidth=4, foreground="black")])
    plt.annotate("", xy=(0, -1.5), xytext=(C1, C2),
                 arrowprops=dict(arrowstyle="->", lw=1.5, color='black', alpha=.7))
    plt.annotate("", xy=(-0.5, -3), xytext=(0, -1.5),
                 arrowprops=dict(arrowstyle="->", lw=1.5, color='black', alpha=.7))
    plt.text(-0.5, -3.2, 'G', va='center', ha='center',
             rotation='horizontal', fontsize=font_size, color='white', alpha=.7,
             path_effects=[pe.withStroke(linewidth=4, foreground="black")])
    plt.annotate("", xy=(-1.4, -3), xytext=(0, -1.5),
                 arrowprops=dict(arrowstyle="->", lw=1.5, color='black', alpha=.7))
    plt.annotate("", xy=(-1.45, -3.5), xytext=(-1.4, -3),
                 arrowprops=dict(arrowstyle="->", lw=1.5, color='black', alpha=.7))
    plt.text(-1.45, -3.7, 'F', va='center', ha='center',
             rotation='horizontal', fontsize=font_size, color='white', alpha=.7,
             path_effects=[pe.withStroke(linewidth=4, foreground="black")])
    plt.annotate("", xy=(-0.1, -1.2), xytext=(C1, C2),
                 arrowprops=dict(arrowstyle="->", lw=2, color='yellow', alpha=0.7))
    plt.annotate("", xy=(-1.2, -2), xytext=(-0.1, -1.2),
                 arrowprops=dict(arrowstyle="->", lw=2, color='yellow', alpha=0.7))
    plt.annotate("", xy=(-1.4, -3), xytext=(-1.2, -2),
                 arrowprops=dict(arrowstyle="->", lw=2, color='yellow', alpha=0.7))
    plt.annotate("", xy=(-1.6, -1.68), xytext=(-1.2, -2),
                 arrowprops=dict(arrowstyle="->", lw=2, color='yellow', alpha=0.7))
    plt.text(-1.2-0.45, -1.75-0.7, r'E$_2$', va='center', ha='center',
             rotation='horizontal', fontsize=font_size, color='yellow', alpha=.7,
             path_effects=[pe.withStroke(linewidth=4, foreground="black")])
    plt.text(-1.2-0.6, -2 + 0.7, r'E$_1$', va='center', ha='center',
             rotation='horizontal', fontsize=font_size, color='yellow', alpha=.7,
             path_effects=[pe.withStroke(linewidth=4, foreground="black")])
    plt.annotate("", xy=(0, -1.5), xytext=(-0.1, -1.2),
                 arrowprops=dict(arrowstyle="->", lw=2, color='yellow', alpha=0.7))
    plt.text(0+0.4, -1.5-0.25, r'E$_3$', va='center', ha='center',
             rotation='horizontal', fontsize=font_size, color='yellow', alpha=.7,
             path_effects=[pe.withStroke(linewidth=4, foreground="black")])
    plt.annotate("", xy=(-1.8, -1.8), xytext=(C1, C2),
                 arrowprops=dict(arrowstyle="->", lw=2, color='black', alpha=0.7))
    plt.annotate("", xy=(-3.25, -2), xytext=(-1.8, -1.8),
                 arrowprops=dict(arrowstyle="->", lw=2, color='black', alpha=0.7))
    plt.text(-3.35, -2, 'D', va='center', ha='center',
             rotation='horizontal', fontsize=font_size, color='white', alpha=.7,
             path_effects=[pe.withStroke(linewidth=4, foreground="black")])

    func.gen_points(0, 0, -1.8, 1.5, -1.7, 1.7, 200)
    func.gen_points(-3, 2, -0.9, 6, -1, 1.5, 100)
    func.gen_points(-4, -4, 0, 3, 0, 1, 100)
    func.gen_points(-1, -3, -0.5, 0.5, -0.5, 1.5, 30)
    func.gen_points(-4, -2, 0, 2, -0.5, 1, 30)
    func.gen_points(2.45, 1.35, -1, 1, -0.25, 0.15, 50)
    func.gen_points(0.8 + 0.15, -3.4 + 1, -0.5, 0.5, -1.5, 1.5, 20)
    func.gen_points(-0.5 + 0.25, -3 + 1, -0.5, 0.5, -1.5, 1.5, 20)
    func.gen_points(2.45 + 0.15, -0.25, -0.15, 0.25, -0.45, 0.35, 20)
    for i in range(0, 20):
        pert1 = np.random.uniform(-0.35, 0.35)
        pert2 = np.random.uniform(-0.35, 0.35)
        colour = np.random.choice(['red', 'blue'], p=[0.1, 0.9])
        plt.plot([0.8 + pert1], [-3.4 + pert2], marker="o", markersize=4, markeredgecolor='black',
                 markerfacecolor=colour,
                 alpha=0.25)
    for i in range(0, 20):
        pert1 = np.random.uniform(-0.35, 0.35)
        pert2 = np.random.uniform(-0.35, 0.35)
        colour = np.random.choice(['red', 'blue'], p=[0.2, 0.8])
        plt.plot([-0.5 + pert1], [-3 + pert2], marker="o", markersize=4, markeredgecolor='black', markerfacecolor=colour,
                 alpha=0.25)
    ax2.axis('off')
    ax2.plot([C1], [C2], marker="X", markersize=8, markeredgecolor="k", markerfacecolor="white", alpha=1, zorder=50)
    plt.savefig('plots_no_distance_weight/CF_paths_no_bound.pdf', bbox_inches='tight')

    plt.show()
