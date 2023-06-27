import numpy as np
import funcs.funcs as funcs
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier as gpc
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KernelDensity
import time
import matplotlib.pyplot as plt
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

plotting = False
plot_decision = True
cf_prior = False
# set parameters
# early termination?
early = False
# pre-made counterfactual point
cf_p = np.array([0.9, 0.25])
# starting point
pnt = np.array([0.1, 0.6])
# minimum probability from KDE
prob = 0
# bandwidth for KDE
band_width=0.05
# number of nearest neighbours
k = 10
# threshold such that f(x)>thresh to be valid counterfactual
thresh = 0.95
# maximum step size allowed between data points to be labelled as feasible
sphere = 0.15
# seed for random generator
seed = 10
# number of samples
samples = 300
# noise in data (1 + noise standard deviation if using blob)
noise = 0.2
# dataset generator
generator = funcs.create_two_moons

data = generator(samples, seed, noise)

X = data.iloc[:, :2]
y = data.y
start_time = time.time()
# clf = gpc(1.0*RBF(1.0)).fit(X, y)
clf = MLPClassifier(hidden_layer_sizes=(15, 15, 15), random_state=1, max_iter=1000).fit(X, y)
kde = KernelDensity(kernel='gaussian', bandwidth=band_width).fit(X)
# clf = LogisticRegression(random_state=0).fit(X, y)
# clf = SVC(gamma='auto', probability=True, kernel='sigmoid', C=1).fit(X, y)
print('Time to train model: {} seconds'.format(np.round(time.time() - start_time, 2)))
print('Accuracy: {}'.format(clf.score(X, y)))
predictions = clf.predict_proba(X)

fig, ax = plt.subplots(figsize=(7, 6))

ax = funcs.plot_dataset(ax, data)

X1 = data[["x1", "x2"]]
if plot_decision:
    ax = funcs.plot_decision_boundary(ax, X1, clf, levels=20)
else:
    h = 0.01
    x1_min, x2_min = np.min(X, axis=0)
    x1_max, x2_max = np.max(X, axis=0)

    x1_cords, x2_cords = np.meshgrid(
        np.arange(x1_min, x1_max + h, h),
        np.arange(x2_min, x2_max + h, h)
    )
    new_X = np.c_[x1_cords.ravel(), x2_cords.ravel()]
    new_X_df = pd.DataFrame(new_X, columns=["x1", "x2"])
    score = kde.score_samples(new_X_df)
    score = score.reshape(x1_cords.shape)
    ax = plt.contourf(x1_cords, x2_cords, score)

Xs = np.asarray((X1))

# search for counterfactual (if not given)
if not cf_prior:
    start_time = time.time()
    steps, cf = funcs.find_cf_mom(pnt, Xs, clf, k=k, thresh=thresh, mom=3)
    # steps, cf = funcs.find_cf(pnt, Xs, clf, k=k, thresh=thresh)
    steps = np.insert(steps, 0, [pnt], axis=0)
    plt.plot(steps[:, 0], steps[:, 1], '-y', label='Searched Path', linewidth=3)
    print('Time to find counterfactual: {} seconds'.format(np.round(time.time() - start_time, 2)))

else:
    cf = cf_p

Xs = np.append(Xs, [cf], axis=0)

# find relevant data points
start_time = time.time()
best_steps = funcs.best_path(pnt, cf, Xs, sphere, clf, kde, thresh, early)
print('Time to find relevant nodes: {} seconds'.format(np.round(time.time() - start_time, 2)))
plt.plot(best_steps[:, 0], best_steps[:, 1], '-g', label='Best Path', linewidth=2)
plt.plot(cf[0], cf[1], '*b', label='$x^\prime$', markersize=12, alpha=0.7, markeredgecolor='white')
plt.plot(pnt[0], pnt[1], 'go', label='$x_0$')
plt.xlim([0, 1])
plt.ylim([0, 1])

# find best path through relevant data points
start_time = time.time()
graph = funcs.create_graph(best_steps, kde, prob)
shortest_path = funcs.shortest_path(graph)
print('Time to find best path: {} seconds'.format(np.round(time.time() - start_time, 2)))

plt.plot(best_steps[shortest_path, 0], best_steps[shortest_path, 1], '-k', label='Best Path', linewidth=1)
plt.xlim([0, 1])
plt.ylim([0, 1])
# plt.xlim(cf[0]-0.05, pnt[0]+0.05)
# plt.ylim(cf[1]-0.05, pnt[1]+0.05)

if plotting == True:
    plt.annotate("", xy=(steps[-1, 0] - (steps[-1, 0] - pnt[0]) * 0.7, steps[-1, 1] - (steps[-1, 1] - pnt[1]) * 0.7),
                 xytext=(pnt[0], pnt[1]),
                 arrowprops=dict(arrowstyle="->, head_width=0.3", lw=3, color='blue', alpha=0.9))
    circle1 = plt.Circle((pnt[0], pnt[1]), 0.1, color='k', clip_on=False, alpha=0.3)
    circle2 = plt.Circle((pnt[0], pnt[1]), 0.1 / 2, color='y', clip_on=False, alpha=0.25)
    circle3 = plt.Circle((best_steps[1, 0], best_steps[1, 1]), 0.1 / 2, color='b', clip_on=False, alpha=0.25)

    plt.xlim([pnt[0] - 0.12, pnt[0] + 0.12])
    plt.ylim([pnt[1] - 0.12, pnt[1] + 0.12])
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    ax.add_patch(circle3)
    plt.annotate("", xy=(pnt[0] + sphere / 1.46, pnt[1] + sphere / 1.46), horizontalalignment="center",
                 xytext=(pnt[0], pnt[1]),
                 arrowprops=dict(arrowstyle="<->, head_width=0.3", lw=3, color='k', alpha=0.9))
    plt.annotate("", xy=(pnt[0] + sphere / 1.1 / 2, pnt[1] + sphere / 2.2 / 2), horizontalalignment="center",
                 xytext=(pnt[0], pnt[1]),
                 arrowprops=dict(arrowstyle="<->, head_width=0.3", lw=3, color='y', alpha=0.9))
    plt.annotate("", xy=(pnt[0] - 0.1, pnt[1] - 0.03), horizontalalignment="center", xytext=(pnt[0], pnt[1]),
                 arrowprops=dict(arrowstyle="-", lw=3, color='g', alpha=0.9))
    plt.annotate("", xy=(pnt[0] - 0.025 * 1.09, pnt[1] - 0.09 * 1.09), horizontalalignment="center",
                 xytext=(pnt[0], pnt[1]),
                 arrowprops=dict(arrowstyle="-", lw=3, color='g', alpha=0.9))

# plt.title('Best Path to Counterfactual using Hyper-sphere of size {} \nfrom {} to {}'.format(sphere, pnt, np.round(cf,2)))
# plt.title('Finding Counterfactual Using {} Neighbours \nand {} Threshold from {}'.format(k, thresh, pnt))
# legend = plt.legend(frameon=True, loc='lower left', fancybox=True, fontsize=12, framealpha=0.25)
plt.savefig(
    'plots/test/adms_graph_dec_mlp_bandwidth{}_mom10_custom_cf_neighbours_{}_from_({}, {})_spheresize_{}_probthresh_{}.png'
    .format(band_width, k, pnt[0], pnt[1], sphere, prob),
    format='png')
plt.axis('off')
plt.show()



print('stop')

