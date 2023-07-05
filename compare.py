import local_face.helpers.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
from local_face.local_face import *
from local_face.helpers.plotters import *
from local_face.helpers.funcs import *
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KernelDensity
import time
import warnings

warnings.filterwarnings("ignore")

# parameters for locating counterfactual and path
k = 15
thresh = 0.95
dist = 0.05
line = 10
# parameters for data generation
seed = 10
samples = 1000
noise = 0.15
# parameters for model creation
band_width = 0.01

# create data
generator = datasets.creat_custom  # get data generator
data = generator(samples, seed, noise)  # generate data
X = data.iloc[:, :2]  # ensure data is 2d
y = data.y  # get target

# train model and density estimator using training data
model = MLPClassifier(hidden_layer_sizes=(14, 11, 8), random_state=1, max_iter=700).fit(X, y)
dense = KernelDensity(kernel='gaussian', bandwidth=band_width).fit(X)
X1 = data[["x1", "x2"]]

# the factual point
factual = 262
# the factual point
cf = 269

start_total = time.time()
# initialise Local-FACE model
face = LocalFace(X1, model, dense)

Xs = np.array(X1)
test = dense.score([Xs[factual, :]])
# generate graph nodes through data from factual to counterfactual
print(r'Creating graph G (Exploit)...')
start_exploit = time.time()
best_steps, G = face.generate_graph(Xs[factual, :], Xs[cf, :], k, thresh, test, line, early=False, method='strict')
# create edges between viable points and calculate the weights
print('Exploit time taken: {} seconds'.format(np.round(time.time() - start_exploit, 2)))
print('---------------------------------')
# calculate shortest path through G from factual to counterfactual
print(r"Finding shortest path from x to x' through G (Enhance)...")
start_enhance = time.time()
shortest_path = face.shortest_path()
print('Enhance time taken: {} seconds'.format(np.round(time.time() - start_enhance, 5)))
print('---------------------------------')
print('Total time taken for local-FACE: {} seconds'.format(np.round(time.time() - start_total, 2)))

print('---------------------------------')
start_total = time.time()
G1 = face_graph(Xs, dist, dense)
shortest_path1 = djik(G1, factual, cf)
print('Total time taken for FACE: {} seconds'.format(np.round(time.time() - start_total, 5)))


# plotting procedure
# plot the data, the decision function, the searched path
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 5), sharey=True)
# finding counterfactual

ax[0] = plot_decision_boundary(ax[0], X1, model, levels=5)
ax[0] = plot_dataset(ax[0], data, size=5)
ax[0] = plot_graph(ax[0], X1, model, best_steps, G, shortest_path)
ax[0].set_xlim([0, 1])
ax[0].set_ylim([0, 1])
ax[0].plot(best_steps[shortest_path, 0], best_steps[shortest_path, 1], '-g', linewidth=2)
ax[0].plot(Xs[shortest_path1, 0], Xs[shortest_path1, 1], '-b', label='FACE', linewidth=2)
ax[0].plot(Xs[factual, 0], Xs[factual, 1], 'go', label='x')
ax[0].plot(best_steps[-1, 0], best_steps[-1, 1], '*b', label='$x^\prime$', markersize=12, alpha=0.7, markeredgecolor='white')
ax[0].title.set_text('Model')
ax[0].legend(loc='lower left', fancybox=True, framealpha=0.2, prop={'size': 16})

ax[1] = plot_density(ax[1], X1, dense, levels=5)
ax[1] = plot_dataset(ax[1], data, size=5)
ax[1] = plot_graph(ax[1], X1, model, best_steps, G, shortest_path)
ax[1].set_xlim([0, 1])
ax[1].set_ylim([0, 1])
ax[1].plot(best_steps[shortest_path, 0], best_steps[shortest_path, 1], '-g', label='', linewidth=2)
ax[1].plot(Xs[shortest_path1, 0], Xs[shortest_path1, 1], '-b', label='FACE', linewidth=2)
ax[1].plot(Xs[factual, 0], Xs[factual, 1], 'go', label='factual')
ax[1].plot(best_steps[-1, 0], best_steps[-1, 1], '*b', label='$x^\prime$', markersize=12, alpha=0.7, markeredgecolor='white')

ax[1].title.set_text('Density')
fig.tight_layout()
plt.savefig('local_face/plots/comparison.png', format='png')
plt.show()

print('stop')
