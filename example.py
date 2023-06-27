import local_face.helpers.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
from local_face.local_face import *
from local_face.helpers.plotters import *
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KernelDensity
import warnings

warnings.filterwarnings("ignore")

# the factual point
factual = np.array([0.04, 0.375])
# parameters for locating counterfactual and path
k = 5
thresh = 0.95
dist = 0.1
# parameters for data generation
seed = 10
samples = 300
noise = 0.15
# parameters for model creation
band_width = 0.025

# create data
generator = datasets.create_two_moons  # get data generator
data = generator(samples, seed, noise)  # generate data
X = data.iloc[:, :2]  # ensure data is 2d
y = data.y  # get target

# train model and density estimator using training data
model = MLPClassifier(hidden_layer_sizes=(14, 11, 8), random_state=1, max_iter=700).fit(X, y)
dense = KernelDensity(kernel='gaussian', bandwidth=band_width).fit(X)
X1 = data[["x1", "x2"]]

# initialise Local-FACE model
face = LocalFace(X1, model, dense)
# find a counterfactual
steps, cf = face.find_cf(factual, k=k, thresh=thresh, mom=3)
# generate graph nodes through data from factual to counterfactual
best_steps = face.generate_graph(factual, cf, dist, thresh, early=True)
# create edges between viable points and calculate the weights
G = face.create_graph(0, 100, method='strict')
# calculate shortest path through G from factual to counterfactual
shortest_path = face.shortest_path()

# plotting procedure
# plot the data, the decision function, the searched path
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 5), sharey=True)
# finding counterfactual
ax[0] = plot_dataset(ax[0], data)
ax[0] = plot_decision_boundary(ax[0], X1, model)
ax[0].plot(factual[0], factual[1], 'go', label='factual')
ax[0].plot(steps[:, 0], steps[:, 1], '-y', label='Explore', linewidth=3)
ax[0].set_xlim([0, 0.5])
ax[0].set_ylim([0.2, 0.85])
ax[0].title.set_text('Explore')

ax[1] = plot_density(ax[1], X1, dense, levels=5)
ax[1].set_xlim([0, 0.5])
ax[1].set_ylim([0.2, 0.85])
ax[1].plot(best_steps[:, 0], best_steps[:, 1], '-k', label='Exploit', linewidth=2.5)
ax[1].plot(factual[0], factual[1], 'go', label='factual')
ax[1].plot(best_steps[-1, 0], best_steps[-1, 1], '*b', label='$x^\prime$', markersize=12, alpha=0.7, markeredgecolor='white')
ax[1].title.set_text('Exploit')

ax[2] = plot_density(ax[2], X1, dense, levels=5)
ax[2].set_xlim([0, 0.5])
ax[2].set_ylim([0.2, 0.85])
ax[2].plot(best_steps[shortest_path, 0], best_steps[shortest_path, 1], '-g', label='Enhance', linewidth=2)
ax[2].plot(factual[0], factual[1], 'go', label='factual')
ax[2].plot(best_steps[-1, 0], best_steps[-1, 1], '*b', label='$x^\prime$', markersize=12, alpha=0.7, markeredgecolor='white')
ax[2].plot([factual[0], best_steps[-1, 0]],
           [factual[1], best_steps[-1, 1]], '-r', linewidth=2)
ax[2].plot([best_steps[shortest_path[1], 0], best_steps[shortest_path[3], 0]],
           [best_steps[shortest_path[1], 1], best_steps[shortest_path[3], 1]], '-r', label='illegal edge', linewidth=2)

ax[2].title.set_text('Enhance')
ax[2].annotate("", xy=(0.21, 0.375), xytext=(0.21, 0.45), arrowprops=dict(arrowstyle="->", lw=2.5, color='black', alpha=.7))
ax[2].text(0.21, 0.475, 'not allowed', va='center', ha='center',
           rotation='horizontal', fontsize=12, color='black', alpha=.7)
ax[2].annotate("", xy=(0.175, 0.67), xytext=(0.21, 0.5), arrowprops=dict(arrowstyle="->", lw=2.5, color='black', alpha=.7))
ax[2].annotate("", xy=(0.28, 0.63), xytext=(0.21, 0.5), arrowprops=dict(arrowstyle="->", lw=2.5, color='black', alpha=.7))

fig.tight_layout()
plt.savefig("paper_figure_strict.pdf", format='pdf')
plt.show()

print('stop')
