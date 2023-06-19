import numpy as np
import funcs.funcs as funcs
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier as gpc
import time
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# set parameters
# number of nearest neighbours
k = 5
# threshold such that f(x)>thresh to be valid counterfactual
thresh = 0.85
# maximum step size allowed between data points to be labelled as feasible
sphere = 0.1
# starting point
pnt = np.array([0.1, 0.9])
# seed for random generator
seed = 10
# number of samples
samples = 250
# noise in data (1 + noise standard deviation if using blob)
noise = 0.2
# dataset generator
generator = funcs.create_two_moons

data = generator(samples, seed, noise)

X = data.iloc[:, :2]
y = data.y
start_time = time.time()
clf = gpc(1.0*RBF(1.0)).fit(X, y)
print('Time to train model: {} seconds'.format(np.round(time.time() - start_time, 2)))
print('Accuracy: {}'.format(clf.score(X, y)))
predictions = clf.predict_proba(X)

fig, ax = plt.subplots(figsize=(7,6))


ax = funcs.plot_dataset(ax, data)

X1 = data[["x1", "x2"]]

ax = funcs.plot_decision_boundary(ax, X1, clf)

Xs = np.asarray((X1))
start_time = time.time()
steps, cf = funcs.find_cf(pnt, Xs, clf, k=k, thresh=thresh)
print('Time to find counterfactual: {} seconds'.format(np.round(time.time() - start_time, 2)))

steps = np.insert(steps, 0, [pnt], axis=0)

start_time = time.time()
best_steps = funcs.best_path(pnt, cf, Xs, sphere, clf, thresh)
print('Time to find best path: {} seconds'.format(np.round(time.time() - start_time, 2)))


plt.plot(steps[:,0], steps[:,1], '-k', label='Searched Path')
plt.plot(best_steps[:,0], best_steps[:,1], '-g', label='Best Path')
plt.plot(pnt[0], pnt[1], 'go', label='$x_0$')
plt.xlim([0,1])
plt.ylim([0,1])

# plt.title('Best Path to Counterfactual using Hyper-sphere of size {} \nfrom {} to {}'.format(sphere, pnt, np.round(cf,2)))
# plt.title('Finding Counterfactual Using {} Neighbours \nand {} Threshold from {}'.format(k, thresh, pnt))
# legend = plt.legend(frameon=True, loc='lower left', fancybox=True, fontsize=12, framealpha=0.25)
plt.savefig('plots/notes/moons_cf_neighbours_{}_from_({}, {})_spheresize_{}.pdf'.format(k, pnt[0], pnt[1], sphere), format='pdf')

plt.show()
