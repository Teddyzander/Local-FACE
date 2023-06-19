import numpy as np
import funcs.funcs as funcs
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier as gpc
from scipy import spatial
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

k = 5
thresh = 0.85
sphere = 0.06
pnt = np.array([0.6, 0.85])

data = funcs.create_two_moons()

X = data.iloc[:, :2]
y = data.y
clf = gpc(1.0*RBF(1.0)).fit(X, y)
print('Accuracy: {}'.format(clf.score(X, y)))
predictions = clf.predict_proba(X)

fig, ax = plt.subplots(figsize=(7,6))


ax = funcs.plot_dataset(ax, data)

X1 = data[["x1", "x2"]]

ax = funcs.plot_decision_boundary(ax, X1, clf)

Xs = np.asarray((X1))
steps, cf = funcs.find_cf(pnt, Xs, clf, k=k, thresh=thresh)
steps = np.insert(steps, 0, [pnt], axis=0)

best_steps = funcs.best_path(pnt, cf, Xs, sphere, clf, thresh)

plt.plot(steps[:,0], steps[:,1], '-k', label='Searched Path')
plt.plot(best_steps[:,0], best_steps[:,1], '-g', label='Best Path')
plt.plot(pnt[0], pnt[1], 'go')

plt.title('Best Path to Counterfactual Using Hyper-sphere of size {} \nfrom {} to {}'.format(sphere, pnt, np.round(cf,2)))
# plt.title('Finding Counterfactual Using {} Neighbours \nand {} Threshold from {}'.format(k, thresh, pnt))
legend = plt.legend(frameon=True, loc='lower left', fancybox=True, fontsize=12, framealpha=0.25)
plt.savefig('plots/best_path/cf_neighbours_{}_from_({}, {})_spheresize_{}.png'.format(k, pnt[0], pnt[1], sphere), format='png')

plt.show()
