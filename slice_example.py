import matplotlib.pyplot as plt
import numpy as np

import local_face.helpers.datasets as datasets
from local_face.local_face import *
from local_face.helpers.plotters import *
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import time
import warnings
import scipy.optimize as optimize
import scipy.integrate as integrate

warnings.filterwarnings("ignore")


def dist(a, b, eps):
    b = b.reshape(a.shape)
    ans = np.linalg.norm(a - b)
    return ans


def obj(x_prime, x, y_prime, model, dense, eps=0.1, prior=10, shift=0.005):
    pred = model.predict_proba([x_prime])[0, 1]
    den = np.exp(dense.score([x_prime]))
    temp = np.array([np.linspace(x.reshape(x_prime.shape)[0], x_prime[0], 20),
                     np.linspace(x.reshape(x_prime.shape)[1], x_prime[1], 20)]).T
    h = np.linalg.norm(x.reshape(x_prime.shape) - x_prime) / 20
    """for i in range(0, len(temp)):
        den += h * np.exp(dense.score([temp[i]]))
    # den = dense.score([x.reshape(x_prime.shape)])
    # den = integrate.nquad(dense.score, x.reshape(x_prime.shape), x_prime)
    if den > 1:
        den = 1
    if den < 0:
        den = 0"""
    length = dist(x, x_prime, eps)
    ans = (pred - y_prime) ** 2 + length**2 + (1 - den) ** 2
    return ans

def obj2(x_prime, x, y_prime, model, eps=9999999):
    pred = model.predict_proba([x_prime])[0, 1]
    length = dist(x, x_prime, eps)
    ans = (pred - y_prime) ** 2 + length ** 2
    return ans

def obj3(x_prime, x, y_prime, model, dense, eps=9999999):
    pred = model.predict_proba([x_prime])[0, 1]
    length = dist(x, x_prime, eps)
    den = np.exp(dense.score([x_prime]))
    ans = (pred - y_prime) ** 2 + length ** 2 + (1 - den) ** 2
    return ans



graph = False
# the factual point
factual = np.array([0.04, 0.375])
# parameters for locating counterfactual and path
k = 5
thresh = 0.95
# parameters for data generation
seed = 10
samples = 2000
noise = 0.15
# parameters for model creation
band_width = 0.01

# create data
generator = datasets.create_two_moons  # get data generator
data = generator(samples, seed, noise)  # generate data
X = data.iloc[:, :2]  # ensure data is 2d
y = data.y  # get target

# train model and density estimator using training data
model = MLPClassifier(hidden_layer_sizes=(14, 11, 6), random_state=1, max_iter=700).fit(X, y)
dense = KernelDensity(kernel='gaussian', bandwidth=band_width).fit(X)
bandwidths = np.logspace(-2, 0, 10)
kd_estimators = GridSearchCV(
    estimator = KernelDensity(kernel="gaussian", metric="l2"),
    param_grid = {"bandwidth": bandwidths}
)

kd_estimators.fit(X)
X1 = data[["x1", "x2"]]

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 12), sharex=True, sharey=True)
ax[0] = plot_dataset(ax[0], data)
ax[0] = plot_decision_boundary(ax[0], X1, model, alpha=0.6)
ax[1] = plot_density(ax[1], X1, dense, alpha=0.6, levels=40)

x0 = np.array([0.1, 0.3])
k=25
org = model.predict_proba([x0])[0, 1]
y_prime = 1
prior = 10000
for i in range(k):
    y = org + ((i+1) / k) * np.abs(org - y_prime)
    result = optimize.minimize(obj, x0.reshape(-1, 1),
                               args=(x0.reshape(-1, 1), y, model, kd_estimators, 0.01, 1), tol=0.00000001)

    ax[0].annotate('',
                  xytext=(x0[0], x0[1]),
                  xy=(result['x'][0], result['x'][1]),
                  arrowprops=dict(arrowstyle="->", color='green'),
                  size=12
                  )
    ax[1].annotate('',
                   xytext=(x0[0], x0[1]),
                   xy=(result['x'][0], result['x'][1]),
                   arrowprops=dict(arrowstyle="->", color='green'),
                   size=12
                   )
    x0 = result['x']
    prior = prior / 2
init_guess = result['x']
x0 = np.array([0.1, 0.3])
y = 1
result = optimize.minimize(obj2, init_guess.reshape(-1, 1),
                            args=(x0.reshape(-1, 1), y, model))
ax[0].annotate('',
                  xytext=(x0[0], x0[1]),
                  xy=(result['x'][0], result['x'][1]),
                  arrowprops=dict(arrowstyle="->", color='black'),
                  size=12
                  )
ax[1].annotate('',
                  xytext=(x0[0], x0[1]),
                  xy=(result['x'][0], result['x'][1]),
                  arrowprops=dict(arrowstyle="->", color='black'),
                  size=12
                  )

x0 = np.array([0.1, 0.3])
y = 1
result = optimize.minimize(obj3, init_guess.reshape(-1, 1),
                            args=(x0.reshape(-1, 1), y, model, dense))
ax[0].annotate('',
                  xytext=(x0[0], x0[1]),
                  xy=(result['x'][0], result['x'][1]),
                  arrowprops=dict(arrowstyle="->", color='purple'),
                  size=12
                  )
ax[1].annotate('',
                  xytext=(x0[0], x0[1]),
                  xy=(result['x'][0], result['x'][1]),
                  arrowprops=dict(arrowstyle="->", color='purple'),
                  size=12
                  )

ax[0].set_xlim([0, 0.5])
ax[0].set_ylim([0.2, 0.7])
ax[1].set_xlim([0, 0.5])
ax[1].set_ylim([0.2, 0.7])
fig.tight_layout(h_pad=0, w_pad=0)
ax[0].set_aspect('equal', adjustable='box')
ax[1].set_aspect('equal', adjustable='box')
plt.savefig("kis25.pdf", format="pdf")
plt.show()

print('stop')
