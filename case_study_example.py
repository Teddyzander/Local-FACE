from data.mimic_preprocessing import *
from local_face.local_face import *
from local_face.helpers.plotters import *
from local_face.helpers.datasets import *
from sklearn.neighbors import KernelDensity
from sklearn.metrics import roc_auc_score
import time
import warnings
import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

graph = False  # plotting
scale = True  # stanardised input data

# parameters for locating counterfactual and path
k = 20
thresh = 0.75
dist = 0.1
seed = 41
method_type = 'strict'
prob_dense = 0.1

# parameters for density model creation
band_width = 0.01

features = ['airway', 'fio2', 'spo2_min',
            'hco3', 'resp_min', 'resp_max',
            'bp_min', 'hr_min', 'hr_max', 'pain',
            'gcs_min', 'temp_min', 'temp_max',
            'haemoglobin', 'k', 'na', 'creatinine', 'bun',
            'bmi', 'los', 'age', 'sex'
            ]

# import rfd testset data
X_train, y_train = load_dataset('mimic',
                                features,
                                scale=scale,
                                test=False
                                )
X_test, y_test = load_dataset('mimic',
                              features,
                              scale=scale
                              )

all_columns = X_train.columns

# trained random forest model
if scale:
    model = pickle.load(
        open(
            'rfd_model/results/rf_standardised_all.pickle',  # updated model including GICU
            # 'rfd_model/results/legacy/rf_standardised_mimic_only.pickle', # previous model just trained on MIMIC
            'rb'
        ))
else:
    model = pickle.load(open('rfd_model/results/rf.pickle', 'rb'))

# sanity check that AUC performance matches expectation
result = roc_auc_score(
    y_test, model.predict_proba(X_test.to_numpy())[:, 1])
print(f'Test set AUC performance {result:.3f}')

# ---- select factual ----
# just randomly from all cases not rfd
factual = np.array(X_test.loc[y_test == 0].sample(n=1, random_state=seed))[0]

# or instead, for false negatives
# factual = factual_selector('mimic', features, model,
#                           seed=seed, scale=scale, alignment='fn')

# print(factual)

# reverse scaling

# print(reverse_scaling('mimic', features, factual))

# train density estimator using training data
X_train_den = np.array(X_train)
dense = KernelDensity(kernel='gaussian', bandwidth=band_width).fit(X_train_den)

# optionally constrain available datapoints to variables of interest
upper_age = ">"+str(factual[20] + 0.05)
lower_age = "<"+str(factual[20] - 0.05)
constraints = [
    [],  # 'airway'
    [],  # 'fio2',
    [],  # 'spo2_min',
    [],  # 'hco3',
    [],  # 'resp_min',
    [],  # 'resp_max',
    [],  # 'bp_min',
    [],  # 'hr_min', ["<45"], [">55"]
    [],  # 'hr_max',
    [],  # 'pain',
    [],  # 'gcs_min',
    [],  # 'temp_min',
    [],  # 'temp_max',
    [],  # 'haemoglobin',
    [],  # 'k',
    [],  # 'na',
    [],  # 'creatinine',
    [],  # 'bun',
    [],  # 'bmi',
    [],  # 'los',
    [upper_age, lower_age],  # 'age',
    [],  # 'sex'
]

X_train = constrain_search(X_train, constraints)
X_test = constrain_search(X_test, constraints)


# y = np.ravel(y)
X_train = np.array(X_train)


start_total = time.time()

# initialise Local-FACE model
face = LocalFace(X_train, model, dense)  # constrained

# find a counterfactual
print(r"Finding counterfactual x' (explore)...")
start_explore = time.time()
steps, cf = face.find_cf(factual, k=k, thresh=thresh, mom=0)
print('Explore time taken: {} seconds'.format(
    np.round(time.time() - start_explore, 2)))
overall_recourse = pd.DataFrame([factual - cf], columns=features)
print('---------------------------------')

# generate graph nodes through data from factual to counterfactual
print(r'Creating graph G (Exploit)...')
start_exploit = time.time()
best_steps, G = face.generate_graph(
    factual, cf, k, thresh, prob_dense, 10, early=True, method=method_type)
# create edges between viable points and calculate the weights
"""prob = face.dense.score([factual])
G = face.create_edges(1, 10, method='strict')"""
print('Exploit time taken: {} seconds'.format(
    np.round(time.time() - start_exploit, 2)))
print('---------------------------------')


# calculate shortest path through G from factual to counterfactual
print(r"Finding shortest path from x to x' through G (Enhance)...")
start_enhance = time.time()
shortest_path = face.shortest_path(method=method_type)
print('Enhance time taken: {} seconds'.format(
    np.round(time.time() - start_enhance, 2)))
print('---------------------------------')
print('Total time taken: {} seconds'.format(
    np.round(time.time() - start_total, 2)))


# Create dataframe of path
path_df = pd.DataFrame(best_steps, columns=features)

# Find most relevant / changing / volatile features to display
# Identify features with largest std
# Extract top n
top_n_features = 3
features_std = path_df.std().sort_values(ascending=False)[0:top_n_features]
print(f'Top {top_n_features} features which vary: \n', features_std)
# Then extract features to list
volatile_feats = features_std.index.values

# Then extract these relevant columns from the path dataframe (combined)
volatile_combined = path_df[volatile_feats]

print('Top features to track overall: \n', volatile_combined)

print('Top features to track between examples: \n')
for i in range(1, len(path_df.index)):
    inst = path_df.iloc[i-1:i+1]
    temp = inst.std().sort_values(ascending=False)[0:top_n_features]
    volatile_feats = temp.index.values

    # Then extract these relevant columns from the path dataframe (combined)
    volatile_combined = temp[volatile_feats]
    print(volatile_combined)

# Plot probabilites over the path
probs = model.predict_proba(best_steps)
rfd_probs = [item[1] for item in probs]
plt.plot(rfd_probs,
         label=('Probability Ready for Discharge')
         )
plt.legend()
plt.show()

# plotting procedure
# plot the data, the decision function, the searched path
if graph:
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 5), sharey=True)
    # finding counterfactual
    ax[0] = plot_dataset(ax[0], data)
    ax[0] = plot_decision_boundary(ax[0], X1, model)
    ax[0].plot(factual[0], factual[1], 'go', label='factual')
    ax[0].plot(steps[:, 0], steps[:, 1], '-y', label='Explore', linewidth=3)
    ax[0].set_xlim([0, 0.5])
    ax[0].set_ylim([0.2, 0.9])
    ax[0].title.set_text('Explore')

    ax[1] = plot_density(ax[1], X1, dense, levels=5)
    ax[1] = plot_graph(ax[1], X1, model, best_steps, G, shortest_path)
    ax[1].set_xlim([0, 0.5])
    ax[1].set_ylim([0.2, 0.9])
    # ax[1].plot(best_steps[:, 0], best_steps[:, 1], '-k', label='Exploit', linewidth=2.5)
    # ax[1].plot(factual[0], factual[1], 'go', label='factual')
    ax[1].plot(best_steps[-1, 0], best_steps[-1, 1], '*b',
               label='$x^\prime$', markersize=12, alpha=0.7, markeredgecolor='white')
    ax[1].title.set_text('Exploit')

    ax[2] = plot_density(ax[2], X1, dense, levels=5, alpha=0.4, over=True)
    ax[2] = plot_decision_boundary(ax[2], X1, model, alpha=0.6)
    ax[2].set_xlim([0, 0.5])
    ax[2].set_ylim([0.2, 0.9])
    ax[2].plot(best_steps[shortest_path, 0],
               best_steps[shortest_path, 1], '-g', label='Enhance', linewidth=2)
    ax[2].plot(factual[0], factual[1], 'go', label='factual')
    ax[2].plot(best_steps[-1, 0], best_steps[-1, 1], '*b',
               label='$x^\prime$', markersize=12, alpha=0.7, markeredgecolor='white')
    ax[2].plot([factual[0], best_steps[-1, 0]],
               [factual[1], best_steps[-1, 1]], '-r', linewidth=2)
    ax[2].plot([best_steps[shortest_path[1], 0], best_steps[shortest_path[3], 0]],
               [best_steps[shortest_path[1], 1], best_steps[shortest_path[3], 1]], '-r', label='illegal edge', linewidth=2)

    ax[2].title.set_text('Enhance')
    ax[2].annotate("", xy=(0.21, 0.375), xytext=(0.21, 0.45), arrowprops=dict(
        arrowstyle="->", lw=2.5, color='black', alpha=.7))
    ax[2].text(0.21, 0.475, 'not allowed', va='center', ha='center',
               rotation='horizontal', fontsize=12, color='black', alpha=.7)
    ax[2].annotate("", xy=(0.19, 0.68), xytext=(0.21, 0.5), arrowprops=dict(
        arrowstyle="->", lw=2.5, color='black', alpha=.7))
    ax[2].annotate("", xy=(0.31, 0.665), xytext=(0.21, 0.5), arrowprops=dict(
        arrowstyle="->", lw=2.5, color='black', alpha=.7))

    fig.tight_layout()
    plt.savefig('local_face/plots/paper_figure_strict_graph.png', format='png')
    plt.show()

if graph:
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6), sharey=True)
    ax = plot_density(ax, X1, dense, levels=5)
    ax = plot_graph(ax, X1, model, best_steps, G, shortest_path)
    ax = plot_dataset(ax, data)
    ax.set_xlim([0, 0.5])
    ax.set_ylim([0.2, 0.85])
    plt.savefig('local_face/plots/network_dense{}.png'.format(factual),
                format='png', dpi=400)
    plt.show()
print('stop')
