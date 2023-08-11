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
seed = 8
method_type = 'strict'
prob_dense = 0.01

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
            'rfd_model/results/rf_combined_standardscale.pickle',
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
upper_age = ">"+str(factual[20] + 0.5)
lower_age = "<"+str(factual[20] - 0.5)
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
    ['!=' + str(factual[21])],  # 'sex'
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

# Plot probabilites over the path
fig, ax = plt.subplots(2, 1)
probs = model.predict_proba(best_steps)
rfd_probs = [item[1] for item in probs]
ax[0].plot(rfd_probs,
           label=('Probability Ready for Discharge')
           )
ax[0].legend()
num_inst = len(path_df.index)
ind_inst = np.arange(0, num_inst)
for i in range(top_n_features):
    ax[1].plot(ind_inst, volatile_combined.iloc[:, i],
               label=str(list(volatile_combined.columns.values)[i]))
ax[1].legend(loc='lower left', framealpha=0.3)
plt.savefig("local_face/plots/RFD/RFD_{}".format(seed))
plt.show()

print('Top features to track between examples:')
for i in range(1, len(path_df.index)):
    print('instance {} with RFD certainty {}'.format(
        i, model.predict_proba([np.array(path_df.iloc[i])])[0, 1]))
    inst = path_df.iloc[i-1:i+1]
    temp = inst.std().sort_values(ascending=False)[0:top_n_features]
    volatile_feats = temp.index.values

    # Then extract these relevant columns from the path dataframe (combined)
    volatile_combined = temp[volatile_feats]
    print(volatile_combined)
