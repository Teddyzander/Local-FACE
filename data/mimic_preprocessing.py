from sklearn.neural_network import MLPClassifier
import pandas as pd
import os
import numpy as np


def load_dataset(dataset_name):
    """Load the dataset from the csv file into a dataframe.

    Args:
        dataset_name (str): dataset name

    Returns:
        X: features of all instances in a dataframe
        y: labels of all instances in a numpy array
    """
    dataset_df = pd.DataFrame()

    # load HELOC dataset
    if dataset_name == "mimic":
        dataset_file_path = ("data/data/mimic_data.csv")
        dataset_df = pd.read_csv(dataset_file_path, header=0, engine="python")
        X, y = process_mimic(dataset_df)

    return X, y


def process_mimic(data_df):
    """Delete instances that have missing fields (-9) across all their features.

    Args:
        data_df (DataFrame): a dataframe file with raw input

    Returns:
        X: features of all data instances in a dataframe (features are not normalised)
        y: labels of all data instances as a numpy array
    """

    X = data_df.iloc[:, 1:-2]
    y = data_df.iloc[:, -1:]  # 1 = ready for discharge

    feature_names = X.columns
    num_features = len(feature_names)

    return X, y


X, y = load_dataset('mimic')

params = {
    # "hidden_layer_sizes": (30, 10, ),
    # "solver": "adam",
    # "activation": "relu",
    # "alpha": 1e-4,
    "max_iter": 5000,
}


model = MLPClassifier(**params)

model.fit(X, np.ravel(y))


print(model.predict(X[:10]))

print(model.predict_proba(X[:10]))
