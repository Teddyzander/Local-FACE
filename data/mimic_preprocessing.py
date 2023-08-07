import pandas as pd
import numpy as np


def load_dataset(dataset_name, features, test=True):
    """Load the RFD dataset from the csv file into a dataframe.

    Args:
        dataset_name (str): dataset name
        test (bool): which split to load, train or test

    Returns:
        X: features of all instances in a dataframe
        y: labels of all instances in a numpy array
    """
    dataset_df = pd.DataFrame()

    # load rfd dataset
    if dataset_name == "mimic":
        if test == True:
            dataset_file_path = ("rfd_model/results/test_data.csv")
        else:
            dataset_file_path = ("rfd_model/results/training_data.csv")
        dataset_df = pd.read_csv(dataset_file_path,
                                 # header=0,
                                 engine="python")
        X = dataset_df[features]
        y = dataset_df['outcome']

    return X, y


def factual_selector(dataset, features, model, target=''):
    '''
    More informed choice of factual
    Can include factual with false negatives

    Args:
        dataset (str): 'mimic'
        features
        model
        target (str): optional 'fn', 'fp'

    Returns:
        factual: randomly selected patient

    '''
    # Load dataset
    X, y = load_dataset(dataset, features)

    # Make predictions on X
    pred = model.predict(X.to_numpy())

    # Compare predictions to reality
    df = pd.DataFrame()
    df['y_true'] = y.tolist()
    df['y_pred'] = pred.tolist()
    print(df)

    comp = np.empty(len(y))
    for id, patient in enumerate(y):
        if patient == pred[id]:
            print('matches')
            np.append(comp, 1)
        else:
            print("doesn't match")
            np.append(comp, 0)

    # Now select factual

    return comp
