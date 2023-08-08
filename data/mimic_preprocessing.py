import pandas as pd
import numpy as np


def load_dataset(dataset_name,
                 features,
                 scale=False,
                 test=True):
    """Load the RFD dataset from the csv file into a dataframe.

    Args:
        dataset_name (str): dataset name
        features (lst): list of features
        scale (bool): if to use scaled data (with standard scalar)
        test (bool): which split to load, train or test

    Returns:
        X: features of all instances in a dataframe
        y: labels of all instances in a numpy array
    """
    dataset_df = pd.DataFrame()

    # load rfd dataset
    if dataset_name == "mimic":
        if scale:
            if test == True:
                dataset_file_path = (
                    "rfd_model/results/test_data_standardised.csv")
            else:
                dataset_file_path = (
                    "rfd_model/results/training_data_standardised.csv")
        else:
            if test == True:
                dataset_file_path = ("rfd_model/results/test_data.csv")
            else:
                dataset_file_path = ("rfd_model/results/training_data.csv")
        dataset_df = pd.read_csv(dataset_file_path,
                                 # header=0,
                                 engine="python")

        # deal with airway and sex not being binary, even though they should

        X = dataset_df[features]
        y = dataset_df['outcome']

    return X, y


def factual_selector(dataset, features, model, scale, seed=42):
    '''
    More informed choice of factual
    Can include factual with false negatives

    Args:
        dataset (str): 'mimic'
        features
        model
        seed (int): for random factual selection

    Returns:
        factual: randomly selected patient

    '''
    # Load dataset
    X, y = load_dataset(dataset, features, scale=scale)

    # Make predictions on X
    pred = model.predict(X.to_numpy())

    # Compare predictions to reality
    df = pd.DataFrame(X)
    df['y_true'] = y.tolist()
    df['y_pred'] = pred.tolist()

    # all false negatives: patients who were ready for discharge but not classified as such
    fn = df[df['y_true'] == 1 & (df['y_pred'] == 0)]

    # Now select factual
    factual = fn.sample(n=1, random_state=seed)
    print(factual)

    factual = factual[features]
    factual = np.array(factual)[0]

    return factual
