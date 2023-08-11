import pandas as pd
import numpy as np


def load_dataset(dataset_name,
                 features,
                 scale=True,
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
                    "rfd_model/results/combined_test_data_standardscale.csv"
                )
            else:
                dataset_file_path = (
                    "rfd_model/results/combined_training_data_standardscale.csv"
                )
        else:
            if test == True:
                dataset_file_path = ("rfd_model/results/legacy/test_data.csv")
            else:
                dataset_file_path = (
                    "rfd_model/results/legacy/training_data.csv")
        dataset_df = pd.read_csv(dataset_file_path,
                                 # header=0,
                                 engine="python")

        # deal with airway and sex not being binary, even though they should

        X = dataset_df[features]
        y = dataset_df['outcome']

    return X, y


def factual_selector(dataset, features, model, scale, seed=42, alignment='fn'):
    '''
    More informed choice of factual
    Can include factual with false negatives

    Args:
        dataset (str): 'mimic'
        features
        model
        seed (int): for random factual selection
        alignment (str): 'fn' for false negatives, 'fp' for false positives

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

    if alignment == 'fn':
        # all false negatives: patients who were ready for discharge but not classified as such
        fn = df[df['y_true'] == 1 & (df['y_pred'] == 0)]
        print(f'In total there are {len(fn)} FN cases')
        # Now select factual
        factual = fn.sample(n=1, random_state=seed)
    elif alignment == 'fp':
        # all false positives: patients who weren't ready for discharge but were classified as such
        fp = df[df['y_true'] == 0 & (df['y_pred'] == 1)]
        print(f'In total there are {len(fp)} FP cases')
        # Now select factual
        factual = fp.sample(n=1, random_state=seed)
    else:
        print('Alignment arg not recognised, please input "fn" or "fp".')
    print(factual)

    factual = factual[features]
    factual = np.array(factual)[0]

    return factual
