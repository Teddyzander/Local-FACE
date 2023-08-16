import pandas as pd
import numpy as np
import os

# input: name, output: processed dataframe file
def load_dataset(dataset_name):
    """Load the dataset from the csv file into a dataframe.

    Args:
        dataset_name (str): dataset name

    Returns:
        X: features of all instances in a dataframe
        y: labels of all instances in a numpy array
    """
    dataset_df = pd.DataFrame()
    current_path = os.path.dirname(os.path.realpath(__file__))
    dataset_folder_path = os.path.join(current_path, "raw_data/")

    # load HELOC dataset
    if dataset_name == "heloc":
        dataset_file_path = "{}{}".format(dataset_folder_path, "heloc/heloc_dataset_v1.csv")
        dataset_df = pd.read_csv(dataset_file_path, header=0, engine="python")
        X, y = process_heloc(dataset_df)

    # load mnist dataset
    elif dataset_name == "mnist":
        dataset_file_path = "{}{}".format(dataset_folder_path, "mnist/mnist_train.csv")
        dataset_df = pd.read_csv(dataset_file_path, delimiter=",", header=None, dtype=np.uint8)
        X, y = process_mnist(dataset_df)

    # load mimic dataset
    elif dataset_name == "mimic":
        dataset_file_path = "{}{}".format(dataset_folder_path, "fm_MIMIC_COMPLETECASE_original.csv")
        dataset_df = pd.read_csv(dataset_file_path, header=0, engine="python")
        X, y = process_mimic(dataset_df)
    
    return X, y

def process_heloc(data_df):
    """Delete instances that have missing fields (-9) across all their features.

    Args:
        data_df (DataFrame): a dataframe file with raw input

    Returns:
        X: features of all data instances in a dataframe (features are not normalised)
        y: labels of all data instances as a numpy array
    """

    X = data_df.iloc[:, 1:]
    feature_names = X.columns
    num_features = len(feature_names)

    # Delete instances with feature values of all "-9".
    for idx, instance in X.iterrows():
        is_nah = [True if instance[i] == -9 else False for i in range(num_features)]
        if len(set(is_nah)) == 1 and True in set(is_nah):
            data_df.drop(index=idx, inplace = True, axis=0)
    
    data_df = data_df.reset_index(drop=True)
    X = data_df.iloc[:, 1:]
    y_original = data_df.iloc[:, 0]
    y = np.asarray([0 if i == "Bad" else 1 for i in y_original])
    return X, y


def process_mnist(data_df):
    """ Process the MNIST dataset such that it is ready for model prediction.

    Args:
        data_df (DataFrame): a dataframe file read from raw input file

    Returns:
        X: features of all data instances as a dataframe
        y: labels of all data instances as a numpy array
    """

    # Pixel values need to be normalised between 0 and 1 for 
    # effective model training and distance calculation.
    X = (data_df.iloc[:, 1:]/255).astype("float32")
    y = np.asarray(data_df.iloc[:, 0])    

    # Sample a subset with 1000 instances for each digit.
    # Originally, each digit has 5000-6000 instances.
    # If we want to keep all instances, comment out the following code
    all_indices = []
    for digit in [1,9]:
    # for digit in np.arange(0,10):
        single_digit_indices = set(np.where(y == digit)[0][:100])
        all_indices = set(all_indices).union(single_digit_indices)

    X = X.iloc[list(all_indices), :]
    y = y[list(all_indices)]

    return X, y

def process_mimic(data_df):

    """ MIMIC dataset cleaned by Chris McWilliams for https://bmjopen.bmj.com/content/9/3/e025925

    Args:
        data_df (DataFrame): a dataframe file with raw input

    Returns:
        X: features of all data instances in a dataframe (features are not normalised)
        y: labels of all data instances as a numpy array
    """

    X = data_df.iloc[:, 1:-2]
    y = data_df.iloc[:, -1:]  # 1 == ready for discharge

    return X, y

