import bcolz
import numpy as np

def load_training_data(path = 'data/', seed=42):
    """Load the training already splitted into training and validation.

    path - string (optional)
        Path where the data is stored.

    seed - integer (optional)
        Seed for the random splitting (if not given 42 is used).

    return
    ------
    train_ft     - Array with the training data.
    train_target - Array with the training target.
    valid_ft     - Array with the validation data.
    valid_target - Array with the validation target.
    """
    trn_features = load_array(path + 'train_features.bc')
    trn_targets = load_array(path + 'targets.bc')
    
    myrandom = np.random.RandomState(seed) # This is to make the analysis reproducible
    msk = myrandom.rand(len(trn_features)) > 0.2
    train_ft = trn_features[msk]
    train_targets = trn_targets[msk]
    valid_ft = trn_features[~msk]
    valid_targets = trn_targets[~msk]
    
    return train_ft, train_targets, valid_ft, valid_targets

def save_array(path, array):
    """Save an array as a bcolz array"""
    c = bcolz.carray(array, rootdir=path, mode='w')
    c.flush()
    
def load_array(path):
    """Load a bcolz array as a numpy array"""
    return bcolz.open(path)[:]


