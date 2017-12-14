import pandas as pd

import keras.backend as K

from gini import *
from data_prep import load_array

def limit_mem():
    """Limit GPU memory use for tensorflow"""
    K.get_session().close()
    cfg = K.tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    K.set_session(K.tf.Session(config=cfg))


def gini_model(model, valid_ft, valid_targets):
    """Gives gini score on a validation set"""
    pred = model.predict(valid_ft)
    return gini_normalized(valid_targets, pred)

def create_submission(submission_file, model):
    """Creates a submission file for Kaggle"""
    test_features = load_array('data/test_features.bc/')
    ids = pd.read_csv('data/test.csv', usecols=['id']).id.values
    predictions = model.predict(test_features)
    pred_df = pd.DataFrame()
    pred_df['id'] = ids
    pred_df['target'] = predictions
    pred_df.to_csv(submission_file,index=False)
    return pred_df
