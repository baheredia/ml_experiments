# Some utils for dog-breed competition

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import os

import keras.backend as K
from keras.preprocessing import image

# To avoid OOM (using TensorFlow)
def limit_mem():
    K.get_session().close()
    cfg = K.tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    K.set_session(K.tf.Session(config=cfg))

########################################
# Batching
########################################
def get_batch(path, gen=image.ImageDataGenerator(),
              batch_size=4, shuffle=False):
    '''Generate an iterator for batching, and the steps needed in each epoch'''
    iterator = gen.flow_from_directory(path, target_size=(224,224),
                                       batch_size=batch_size,
                                       shuffle=shuffle)
    steps_per_epoch = int(iterator.n/batch_size)
    return iterator, steps_per_epoch

########################################
# Plotting
########################################
def plot_path(path):
    plt.imshow(image.load_img(path))

def plot_array(array):
    plt.imshow(image.array_to_img(array))


########################################
# Submission
########################################

_batch, _  = get_batch('data/sample/valid')
DIC_CLASSES = _batch.class_indices
INV_DIC = {str(v): k for k,v in DIC_CLASSES.items()}


def adj(prediction, top_sum=0.98):
    """Adjust the output of a softmax so that the values are not too extreme"""
    low_bar = 1-top_sum
    adj_pred = np.copy(prediction)
    old_top_sum = np.sum(adj_pred[adj_pred>=low_bar])
    # We scale all the probabilities which are in a nice range, to add up to top_sum
    adj_pred[adj_pred>=low_bar] = (top_sum/old_top_sum)*adj_pred[adj_pred>=low_bar] 
    # And set the rest so that they add up to low_bar.
    adj_pred[adj_pred < low_bar] = low_bar/np.sum(adj_pred < low_bar)
    return adj_pred


test_path = 'data/test/'
def prepare_submission(submission_file, model, top_sum=0.98):
    """Creates the file for the submission.
    submission_file -> filename where to save the submission
    model -> the Keras model which makes the prediction
    top_sum -> to cap the predictions

    returns a pandas DataFrame with the predictions."""
    test_batches, test_steps = get_batch(test_path, batch_size=64, shuffle=False)
    names_of_pics = sorted(os.listdir(test_path+'unknown'))
    predictions = model.predict_generator(test_batches, steps=test_steps)
    test_df = pd.DataFrame()
    test_df['id'] = [name[:-4] for name in names_of_pics]
    test_df.set_index('id', inplace=True)
    probs = np.array([adj(pred, top_sum) for pred in predictions])
    for i in range(len(INV_DIC)):
        test_df[INV_DIC[str(i)]] = np.transpose(probs)[i]
    
    test_df.to_csv(submission_file)
    return test_df
