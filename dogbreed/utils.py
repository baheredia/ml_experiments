# Some utils for dog-breed competition

import matplotlib.pyplot as plt

import keras.backend as K
from keras.preprocessing import image

# To avoid OOM (using TensorFlow)
def limit_mem():
    K.get_session().close()
    cfg = K.tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    K.set_session(K.tf.Session(config=cfg))


def get_batch(path, gen=image.ImageDataGenerator(),
              batch_size=4, shuffle=False):
    '''Generate an iterator for batching, and the steps needed in each epoch'''
    iterator = gen.flow_from_directory(path, target_size=(224,224),
                                       batch_size=batch_size,
                                       shuffle=shuffle)
    steps_per_epoch = int(iterator.n/batch_size)
    return iterator, steps_per_epoch

def plot_path(path):
    plt.imshow(image.load_img(path))

def plot_array(array):
    plt.imshow(image.array_to_img(array))
