import numpy as np

import keras.backend as K
from keras.layers import Dense, Flatten, Lambda, BatchNormalization, Dropout
from keras.layers import Conv2D, MaxPool2D
from keras.models import Model, Sequential
from keras.optimizers import Adam, RMSprop
from keras.preprocessing import image

VGG16_PATH = 'models/vgg16_weights_tf_dim_ordering_tf_kernels.h5'

vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((1,1,3))
def preproc(x):
    x = x - vgg_mean
    return x[:,:,:,::-1]

def conv_block(model, layers, filters):
    for i in range(layers):
        model.add(Conv2D(filters, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

def fc_block(model, do):
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(do))

def vgg16(do):
    model = Sequential()
    
    model.add(Lambda(preproc, input_shape=(224,224,3)))
    
    conv_block(model, 2, 64)
    conv_block(model, 2, 128)
    conv_block(model, 3, 256)
    conv_block(model, 3, 512)
    conv_block(model, 3, 512)
    
    model.add(Flatten())
    fc_block(model, do)
    fc_block(model, do)
    model.add(Dense(1000, activation='softmax'))

    model.load_weights(VGG16_PATH)
    
    return model

def vgg16_dogbreed(do):
    model = vgg16(do)

    model.pop()
    model.add(Dense(120, activation='softmax'))

    return model


    
