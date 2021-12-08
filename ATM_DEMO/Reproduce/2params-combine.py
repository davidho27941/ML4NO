#!/usr/bin/python3
# Install TensorFlow
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras import activations
from tensorflow.keras.layers import Dense, Dropout, Flatten , Convolution2D, MaxPooling2D , Lambda, Conv2D, Activation,Concatenate, Input, BatchNormalization
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam , SGD , Adagrad
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers , initializers
import tensorflow.keras.backend as K
from sklearn.preprocessing import StandardScaler
from joblib import dump, load

import numpy as np
import matplotlib.pyplot as plt
import autokeras as ak
import os 
import sys
import time
import importlib
import logging
from tqdm import tqdm

importlib.reload(logging)
logging.basicConfig(level = logging.INFO)

# limit GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   # Restrict TensorFlow to only use the first GPU
try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10000)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
except RuntimeError as e:
# Visible devices must be set before GPUs have been initialized
    print(e)

logging.info("numpy Version is {}".format(np.__version__))
# logging.info("autokeras Version is {}".format(ak.__version__))
logging.info("tensorflow Version is {}".format(tf.keras.__version__))
logging.info("\n")





def Regression_Model(num_of_bins):

    input_shape = (num_of_bins,)
    model = Sequential(name = 'Regression_Model')
    model.add(BatchNormalization(input_shape=input_shape, name = 'BatchNormalization'))
    model.add(Dense(512, activation='relu', name = 'dense_1'))
    model.add(Dense(512, activation='relu', name = 'dense_2'))
    model.add(Dense(1024, activation='relu', name = 'dense_3'))
    
    out_theta23 = Dense(1, activation='relu', name = "out_theta23")(model.output)
    out_delta = Dense(2, activation='linear', name = "out_delta")(model.output)
    
    
    regression_model = Model(inputs=model.input, 
                         outputs=[out_theta23, out_delta], 
                         name = 'Regression_Model')
    
    model_opt = keras.optimizers.Adam()
    
    
    regression_model.compile(loss="mean_squared_error",
                       optimizer=model_opt,
                       metrics=["mse"])
    
    return regression_model



data = np.load('../../Data/n1000000_0910_all_flat.npz')

data_selection = 1 # 0 for all, 1 for lowE(<5GeV), 2 for high(>5GeV)

if data_selection == 0:
    data_all = np.column_stack([data['ve_dune'], data['vu_dune'], data['vebar_dune'], data['vubar_dune']])
elif data_selection == 1:
    data_all = np.column_stack([data['ve_dune'][:,:36], data['vu_dune'][:,:36], data['vebar_dune'][:,:36], data['vubar_dune'][:,:36]])
elif data_selection == 2:
    data_all = np.column_stack([data['ve_dune'][:,36:], data['vu_dune'][:,36:], data['vebar_dune'][:,36:], data['vubar_dune'][:,36:]])

    
"""
v3
"""
target = np.column_stack( [data["theta23"] , data["delta"]/180*np.pi])

x_train = data_all[:10000]
y_train = target[:10000]
y_train_theta23 = y_train[:,0]
y_train_delta = np.column_stack([np.sin(y_train[:,1]), np.cos(y_train[:,1])]) 

x_train2 = data_all[10000:900000]
y_train2 = target[10000:900000]
y_train2_theta23 = y_train2[:,0]
y_train2_delta = np.column_stack([np.sin(y_train2[:,1]), np.cos(y_train2[:,1])]) 


x_test = data_all[900000:]
y_test = target[900000:]
y_test_theta23 = y_test[:,0]
y_test_delta = np.column_stack([np.sin(y_test[:,1]), np.cos(y_test[:,1])]) 

model = Regression_Model(len(x_train[0]))
model.summary()

model.fit(x_train2, 
          [y_train2_theta23, y_train2_delta],
           validation_split = 0.1,
           batch_size=64,
           epochs=20,
           verbose=1,
           shuffle = True
         )


index = 1
while os.path.isfile('./models/1208_2params_{}_v3.h5'.format(index)):
    index += 1
model.save('./models/1208_2params_{}_v3.h5'.format(index))


scale_steps = np.logspace(-3, 0, 30)
before_train_loss = []
after_train_loss = []

for scale in scale_steps:
    x_train2_gen = np.random.normal(x_train2, np.sqrt(x_train2)*scale)
    x_test_gen = np.random.normal(x_test, np.sqrt(x_test)*scale)

    before_train_loss.append(model.evaluate(x_test_gen, y_test)[0])

    model.fit(x_train2_gen, 
              [y_train2_theta23, y_train2_delta],
               validation_split = 0.1,
               batch_size=64,
               epochs=5,
               verbose=1,
               shuffle = True
             )

    after_train_loss.append(model.evaluate(x_test_gen, y_test)[0])
    
    
    
    
model_index = index
index = 1
path = './models_furthurTrain/1208_2params_{}_{}_v3.h5'
while os.path.isfile(path.format(model_index, index)):
    index += 1
model.save(path.format(model_index, index))
outfile = {'scale_steps': scale_steps,
           'before_train_loss': before_train_loss,
           'after_train_loss' :after_train_loss}
np.save(file = './models_furthurTrain/1208_2params_{}_{}_result_v3.npy'.format(model_index, index),
        arr = outfile)

x_test2_gen = np.random.poisson(x_test)

for i in range(10):
    x_train2_gen = np.random.poisson(x_train2)
    
    model.fit(x_train2_gen, 
              [y_train2_theta23, y_train2_delta],
              validation_split=0.1,
               batch_size=64,
               epochs=1,
               verbose=1,
               shuffle = True
             )
model.evaluate(x_test2_gen, [y_test_theta23, y_test_delta])



furthur_index = index
index = 1
path = './models_PoissonTrain/1208_2params_{}_{}_{}_v3.h5'
while os.path.isfile(path.format(model_index, furthur_index, index)):
    index += 1
model.save(path.format(model_index, furthur_index, index))





