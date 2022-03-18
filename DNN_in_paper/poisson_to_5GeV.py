#!/usr/bin/python3
# Install TensorFlow
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten , Convolution2D, MaxPooling2D , Lambda, Conv2D, Activation,Concatenate, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam , SGD , Adagrad
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers , initializers, activations
import tensorflow.keras.backend as K
from sklearn.preprocessing import StandardScaler
from joblib import dump, load

import numpy as np
import matplotlib.pyplot as plt
# import autokeras as ak
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



start = time.time()
# ##############################################################################################################


"""
Define Model
"""
#======================================================#

"""
Model 1
"""
def Regression_Model(num_of_bins):

    input_shape = (num_of_bins,)
    model = Sequential(name = 'Regression_Model')
    model.add(Input(shape=input_shape,name = 'input'))
    model.add(Dense(512, activation='relu', name = 'dense_1'))
    model.add(Dense(256, activation='relu', name = 'dense_2'))
    model.add(Dense(512, activation='relu', name = 'dense_3'))
    out_theta23 = Dense(1, activation='linear', name = "out_theta23")(model.output)
    out_delta = Dense(2, activation='linear', name = "out_delta")(model.output)
    
    
    regression_model = Model(inputs=model.input, 
                             outputs=[out_theta23, out_delta], 
                             name = 'Regression_Model')
    
    model_opt = keras.optimizers.Adadelta(learning_rate=0.0060216)
    
    
    regression_model.compile(loss="mean_squared_error",
                       optimizer=model_opt,
                       metrics=["mse"])
    
    return regression_model





#======================================================#



"""
Load Data
"""
#======================================================#
data = np.load('../Data/n1000000_0910_all_flat.npz')
#======================================================#


"""
Stack Data
"""
#======================================================#
data_all = np.column_stack([data['ve_dune'][:,:36], data['vu_dune'][:,:36], data['vebar_dune'][:,:36], data['vubar_dune'][:,:36]])



"""
Standardization
"""
scaler = StandardScaler()
scaler.fit(data_all)



target = np.column_stack( [data['theta23'] , data["delta"]/180*np.pi])

x_train = data_all[:900000]
y_train = target[:900000]
y_train_theta23 = y_train[:,0]
y_train_delta = np.column_stack([np.sin(y_train[:,1]), np.cos(y_train[:,1])]) 

x_test = data_all[900000:]
y_test = target[900000:]
y_test_theta23 = y_test[:,0]
y_test_delta = np.column_stack([np.sin(y_test[:,1]), np.cos(y_test[:,1])]) 


logging.info("# of train: {}".format(len(x_train)))
logging.info("# of test : {}".format(len(x_test)))
logging.info("y_train_theta23.shape : {}".format(y_train_theta23.shape))
logging.info("y_train_delta.shape : {}".format(y_train_delta.shape))
logging.info("y_test_theta23.shape : {}".format(y_test_theta23.shape))
logging.info("y_test_delta.shape : {}".format(y_test_delta.shape))
#======================================================#


"""
Create Model
"""
#======================================================#
model = Regression_Model(len(x_train[0]))


model.summary()
#======================================================#


"""
Model Training (Poisson Noise)
"""
#======================================================#

time.sleep(0.5)

#++++++++++++++++++++++++++++++++++++++++++#
logging.info("Add Poisson Noise")
logging.info("=====START=====")
t1_time = time.time()
time.sleep(0.5)

"""
Add Poisson Noise
"""
x_train_poisson = np.random.poisson(x_train)
x_train_poisson = scaler.transform(x_train_poisson) 
y_train_theta23_tmp, y_train_delta_tmp = y_train_theta23, y_train_delta

for i in tqdm(range(10)):
    x_train_poisson_tmp = np.random.poisson(x_train)
    x_train_poisson_tmp = scaler.transform(x_train_poisson_tmp) 
    x_train_poisson = np.concatenate([x_train_poisson,x_train_poisson_tmp])
    y_train_theta23 = np.concatenate([y_train_theta23,y_train_theta23_tmp])
    y_train_delta = np.concatenate([y_train_delta,y_train_delta_tmp])

    
logging.info("x_train_poisson.shape : {}".format(x_train_poisson.shape))
logging.info("y_train_theta23.shape : {}".format(y_train_theta23.shape))
logging.info("y_train_delta.shape : {}".format(y_train_delta.shape))

t2_time = time.time()
logging.info("\033[3;33m Time Cost for this Step : {:.4f} min\033[0;m".format((t2_time-t1_time)/60.))
logging.info("=====Finish=====")
logging.info("\n")


"""
Model Training
"""
check_list=[]
checkpoint = ModelCheckpoint(filepath= "./Model/checkmodel_20.h5",
                            save_best_only=True,
                            verbose=0)
csv_logger = CSVLogger("./Training_loss/training_log_20.csv")
earlystopping = EarlyStopping(
                    monitor="val_loss", 
                    min_delta=0,
                    patience=20,
                    verbose=0,
                    mode="auto",
                    baseline=None,
                    restore_best_weights=False,
                )

check_list.append(checkpoint)
check_list.append(csv_logger)
check_list.append(earlystopping)


model.fit( x_train_poisson,
           [y_train_theta23, y_train_delta],
           validation_split = 0.1,
           batch_size=64,
           epochs=200,
           verbose=1,
           shuffle = True,
           callbacks=check_list
         )

model.save("./Model/poisson_20.h5")
#++++++++++++++++++++++++++++++++++++++++++#
    
#======================================================#

x_test_gen = np.random.poisson(x_test)
model.evaluate(x_test_gen, [y_test_theta23, y_test_delta])



##############################################################################################################
finish = time.time()

totaltime =  finish - start
logging.info("\033[3;33mTime Cost : {:.4f} min\033[0;m".format(totaltime/60.))