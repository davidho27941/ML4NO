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
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6000)])
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
##############################################################################################################
if len(sys.argv) != 2:
    logging.info("********* Please Check Input Argunment *********")
    logging.info("********* Usage: python3 poisson_learning_study_new_prediction_to_5GeV.py experiment  *********")
    raise ValueError("********* Usage: python3 poisson_learning_study_new_prediction_to_5GeV.py experiment  *********")
    

try:

    """
    Define Name
    """
    experiment = str(sys.argv[1])

    logging.info("experiment: {}".format(experiment))

except:
    print("********* Please Check Input Argunment *********")
    print("********* Usage: python3 poisson_learning_study_new_prediction_to_5GeV.py experiment *********")
    sys.exit(1)
    
    

"""
Define Model
"""
#------------------------------------------------------------------------------------------------
# 1D CNN Ref: https://blog.goodaudience.com/introduction-to-1d-convolutional-neural-networks-in-keras-for-time-sequences-3a7ff801a2cf
#------------------------------------------------------------------------------------------------

#======================================================#
# """
# Model 1 #11/04 modified
# """
# def my_loss_fn(y_true, y_pred):
#     squared_difference = tf.square(y_true - y_pred)
#     mse = tf.reduce_mean(squared_difference, axis=-1) 
    
#     preserve_length = tf.sqrt(tf.reduce_sum(tf.square(y_pred), axis=-1))
#     preserve_length = tf.abs(tf.subtract(preserve_length, 1))
    
#     loss = tf.add(mse, preserve_length)
#     return loss #tf.reduce_mean(squared_difference, axis=-1)  # Note the `axis=-1`

# def Regression_Model(trig=False):
#     #Ref: https://towardsdatascience.com/can-machine-learn-the-concept-of-sine-4047dced3f11
#     inputs = Input(shape=(36,4),name = 'input')
#     conv1d_1 = Conv1D(filters=5, kernel_size=10,strides=1, activation='relu', name = 'conv1d_1')(inputs)
#     conv1d_2 = Conv1D(filters=5, kernel_size=10,strides=1, activation='relu', name = 'conv1d_2')(conv1d_1)
#     flatten_1 = Flatten(name = "flatten_1")(conv1d_2)
#     dense_1 = Dense(512, activation='relu', name = 'dense_1')(flatten_1)
#     dense_2 = Dense(256, activation='relu', name = 'dense_2')(dense_1)
#     dense_3 = Dense(256, activation='relu', name = 'dense_3')(dense_2)
#     dense_4 = Dense(128, activation='relu', name = 'dense_4')(dense_3)
#     dense_5 = Dense(128, activation='relu', name = 'dense_5')(dense_4)
#     if trig == False:
#         out1 = Dense(1, activation='linear', name = "output1")(dense_5)
# #         out2 = Dense(1, activation='linear', name = "output2")(dense_5)
#     elif trig == True:
#         out1 = Dense(4, activation='linear', name = "output1")(dense_5)
# #         out2 = Dense(2, activation='linear', name = "output2")(dense_5)

# #     regression_model = Model(inputs=inputs, outputs=[out1,out2], name = 'Model')
#     regression_model = Model(inputs=inputs, outputs=out1, name = 'Model')



# #     model_opt = keras.optimizers.Adadelta()
#     model_opt = keras.optimizers.Adam()

#     regression_model.compile(loss="mean_squared_error",
#                        optimizer=model_opt,
#                        metrics=["mse","mae"])

#     return regression_model

# """
# Model 2 #11/04 modified
# """
# def Regression_Model(trig=False):
#     #Ref: https://towardsdatascience.com/can-machine-learn-the-concept-of-sine-4047dced3f11
#     inputs = Input(shape=(36,4),name = 'input')
#     conv1d_1 = Conv1D(filters=5, kernel_size=10,strides=1, activation='relu', name = 'conv1d_1')(inputs)
#     pooling1d_1 = MaxPooling1D(pool_size=2, strides=None, padding="valid", name = 'pooling1d_1')(conv1d_1)
#     conv1d_2 = Conv1D(filters=5, kernel_size=5,strides=1, activation='relu', name = 'conv1d_2')(pooling1d_1)
#     pooling1d_2 = MaxPooling1D(pool_size=2, strides=None, padding="valid", name = 'pooling1d_2')(conv1d_2)
#     flatten_1 = Flatten(name = "flatten_1")(pooling1d_2)
#     dense_1 = Dense(512, activation='relu', name = 'dense_1')(flatten_1)
#     dense_2 = Dense(256, activation='relu', name = 'dense_2')(dense_1)
#     dense_3 = Dense(256, activation='relu', name = 'dense_3')(dense_2)
#     dense_4 = Dense(128, activation='relu', name = 'dense_4')(dense_3)
#     dense_5 = Dense(128, activation='relu', name = 'dense_5')(dense_4)
    
#     if trig == False:
#         out1 = Dense(1, activation='linear', name = "output1")(dense_5)
# #         out2 = Dense(1, activation='linear', name = "output2")(dense_5)
#     elif trig == True:
#         out1 = Dense(4, activation='linear', name = "output1")(dense_5)
# #         out2 = Dense(2, activation='linear', name = "output2")(dense_5)

# #     regression_model = Model(inputs=inputs, outputs=[out1,out2], name = 'Model')
#     regression_model = Model(inputs=inputs, outputs=out1, name = 'Model')



# #     model_opt = keras.optimizers.Adadelta()
#     model_opt = keras.optimizers.Adam()

#     regression_model.compile(loss="mean_squared_error",
#                        optimizer=model_opt,
#                        metrics=["mse","mae"])

#     return regression_model


# """
# Model 3 #11/05 modified
# """
# def Regression_Model(trig=False):
#     #Ref: https://towardsdatascience.com/can-machine-learn-the-concept-of-sine-4047dced3f11
#     inputs = Input(shape=(36,4),name = 'input')
#     conv1d_1 = Conv1D(filters=5, kernel_size=10,strides=1, activation='relu', name = 'conv1d_1')(inputs)
#     conv1d_2 = Conv1D(filters=5, kernel_size=10,strides=1, activation='relu', name = 'conv1d_2')(conv1d_1)
#     flatten_1 = Flatten(name = "flatten_1")(conv1d_2)
#     dense_1 = Dense(512, activation='relu', name = 'dense_1')(flatten_1)
#     dense_2 = Dense(256, activation='relu', name = 'dense_2')(dense_1)
#     dense_3 = Dense(256, activation='relu', name = 'dense_3')(dense_2)
#     dense_4 = Dense(128, activation='relu', name = 'dense_4')(dense_3)
#     dense_5 = Dense(128, activation='relu', name = 'dense_5')(dense_4)
    
#     if trig == False:
#         out1 = Dense(1, activation='linear', name = "output1")(dense_5)
        
#     elif trig == True:
#         out_delta = Dense(2, activation='linear', name = "out_delta")(dense_5)
#         out_theta23 = Dense(1, activation='relu', name = "out_theta23")(dense_5)

#     regression_model = Model(inputs=inputs, outputs=[out_delta, out_theta23], name = 'Model_3')



# #     model_opt = keras.optimizers.Adadelta()
#     model_opt = keras.optimizers.Adam()

#     regression_model.compile(loss="mean_squared_error",
#                        optimizer=model_opt,
#                        metrics=["mse","mae"])

#     return regression_model

"""
Model 4 #11/05 modified
"""
def Regression_Model(trig=False):
    #Ref: https://towardsdatascience.com/can-machine-learn-the-concept-of-sine-4047dced3f11
    inputs = Input(shape=(36,4),name = 'input')
    conv1d_1 = Conv1D(filters=5, kernel_size=10,strides=1, activation='relu', name = 'conv1d_1')(inputs)
    pooling1d_1 = MaxPooling1D(pool_size=2, strides=None, padding="valid", name = 'pooling1d_1')(conv1d_1)
    conv1d_2 = Conv1D(filters=5, kernel_size=5,strides=1, activation='relu', name = 'conv1d_2')(pooling1d_1)
    pooling1d_2 = MaxPooling1D(pool_size=2, strides=None, padding="valid", name = 'pooling1d_2')(conv1d_2)
    flatten_1 = Flatten(name = "flatten_1")(pooling1d_2)
    dense_1 = Dense(512, activation='relu', name = 'dense_1')(flatten_1)
    dense_2 = Dense(256, activation='relu', name = 'dense_2')(dense_1)
    dense_3 = Dense(256, activation='relu', name = 'dense_3')(dense_2)
    dense_4 = Dense(128, activation='relu', name = 'dense_4')(dense_3)
    dense_5 = Dense(128, activation='relu', name = 'dense_5')(dense_4)
    
    if trig == False:
        out1 = Dense(1, activation='linear', name = "output1")(dense_5)
        
    elif trig == True:
        out_delta = Dense(2, activation='linear', name = "out_delta")(dense_5)
        out_theta23 = Dense(1, activation='relu', name = "out_theta23")(dense_5)

    regression_model = Model(inputs=inputs, outputs=[out_delta, out_theta23], name = 'Model_3')



#     model_opt = keras.optimizers.Adadelta()
    model_opt = keras.optimizers.Adam()

    regression_model.compile(loss="mean_squared_error",
                       optimizer=model_opt,
                       metrics=["mse","mae"])

    return regression_model



#======================================================#
"""
Load Data
"""
#======================================================#
training_data = np.load("../Data/n1000000_0910_all_flat.npz")
#======================================================#


"""
Stack Data
"""
#======================================================#
data_all = np.column_stack([training_data["ve_"+str(experiment)][:,:36], training_data["vu_"+str(experiment)][:,:36], training_data["vebar_"+str(experiment)][:,:36], training_data["vubar_"+str(experiment)][:,:36]])


target = np.column_stack( [training_data["delta"], training_data["theta23"]])
target = target/180*np.pi 

x_train = data_all[:900000]
y_train = target[:900000]
# y_train = np.column_stack([np.sin(y_train[:,0]), np.cos(y_train[:,0]), np.sin(y_train[:,1]), np.cos(y_train[:,1])]) # 11/04 modified
y_train_delta = np.column_stack([np.sin(y_train[:,0]), np.cos(y_train[:,0])]) # 11/05 modified
y_train_theta23 = y_train[:,1]

x_test = data_all[900000:]
y_test = target[900000:]
# y_test = np.column_stack([np.sin(y_test[:,0]), np.cos(y_test[:,0]), np.sin(y_test[:,1]), np.cos(y_test[:,1])]) # 11/04  modified
y_test_delta = np.column_stack([np.sin(y_test[:,0]), np.cos(y_test[:,0])]) # 11/05  modified
y_test_theta23 = y_test[:,1]

logging.info("X train shape: {}".format(x_train.shape))
logging.info("X test shape: {}".format(x_test.shape))
logging.info("Y train shape: {}".format(y_train.shape))
logging.info("Y test shape: {}".format(y_test.shape))


#======================================================#



"""
Create Model
"""
#======================================================#
model = Regression_Model(trig=True)


model.summary()
#======================================================#



"""
Model Training (Poisson Noise)
"""
#======================================================#
for i in tqdm(range(0,300,1)):
    time.sleep(0.5)


    logging.info("# of training: {}".format(i))
    logging.info("Add Poisson Noise")
    logging.info("=====START=====")
    t1_time = time.time()
    time.sleep(0.5)
    
    """
    Add Poisson Noise
    """
    x_train_poisson = np.random.poisson(x_train)
    x_train_poisson = x_train_poisson.reshape(len(x_train_poisson),4,36)
    x_train_poisson = np.array([ element.T for element in x_train_poisson])
    
    logging.info("\n")
    logging.info("x_train_poisson shape: {}".format(x_train_poisson.shape))
    logging.info("\n")
    
    t2_time = time.time()
    logging.info("\033[3;33m Time Cost for this Step : {:.4f} min\033[0;m".format((t2_time-t1_time)/60.))
    logging.info("=====Finish=====")
    logging.info("\n")


    """
    Model Training
    """
    check_list=[]
    csv_logger = CSVLogger("./Training_loss/" + str(experiment) + "_" + 
                           "training_log_poisson_" +str(i)+ "_4.csv")


    check_list.append(csv_logger)


    model.fit( x_train_poisson,
#                y_train, # 11/05 modified
              [y_train_delta, y_train_theta23],  # 11/05 modified
               validation_split = 0.1,
               batch_size=64,
               epochs=1,
               verbose=1,
               shuffle = True,
               callbacks=check_list
             )

    model.save("./Model/" + str(experiment) + "_" + 
                           "poisson_" + str(i)+ "_4.h5")
#======================================================#
    

##############################################################################################################
finish = time.time()

totaltime =  finish - start
logging.info("\033[3;33mTime Cost : {:.4f} min\033[0;m".format(totaltime/60.))