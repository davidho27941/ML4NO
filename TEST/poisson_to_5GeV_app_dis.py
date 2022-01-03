#!/usr/bin/python3
# Install TensorFlow
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten , Convolution2D, MaxPooling2D , Lambda, Conv2D, Activation,Concatenate, Input, BatchNormalization
from tensorflow.keras.layers import Conv1D, MaxPooling1D
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
# if len(sys.argv) != 3:
#     logging.info("********* Please Check Input Argunment *********")
#     logging.info("********* Usage: python3 poission_learning_study_v1_to_5GeV.py experiment physics_parameter  *********")
#     raise ValueError("********* Usage: python3 poission_learning_study_v1_to_5GeV.py experiment physics_parameter  *********")
    

# try:

#     """
#     Define Name
#     """
#     experiment = str(sys.argv[1])
#     physics_parameter = str(sys.argv[2])

    
#     logging.info("experiment: {}".format(experiment))
#     logging.info("physics parameter: {}".format(physics_parameter))

# except:
#     print("********* Please Check Input Argunment *********")
#     print("********* Usage: python3 poission_learning_study_v1_to_5GeV.py experiment physics_parameter *********")
#     sys.exit(1)


"""
Define Model
"""
#======================================================#

"""
Model 1
"""

"""
Model 5 #11/08 modified
""" 
def Regression_Model(num_of_bins):

    appearance_inputs = Input(shape=(36,2),name = 'appearance_input')
    appearance_conv1d_1 = Conv1D(filters=5, kernel_size=10,strides=1, activation='relu', name = 'appearance_conv1d_1')(appearance_inputs)
    appearance_pooling1d_1 = MaxPooling1D(pool_size=2, strides=None, padding="valid", name = 'appearance_pooling1d_1')(appearance_conv1d_1)
#     appearance_conv1d_2 = Conv1D(filters=5, kernel_size=5,strides=1, activation='relu', name = 'appearance_conv1d_2')(appearance_pooling1d_1)
#     appearance_pooling1d_2 = MaxPooling1D(pool_size=2, strides=None, padding="valid", name = 'appearance_pooling1d_2')(appearance_conv1d_2)
    appearance_flatten_1 = Flatten(name = "appearance_flatten_1")(appearance_pooling1d_1)
    
    disappearance_inputs = Input(shape=(36,2),name = 'disappearance_input')
    disappearance_conv1d_1 = Conv1D(filters=5, kernel_size=10,strides=1, activation='relu', name = 'disappearance_conv1d_1')(disappearance_inputs)
    disappearance_pooling1d_1 = MaxPooling1D(pool_size=2, strides=None, padding="valid", name = 'disappearance_pooling1d_1')(disappearance_conv1d_1)
#     disappearance_conv1d_2 = Conv1D(filters=5, kernel_size=5,strides=1, activation='relu', name = 'disappearance_conv1d_2')(disappearance_pooling1d_1)
#     disappearance_pooling1d_2 = MaxPooling1D(pool_size=2, strides=None, padding="valid", name = 'disappearance_pooling1d_2')(disappearance_conv1d_2)
    disappearance_flatten_1 = Flatten(name = "disappearance_flatten_1")(disappearance_pooling1d_1)
    
    
    mergedOut = Concatenate()([appearance_flatten_1,disappearance_flatten_1])
    dense_1 = Dense(512, activation='relu', name = 'dense_1')(mergedOut)
    dense_2 = Dense(256, activation='relu', name = 'dense_2')(dense_1)
    dense_3 = Dense(512, activation='relu', name = 'dense_3')(dense_2)
    

    out_theta23 = Dense(1, activation='relu', name = "out_theta23")(dense_3)
    out_delta = Dense(2, activation='linear', name = "out_delta")(dense_3)
    
    
    regression_model = Model(inputs=[appearance_inputs,disappearance_inputs], 
                             outputs=[out_theta23, out_delta], 
                             name = 'Regression_Model')
    
    model_opt = keras.optimizers.Adam() #learning_rate=0.0060216
#     model_opt = keras.optimizers.Adadelta(learning_rate=0.0060216)
    
    
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
data_all = np.column_stack([data['ve_dune'][:,:36], data['vebar_dune'][:,:36], data['vu_dune'][:,:36],data['vubar_dune'][:,:36]])



"""
Standardization
"""
scaler = StandardScaler()
scaler.fit(data_all)
# data_all = scaler.transform(data_all)

# file_path = "/dicos_ui_home/alanchung/ML4NO/Standardization.h5"
# if not os.path.isfile(file_path):
#     dump(scaler, file_path)
# else:
#     pass



# target = np.column_stack( [(data['theta23']-38.9-12.2/2)/6.1 , data["delta"]/180*np.pi])
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

# x_train_poisson = np.column_stack([x_train_poisson[:,:36]/169.042, x_train_poisson[:,36:72]/787.425, x_train_poisson[:,72:108]/78.6681, x_train_poisson[:,108:144]/334.052])
for i in tqdm(range(10)):
    x_train_poisson_tmp = np.random.poisson(x_train)
    x_train_poisson_tmp = scaler.transform(x_train_poisson_tmp) 
#     x_train_poisson_tmp = np.column_stack([x_train_poisson_tmp[:,:36]/169.042, x_train_poisson_tmp[:,36:72]/787.425, x_train_poisson_tmp[:,72:108]/78.6681, x_train_poisson_tmp[:,108:144]/334.052])

    x_train_poisson = np.concatenate([x_train_poisson,x_train_poisson_tmp])
    y_train_theta23 = np.concatenate([y_train_theta23,y_train_theta23])
    y_train_delta = np.concatenate([y_train_delta,y_train_delta])
    

appearance_x_train_poisson = x_train_poisson[:,:72]
appearance_x_train_poisson = appearance_x_train_poisson.reshape(len(appearance_x_train_poisson),2,36)
appearance_x_train_poisson = np.array([ element.T for element in appearance_x_train_poisson])

disappearance_x_train_poisson = x_train_poisson[:,72:]
disappearance_x_train_poisson = disappearance_x_train_poisson.reshape(len(disappearance_x_train_poisson),2,36)
disappearance_x_train_poisson = np.array([ element.T for element in disappearance_x_train_poisson])

    
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
checkpoint = ModelCheckpoint(filepath= "./Model/checkmodel_app_dis.h5",
                            save_best_only=True,
                            verbose=0)
csv_logger = CSVLogger("./Training_loss/training_log_app_dis.csv")
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


model.fit( [appearance_x_train_poisson, disappearance_x_train_poisson],
           [y_train_theta23, y_train_delta],
           validation_split = 0.1,
           batch_size=64,
           epochs=200,
           verbose=1,
           shuffle = True,
           callbacks=check_list
         )

model.save("./Model/poisson_app_dis.h5")
#++++++++++++++++++++++++++++++++++++++++++#
    
#======================================================#

x_test_gen = np.random.poisson(x_test)
model.evaluate(x_test_gen, [y_test_theta23, y_test_delta])



##############################################################################################################
finish = time.time()

totaltime =  finish - start
logging.info("\033[3;33mTime Cost : {:.4f} min\033[0;m".format(totaltime/60.))