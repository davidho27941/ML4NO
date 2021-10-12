#!/usr/bin/python3
# Install TensorFlow
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten , Convolution2D, MaxPooling2D , Lambda, Conv2D, Activation,Concatenate, Input, BatchNormalization
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
##############################################################################################################
if len(sys.argv) < 3:
    logging.info("********* Please Check Input Argunment *********")
    logging.info("********* Usage: python3 poisson_loop_study_full_energy.py experiment physics_parameter  *********")
    raise ValueError("********* Usage: python3 poisson_loop_study_full_energy.py experiment physics_parameter  *********")
    
try:

    """
    Define Name
    """
    experiment = str(sys.argv[1])
    physics_parameter = str(sys.argv[2])

    
    logging.info("experiment: {}".format(experiment))
    logging.info("physics parameter: {}".format(physics_parameter))
    
    
except:
    logging.info("********* Please Check Input Argunment *********")
    logging.info("********* Usage: python3 poisson_loop_study_full_energy.py experiment physics_parameter  *********")
    sys.exit(1)

    


"""
Load Data
"""
training_data = np.load("../Data/n1000000_0910_all_flat.npz") 

#TODO : change to be argv[?]




"""
Stack Data
"""
#======================================================#
data_all = np.column_stack([training_data["ve_"+str(experiment)][:,:36], training_data["vu_"+str(experiment)][:,:36], training_data["vebar_"+str(experiment)][:,:36], training_data["vubar_"+str(experiment)][:,:36]])


target = training_data[physics_parameter]

x_train = data_all[:10000]
y_train = target[:10000]

x_train2 = data_all[10000:900000]
y_train2 = target[10000:900000]

x_test = data_all[900000:]
y_test = target[900000:]



logging.info("# of train: {}".format(len(x_train2)))
logging.info("# of test : {}".format(len(x_test)))
#======================================================#


"""
Load Model
"""
#======================================================#
model = load_model("./Model_to_5Gev/" + str(experiment) + "_" + 
                       str(physics_parameter) + "_" + 
                       "std_35_" + 
                       ".h5")
std = np.logspace(-3, np.log10(2), 40)[35]

logging.info("std = {:.3f}".format(std))
model.summary()
#======================================================#



"""
Adding Poisson Fluctuation 
"""
#======================================================#

model_learning_poisson = model

for i in tqdm(range(20)):
    time.sleep(0.5)
    x_train2_gen = np.random.poisson(x_train2)

    model_learning_poisson.fit(x_train2_gen, y_train2,
              validation_split=0.1,
               batch_size=64,
               epochs=1,
               verbose=1,
               shuffle = True
             )
    
    """
    Save Poisson Model
    """
    #++++++++++++++++++++++++++++++++++++++++++#
    logging.info("\n")
    logging.info("Save Learnt Poisson Model")
    logging.info("  # of loop  : {}".format(i))
    logging.info("=====START=====")
    time.sleep(0.5)

    model_learning_poisson.save("./Model_poisson_loop_to_5Gev/" + str(experiment) + "_" + 
                       str(physics_parameter) + "_" + 
                       "std_35" + "_poisson_" + str(i) + 
                       ".h5")

    logging.info("=====Finish=====")
    logging.info("\n")
    #++++++++++++++++++++++++++++++++++++++++++#




#======================================================#
    


##############################################################################################################
finish = time.time()

totaltime =  finish - start
logging.info("\033[3;33mTime Cost : {:.4f} min\033[0;m".format(totaltime/60.))