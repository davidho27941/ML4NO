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
if len(sys.argv) != 3:
    logging.info("********* Please Check Input Argunment *********")
    logging.info("********* Usage: python3 poission_learning_study_v2.py experiment physics_parameter  *********")
    raise ValueError("********* Usage: python3 poission_learning_study_v2.py experiment physics_parameter  *********")
    

try:

    """
    Define Name
    """
    experiment = str(sys.argv[1])
    physics_parameter = str(sys.argv[2])

    
    logging.info("experiment: {}".format(experiment))
    logging.info("physics parameter: {}".format(physics_parameter))

except:
    print("********* Please Check Input Argunment *********")
    print("********* Usage: python3 poission_learning_study_v2.py experiment physics_parameter *********")
    sys.exit(1)
    
    

"""
Define Model
"""
#======================================================#
def Regression_Model(name, num_of_bins):

    input_shape = (num_of_bins,)
    model = Sequential(name = "Poisson_Model_for_" + str(name))
    model.add(BatchNormalization(input_shape=input_shape, name = 'BatchNormalization'))
#     model.add(keras.Input(shape=input_shape, name = 'input'))
#     model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l1(regular_l1),  name = 'dense_1'))
#     model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l1(regular_l1), name = 'dense_2'))
#     model.add(Dense(1024, activation='relu', kernel_regularizer=regularizers.l1(regular_l1), name = 'dense_3'))
    model.add(Dense(512, activation='relu',  name = 'dense_1'))
    model.add(Dense(512, activation='relu', name = 'dense_2'))
    model.add(Dense(1024, activation='relu', name = 'dense_3'))
    model.add(Dense(1, activation="relu", name = physics_parameter))
#     model.add(Dense(1, activation=activations.relu(max_value=1), name = physics_parameter))
    model.add(Dropout(0.1))
    
#     model_opt = keras.optimizers.Adadelta()
    model_opt = keras.optimizers.Adam()
    
    
    model.compile(loss="mean_squared_error",
                       optimizer=model_opt,
                       metrics=["mse"])
    
    return model
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
data_all = np.column_stack([training_data["ve_"+str(experiment)], training_data["vu_"+str(experiment)], training_data["vebar_"+str(experiment)], training_data["vubar_"+str(experiment)]])




# """
# Standardization
# """
# scaler = StandardScaler()
# scaler.fit(data_all)
# data_all = scaler.transform(data_all)

# file_path = "/dicos_ui_home/alanchung/ML4NO/Standardization.h5"
# if not os.path.isfile(file_path):
#     dump(scaler, file_path)
# else:
#     pass



target = training_data[physics_parameter]

x_train = data_all[:900000]
y_train = target[:900000]

x_test = data_all[900000:]
y_test = target[900000:]


logging.info("# of train: {}".format(len(x_train)))
logging.info("# of test : {}".format(len(x_test)))
#======================================================#



"""
Create Model
"""
#======================================================#
model = Regression_Model(physics_parameter, x_train.shape[1])
model.summary()
#======================================================#





"""
Model Training (Poisson Noise)
"""
#======================================================#
for i in tqdm(range(40)):
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


    t2_time = time.time()
    logging.info("\033[3;33m Time Cost for this Step : {:.4f} min\033[0;m".format((t2_time-t1_time)/60.))
    logging.info("=====Finish=====")
    logging.info("\n")


    """
    Model Training
    """
    check_list=[]
    csv_logger = CSVLogger("./Training_loss_v2/" + str(experiment) + "_" + 
                           str(physics_parameter) + "_" +
                           "training_log_poisson_" +str(i)+ ".csv")


    check_list.append(csv_logger)


    model.fit(x_train_poisson, y_train,
               validation_split = 0.1,
               batch_size=64,
               epochs=1,
               verbose=1,
               shuffle = True,
               callbacks=check_list
             )

    model.save("./Model_v2/" + str(experiment) + "_" + 
                           str(physics_parameter) + "_" +
                           "poisson_" + str(i)+ ".h5")
#======================================================#
    

##############################################################################################################
finish = time.time()

totaltime =  finish - start
logging.info("\033[3;33mTime Cost : {:.4f} min\033[0;m".format(totaltime/60.))