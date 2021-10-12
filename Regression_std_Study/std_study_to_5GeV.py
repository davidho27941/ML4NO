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
    logging.info("********* Usage: python3 std_study.py experiment physics_parameter  *********")
    raise ValueError("********* Usage: python3 std_study.py experiment physics_parameter  *********")
    
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
    logging.info("********* Usage: python3 std_study.py experiment physics_parameter  *********")
    sys.exit(1)

    
    
"""
Define Model
"""
#======================================================#
def Regression_Model(name, num_of_bins):

    input_shape = (num_of_bins,)
    model = Sequential(name = "Regression_Model_for_" + str(name))
    model.add(BatchNormalization(input_shape=input_shape, name = 'BatchNormalization'))
    model.add(Dense(512, activation='relu', name = 'dense_1'))
    model.add(Dense(512, activation='relu', name = 'dense_2'))
    model.add(Dense(1024, activation='relu', name = 'dense_3'))
    model.add(Dense(1, activation='relu', name = physics_parameter))
    
    
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
Model Training (Asimov Data)
"""
#======================================================#
check_list=[]
checkpoint = ModelCheckpoint(
            filepath="./Model_to_5Gev/" + str(experiment) + "_" + 
                       str(physics_parameter) + "_" + 
                       str("asimov") + "_" + 
                       "checkpoint.h5",
            save_best_only=True,
            verbose=1)

csv_logger = CSVLogger("./Training_loss_to_5Gev/" + str(experiment) + "_" + 
                       str(physics_parameter) + "_" + 
                       str("asimov") + "_" + 
                       "training_log.csv")

check_list.append(csv_logger)
check_list.append(checkpoint)

model.fit(x_train, y_train,
           validation_split = 0.1,
           batch_size=64,
           epochs=20,
           verbose=1,
           shuffle = True,
           callbacks=check_list
         )

model.save("./Model_to_5Gev/" + str(experiment) + "_" + 
                       str(physics_parameter) + "_" + 
                       str("asimov") + 
                       ".h5")
#======================================================#



"""
Adding Fluctuation and Further Train the Model
"""
#======================================================#
scale_steps = np.logspace(-3, np.log10(2), 40)
before_train_loss = []
after_train_loss = []

for n_std, scale in enumerate(tqdm(scale_steps)):
    time.sleep(0.5)
    
    
    x_train2_gen = np.random.normal(x_train2, np.sqrt(x_train2)*scale)
    x_test_gen = np.random.normal(x_test, np.sqrt(x_test)*scale)

    before_train_loss.append(model.evaluate(x_test_gen, y_test)[0])
    
    
    check_list=[]
    checkpoint = ModelCheckpoint(
                filepath="./Model_to_5Gev/" + str(experiment) + "_" + 
                           str(physics_parameter) + "_" + 
                           "std_" + str(n_std) + "_" + 
                           "checkpoint.h5",
                save_best_only=True,
                verbose=1)

    csv_logger = CSVLogger("./Training_loss_to_5Gev/" + str(experiment) + "_" + 
                           str(physics_parameter) + "_" + 
                           "std_" + str(n_std) + "_" + 
                           "training_log.csv")

    check_list.append(csv_logger)
    check_list.append(checkpoint)


    model.fit(x_train2_gen, y_train2,
               validation_split = 0.1,
               batch_size=64,
               epochs=5, 
               verbose=1,
               shuffle = True,
              callbacks=check_list
             )
    
    after_train_loss.append(model.evaluate(x_test_gen, y_test)[0])

    
    model.save("./Model_to_5Gev/" + str(experiment) + "_" + 
                       str(physics_parameter) + "_" + 
                       "std_" + str(n_std) + "_" + 
                       ".h5")
    
    
    """
    Learning Poisson Noise
    """
    logging.info("\n")
    logging.info("Learning Poisson Noise")
    logging.info("# of std: {}".format(n_std))
    logging.info("  scale : {}".format(scale))
    logging.info("=====START=====")
    t1_time = time.time()
    time.sleep(0.5)
    
    model_learning_poisson = model

    for i in tqdm(range(10)):
        time.sleep(0.5)
        x_train2_gen = np.random.poisson(x_train2)
        
        model_learning_poisson.fit(x_train2_gen, y_train2,
                  validation_split=0.1,
                   batch_size=64,
                   epochs=1,
                   verbose=1,
                   shuffle = True
                 )
        
    x_test2_gen = np.random.poisson(x_test)
    model_learning_poisson.evaluate(x_test2_gen, y_test)
    
    
    """
    Save Poisson Model
    """
    #++++++++++++++++++++++++++++++++++++++++++#
    logging.info("\n")
    logging.info("Save Learnt Poisson Model")
    logging.info("# of std: {}".format(n_std))
    logging.info("  scale : {}".format(scale))
    logging.info("=====START=====")
    time.sleep(0.5)
    
    model_learning_poisson.save("./Model_to_5Gev/" + str(experiment) + "_" + 
                       str(physics_parameter) + "_" + 
                       "std_" + str(n_std) + "_poisson_10_" + 
                       ".h5")
    
    logging.info("=====Finish=====")
    logging.info("\n")
    #++++++++++++++++++++++++++++++++++++++++++#
        
    
    t2_time = time.time()
    logging.info("\033[3;33m Time Cost for this Step : {:.4f} min\033[0;m".format((t2_time-t1_time)/60.))
    logging.info("=====Finish=====")
    logging.info("\n")
#======================================================#
    


##############################################################################################################
finish = time.time()

totaltime =  finish - start
logging.info("\033[3;33mTime Cost : {:.4f} min\033[0;m".format(totaltime/60.))