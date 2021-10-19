#!/usr/bin/python3
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
from numpy import random
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
# # if gpus:
# #   # Restrict TensorFlow to only use the first GPU
# try:
#     tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
#     tf.config.experimental.set_virtual_device_configuration(
#     gpus[0],
#     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1000)])
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
# except RuntimeError as e:
# # Visible devices must be set before GPUs have been initialized
#     print(e)

logging.info("numpy Version is {}".format(np.__version__))
# logging.info("autokeras Version is {}".format(ak.__version__))
logging.info("tensorflow Version is {}".format(tf.keras.__version__))
logging.info("\n")




##############################################################################################################

def Contour(resolution_dictinary, model_dictinary, bin_size, model="model_1"):
    logging.info("==========================================================")
    logging.info("===================== Model: {} =====================".format(model))
    logging.info("==========================================================")
    logging.info("\n")
    for i, std_scale in enumerate(tqdm(resolution_dictinary)):
        
        if model == "model_1" or model == "model_2":
            prediction_asimov = model_dictinary[std_scale].predict(data_asimov)[0][0]
            tmp = model_dictinary[std_scale].predict(data_poisson)[:,0]
            
        elif model == "model_3" or model == "model_4" or model == "model_5":
            prediction_asimov = model_dictinary[std_scale].predict([data_asimov,data_asimov[:,:72]-data_asimov[:,72:]])[0][0]
            tmp = model_dictinary[std_scale].predict([data_poisson,data_poisson[:,:72]-data_poisson[:,72:]])[:,0]
        else:
            logging.info("Please Check Input Model!!!")
            raise ValueError("Please Check Input Model!!!")

        hist, bin_edges = np.histogram(tmp, bins = bin_size)
        max_poi = np.where(hist == hist.max())

        """
        left boundary
        """
        tot_event_num = hist.sum()
        left_area = 0
        for left_boundary in range(len(hist)):
            left_area += hist[left_boundary]
            if left_area/tot_event_num >= 0.34:
                break

        left_boundary = bin_edges[left_boundary]

        """
        right boundary
        """
        tot_event_num = hist.sum()
        right_area = 0
        for right_boundary in np.linspace(len(hist)-1,0,len(hist)):
            right_area += hist[int(right_boundary)]
            if right_area/tot_event_num >= 0.34:
                break

        right_boundary = bin_edges[int(right_boundary)]
        
#         logging.info("\n")
#         logging.info(i)
#         logging.info("prediction asimov: {:.1f}".format(prediction_asimov))
#         logging.info("right_boundary: {:.1f}".format(right_boundary))
#         logging.info("best fit: {:.1f}".format(bin_edges[max_poi][0]))
#         logging.info("left_boundary: {:.1f}".format(left_boundary))
#         logging.info("################")
#         logging.info("\n")

        resolution_dictinary[std_scale]["poission_prediction"] = tmp
        resolution_dictinary[std_scale]["asimov"] = prediction_asimov
        resolution_dictinary[std_scale]["hist"] = hist
        resolution_dictinary[std_scale]["best_fit"] = bin_edges[max_poi][0]
        resolution_dictinary[std_scale]["p_yerr"] = right_boundary
        resolution_dictinary[std_scale]["n_yerr"] = left_boundary
    
    return resolution_dictinary
##############################################################################################################





try:

    """
    Define Name
    """
    N = int(sys.argv[1])
    
    logging.info("N: {}".format(N))
    
    
    
except:
    logging.info("********* Please Check Input Argunment *********")
    logging.info("********* Usage: python3 study_test_sample_statistics.py N  *********")
    sys.exit(1)

    






start = time.time()
#======================================================#
#Load Model
experiment = "dune"

model_theta23 = { }
model_delta = { }
scale_range = range(-1, 40, 1)

for i in scale_range:
    if i == -1:       
        model_theta23.update({"asimov": 0}) 
        model_delta.update({"asimov": 0}) 
    else:
        model_theta23.update({"std_scale_"+str(i): 0}) 
        model_delta.update({"std_scale_"+str(i): 0}) 
                      
        

for i, (std_scale, nth_scale) in enumerate(zip(model_theta23, scale_range)):
    
    if std_scale == "asimov" :
        file_path = "/dicos_ui_home/alanchung/ML4NO/Regression_std_Study/Model_to_5Gev/dune_theta23_asimov.h5"
    else:
        file_path = "/dicos_ui_home/alanchung/ML4NO/Regression_std_Study/Model_to_5Gev/" + str(experiment) + "_" + "theta23_std" + "_" + str(nth_scale) + "_poisson_10_.h5"
    
    if os.path.isfile(file_path):
        
#         logging.info(str(file_path) +" exists.")

        model_theta23[std_scale] = load_model(file_path)

    else:
        logging.info("Please Check Input Files!!!")
        raise ValueError("Please Check Input Files!!!")

logging.info("\n")
        
for i, (std_scale, nth_scale) in enumerate(zip(model_delta, scale_range)):
    
    if std_scale == "asimov" :
        file_path = "/dicos_ui_home/alanchung/ML4NO/Regression_std_Study/Model_to_5Gev/dune_delta_asimov_4.h5"
    else:
        file_path = "/dicos_ui_home/alanchung/ML4NO/Regression_std_Study/Model_to_5Gev/" + str(experiment) + "_" + "delta_std" + "_" + str(nth_scale) + "_poisson_10_4.h5"
    
    if os.path.isfile(file_path):
        
#         logging.info(str(file_path) +" exists.")

        model_delta[std_scale] = load_model(file_path)

    else:
        logging.info("Please Check Input Files!!!")
        raise ValueError("Please Check Input Files!!!")
        

    
if model_theta23["asimov"] != 0 and model_delta["asimov"] != 0:
    logging.info("\n")
    model_theta23["asimov"].summary()
    logging.info("\n")
    model_delta["asimov"].summary()
    
elif model_theta23["std_scale_0"] != 0 and model_delta["std_scale_0"] != 0:
    logging.info("\n")
    model_theta23["std_scale_0"].summary()
    logging.info("\n")
    model_delta["std_scale_0"].summary()

else:
    pass
    
logging.info("\n")
logging.info("All Models are loaded!")
#======================================================#



#======================================================#
test_data = np.load('../Data/sample_NuFit0911.npz')
data_mid = np.column_stack([test_data["ve_"+str(experiment)][:,:36], test_data["vu_"+str(experiment)][:,:36], test_data["vebar_"+str(experiment)][:,:36], test_data["vubar_"+str(experiment)][:,:36]])
# data_mid = np.column_stack([data['ve_dune'], data['vu_dune'], data['vebar_dune'], data['vubar_dune']])
# data_mid = np.column_stack([data['ve_dune'], data['vu_dune'], data['vebar_dune'], data['vubar_dune'],data['ve_t2hk'], data['vu_t2hk'], data['vebar_t2hk'], data['vubar_t2hk']])
data_IO_mid = data_mid[0]
data_NO_mid = data_mid[1]

logging.info("Test IO Data Shape:{}".format(data_IO_mid.shape))
logging.info("Test NO Data Shape:{}".format(data_NO_mid.shape))
#======================================================#



#======================================================#
# hf = h5py.File("./Study_test_sample_statistics/resolution.h5", 'w')


IO_or_NO = 0 # 0 for IO and 1 for NO


if IO_or_NO == 0:
    logging.info("IO")
    logging.info("True point: theta_23 = {:.2f} \delta_cp = {:.2f}".format(test_data['theta23'][0], test_data['delta'][0]))

    data_asimov = data_IO_mid.reshape(1,data_IO_mid.shape[0])
    data_poisson = random.poisson(data_IO_mid, size = (N, len(data_IO_mid)))
    ordering = "Inverse Ordering"

else:
    logging.info("NO")
    logging.info("True point: theta_23 = {:.2f} \delta_cp = {:.2f}".format(test_data['theta23'][1], test_data['delta'][1]))

    data_asimov = data_NO_mid.reshape(1,data_NO_mid.shape[0])
    data_poisson = random.poisson(data_NO_mid, size = (N, len(data_NO_mid)))
    ordering = "Normal Ordering"





theta23_resolution = {}
delta_resolution = {}

for i in scale_range:

    if i == -1:  
        theta23_resolution.update({"asimov": {"poission_prediction":0, "asimov": 0, "hist": 0, "best_fit": 0, "p_yerr": 0 , "n_yerr": 0 , },}) 
        delta_resolution.update({"asimov": {"poission_prediction":0, "asimov": 0, "hist": 0, "best_fit": 0, "p_yerr": 0 , "n_yerr": 0 , },}) 

    else:
        theta23_resolution.update({"std_scale_"+str(i): {"poission_prediction":0, "asimov": 0, "hist": 0, "best_fit": 0, "p_yerr": 0 , "n_yerr": 0 , },}) 
        delta_resolution.update({"std_scale_"+str(i): {"poission_prediction":0, "asimov": 0, "hist": 0, "best_fit": 0, "p_yerr": 0 , "n_yerr": 0 , },}) 



# bins_theta23 = np.linspace(38.9, 51.1, 1000)
bins_theta23 = np.linspace(0, 360, 30000)
theta23_resolution = Contour(theta23_resolution, model_theta23, bins_theta23, model="model_1")

bins_delta = np.linspace(0, 360, 1000)
delta_resolution = Contour(delta_resolution, model_delta, bins_delta, model="model_4")




np.savez_compressed("./Study_test_sample_statistics/theta23_resolution_" + str(N), data=theta23_resolution)
np.savez_compressed("./Study_test_sample_statistics/delta_resolution_" + str(N), data=delta_resolution)


    
#======================================================#
    



##############################################################################################################
finish = time.time()
totaltime =  start - finish
print("\033[3;33mTime Cost : {:.4f} min\033[0;m".format(totaltime/60.))