#!/bin/python3
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


import numpy as np
import matplotlib.pyplot as plt
# import autokeras as ak
import os 
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
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5000)])
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
#######################################################


def Regression_Model(num_of_bins):

    input_shape = (num_of_bins,)
    model = Sequential(name = 'Regression_Model_theta23')
    model.add(BatchNormalization(input_shape=input_shape, name = 'BatchNormalization'))
    model.add(Dense(512, activation='relu', name = 'dense_1'))
    model.add(Dense(512, activation='relu', name = 'dense_2'))
    model.add(Dense(1024, activation='relu', name = 'dense_3'))
    model.add(Dense(1, activation='relu', name = 'theta23'))
    
    
#     model_opt = keras.optimizers.Adadelta()
    model_opt = keras.optimizers.Adam()
    
    
    model.compile(loss="mean_squared_error",
                       optimizer=model_opt,
                       metrics=["mse"])
    
    return model



# Load Data
data = np.load("../Data/n1000000_0910_all_flat.npz")


# Stack Data
# data_all = np.column_stack([data['ve_dune'], data['vu_dune'], data['vebar_dune'], data['vubar_dune']])
data_all = np.column_stack([data['ve_t2hk'], data['vu_t2hk'], data['vebar_t2hk'], data['vubar_t2hk']])
# data_all = np.column_stack([data['ve_dune'], data['vu_dune'], data['vebar_dune'], data['vubar_dune'],data['ve_t2hk'], data['vu_t2hk'], data['vebar_t2hk'], data['vubar_t2hk']])
target = data['theta23']

x_train = data_all[:10000]
y_train = target[:10000]

x_train2 = data_all[10000:900000]
y_train2 = target[10000:900000]

x_test = data_all[900000:]
y_test = target[900000:]



# # Find Best Model by AutoKeras
# clf = ak.StructuredDataRegressor(overwrite=True, max_trials=50)
# clf.fit(x_train, y_train,
#            validation_split = 0.1,
#            batch_size=64,
#            epochs=20,
#            verbose=1,
#            shuffle = True
#        )


# # Export Model
# model = clf.export_model()
# model.summary()



model = Regression_Model(len(x_train[0]))
model.summary()

# Test on test data set
model.fit(x_train2, y_train2,
           validation_split = 0.1,
           batch_size=64,
           epochs=20,
           verbose=1,
           shuffle = True
         )



# Save the Best Model
index = 1
while os.path.isfile("./models/t2hk_theta23_{}.h5".format(index)):
    index += 1
model.save("./models/t2hk_theta23_{}.h5".format(index))



# Adding Fluctuation and Further Train the Model
scale_steps = np.logspace(-3, 0, 30)
before_train_loss = []
after_train_loss = []

for scale in tqdm(scale_steps):
    time.sleep(0.5)
    x_train2_gen = np.random.normal(x_train2, np.sqrt(x_train2)*scale)
    x_test_gen = np.random.normal(x_test, np.sqrt(x_test)*scale)

    before_train_loss.append(model.evaluate(x_test_gen, y_test)[0])

    model.fit(x_train2_gen, y_train2,
               validation_split = 0.1,
               batch_size=64,
               epochs=5,
               verbose=1,
               shuffle = True
             )

    after_train_loss.append(model.evaluate(x_test_gen, y_test)[0])
    
    
    
    
# Save the Futher Training Model
model_index = index
index = 1
path = "./models_furthurTrain/t2hk_theta23_{}_{}.h5"
while os.path.isfile(path.format(model_index, index)):
    index += 1
model.save(path.format(model_index, index))
outfile = {'scale_steps': scale_steps,
           'before_train_loss': before_train_loss,
           'after_train_loss' :after_train_loss}
np.save(file = "./models_furthurTrain/t2hk_theta23_{}_{}_result.npy".format(model_index, index),
        arr = outfile)



# Adding Poission Fluctuation and Further Train the Model
x_test2_gen = np.random.poisson(x_test)

for i in tqdm(range(10)):
    time.sleep(0.5)
    x_train2_gen = np.random.poisson(x_train2)
    
    model.fit(x_train2_gen, y_train2,
              validation_split=0.1,
               batch_size=64,
               epochs=1,
               verbose=1,
               shuffle = True
             )
model.evaluate(x_test2_gen, y_test)


    
    
# Save the Last Model
furthur_index = index
index = 1
path = "./models_PoissonTrain/t2hk_theta23_{}_{}_{}.h5"
while os.path.isfile(path.format(model_index, furthur_index, index)):
    index += 1
model.save(path.format(model_index, furthur_index, index))



#######################################################
finish = time.time()

totaltime =  finish - start
logging.info("\033[3;33mTime Cost : {:.4f} min\033[0;m".format(totaltime/60.))