#!/usr/bin/python3
# Install TensorFlow
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras import layers
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
# ##############################################################################################################

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
data_all = np.column_stack([training_data['ve_dune'][:,:36], training_data['vu_dune'][:,:36], training_data['vebar_dune'][:,:36], training_data['vubar_dune'][:,:36]])


"""
Standardization
"""
scaler = StandardScaler()
scaler.fit(data_all)

target = np.column_stack( [training_data["theta23"], training_data["delta"]/180*np.pi ])
# target = target/180*np.pi 

x_train = data_all[:900000]
y_train = target[:900000]
y_train_delta = np.column_stack([np.sin(y_train[:,1]), np.cos(y_train[:,1])]) 
y_train_theta23 = y_train[:,0]



x_test = data_all[900000:]
y_test = target[900000:]
y_test_delta = np.column_stack([np.sin(y_test[:,1]), np.cos(y_test[:,1])]) 
y_test_theta23 = y_test[:,0]


logging.info("X train shape: {}".format(x_train.shape))
logging.info("X test shape: {}".format(x_test.shape))
logging.info("Y train shape: {}".format(y_train.shape))
logging.info("Y test shape: {}".format(y_test.shape))


x_test_poisson = np.random.poisson(x_test)
x_test_poisson = scaler.transform(x_test_poisson) 
x_test_poisson = x_test_poisson.reshape(len(x_test_poisson),4,36)
x_test_poisson = np.array([ element.T for element in x_test_poisson])

logging.info("\n")
# logging.info("x_train_poisson shape: {}".format(x_train_poisson.shape))
logging.info("x_test_poisson shape: {}".format(x_test_poisson.shape))
logging.info("\n")
#======================================================#


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
y_train_tmp = y_train
for i in tqdm(range(10)):
    x_train_poisson_tmp = np.random.poisson(x_train)
    x_train_poisson_tmp = scaler.transform(x_train_poisson_tmp) 


    x_train_poisson = np.concatenate([x_train_poisson,x_train_poisson_tmp])
    y_train_theta23 = np.concatenate([y_train_theta23,y_train_theta23_tmp])
    y_train_delta = np.concatenate([y_train_delta,y_train_delta_tmp])
    y_train = np.concatenate([y_train,y_train_tmp])
    
x_train_poisson = x_train_poisson.reshape(len(x_train_poisson),4,36)
x_train_poisson = np.array([ element.T for element in x_train_poisson])
    
logging.info("x_train_poisson.shape : {}".format(x_train_poisson.shape))
logging.info("y_train_theta23.shape : {}".format(y_train_theta23.shape))
logging.info("y_train_delta.shape : {}".format(y_train_delta.shape))
logging.info("y_train.shape : {}".format(y_train.shape))

t2_time = time.time()
logging.info("\033[3;33m Time Cost for this Step : {:.4f} min\033[0;m".format((t2_time-t1_time)/60.))
logging.info("=====Finish=====")
logging.info("\n")

#======================================================#

"""
Define Model
"""
#======================================================#
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    

latent_dim = 15
logging.info("latent_dim : {}".format(latent_dim))


"""
Encoder Model
"""
encoder_inputs = layers.Input(shape=(36,4),name = 'encoder_inputs')

x = layers.Conv1D(filters=5, kernel_size=10,strides=1, activation='relu', name = 'conv1d_1')(encoder_inputs)
x = layers.Conv1D(filters=5, kernel_size=10,strides=1, activation='relu', name = 'conv1d_2')(x)
x = layers.Flatten(name = 'flatten')(x)
x = layers.Dense(16, activation="relu", name = 'dense_1')(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()


"""
Decoder Model
"""
latent_dim_2 = 2

latent_inputs = keras.Input(shape=(latent_dim,),name = 'latent_inputs')
x = layers.Dense(64, activation="relu", name = 'dense_1')(latent_inputs)
x = layers.Dense(32, activation="relu", name = 'dense_2')(x)
x = layers.Dense(16, activation="relu", name = 'dense_3')(x)
z2_mean = layers.Dense(latent_dim_2, name="z2_mean")(x)
z2_log_var = layers.Dense(latent_dim_2, name="z2_log_var")(x)
z2 = Sampling(name = 'Sampling_decoder')([z2_mean, z2_log_var])
decoder = keras.Model(latent_inputs, [z2_mean, z2_log_var, z2], name="decoder")
decoder.summary()





#Ref: https://keras.io/guides/customizing_what_happens_in_fit/
#Ref: https://keras.io/examples/generative/vae/
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        x, y = data
        
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x)
            
            reconstruction_mean, reconstruction_var, reconstruction  = self.decoder(z)
            mse = tf.keras.losses.MeanSquaredError()
            
            reconstruction_loss = tf.reduce_mean(
                                        tf.reduce_sum(
                                                        mse(y, reconstruction)
                                                    )
                                    )
            
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


#======================================================#





"""
Model Training
"""
vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())
# vae.compile(optimizer=keras.optimizers.Adadelta())


# check_list=[]
# checkpoint = ModelCheckpoint(filepath= "./Model/VAE_1DCNN_checkmodel.h5",
#                             save_best_only=True,
#                             verbose=0)
# csv_logger = CSVLogger("./Training_loss/VAE_1DCNN_training_log.csv")
# earlystopping = EarlyStopping(
#                     monitor="val_loss",
#                     min_delta=0,
#                     patience=20,
#                     verbose=0,
#                     mode="auto",
#                     baseline=None,
#                     restore_best_weights=False,
#                 )

# check_list.append(checkpoint)
# check_list.append(csv_logger)
# check_list.append(earlystopping)

vae.fit( x = x_train_poisson,
         y = y_train,
#            validation_split = 0.1,
           batch_size=64,
           epochs=200,
           verbose=1,
#            shuffle = True,
#            callbacks=check_list
         )

vae.encoder.save("./Model/VAE_1DCNN_encoder_v4.h5")
vae.decoder.save("./Model/VAE_1DCNN_decoder_v4.h5")


##############################################################################################################
finish = time.time()

totaltime =  finish - start
logging.info("\033[3;33mTime Cost : {:.4f} min\033[0;m".format(totaltime/60.))