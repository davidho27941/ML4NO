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
Encoder 1 (parameter + spectrum)
"""
# parameter
encoder_parameter_inputs = layers.Input(shape=(2,),name = 'encoder_parameter_inputs')
x_parameter = layers.Dense(64, activation="relu", name = 'dense_parameter_1')(encoder_parameter_inputs)
x_parameter = layers.Dense(32, activation="relu", name = 'dense_parameter_2')(x_parameter)
x_parameter = layers.Dense(16, activation="relu", name = 'dense_parameter_3')(x_parameter)

# spectrum
encoder_spectrum_inputs = layers.Input(shape=(36,4),name = 'encoder_spectrum_inputs')
x_spectrum = layers.Conv1D(filters=5, kernel_size=10,strides=1, activation='relu', name = 'conv1d_spectrum_1')(encoder_spectrum_inputs)
x_spectrum = layers.Conv1D(filters=5, kernel_size=10,strides=1, activation='relu', name = 'conv1d_spectrum_2')(x_spectrum)
x_spectrum = layers.Flatten(name = 'flatten_spectrum')(x_spectrum)
x_spectrum = layers.Dense(16, activation="relu", name = 'dense_spectrum_1')(x_spectrum)

# merged
mergedOut_Encoder_1 = Concatenate()([x_parameter,x_spectrum])

# sampling
z_mean = layers.Dense(latent_dim, name="z_mean")(mergedOut_Encoder_1)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(mergedOut_Encoder_1)
z = Sampling(name = 'Sampling_encoder')([z_mean, z_log_var])

# build model
encoder_1 = keras.Model([encoder_parameter_inputs, encoder_spectrum_inputs], [z_mean, z_log_var, z], name="encoder_1")
encoder_1.summary()

"""
Encoder 2 (spectrum)
"""
# spectrum
encoder_spectrum_inputs = layers.Input(shape=(36,4),name = 'encoder_spectrum_inputs')
x_spectrum = layers.Conv1D(filters=5, kernel_size=10,strides=1, activation='relu', name = 'conv1d_spectrum_1')(encoder_spectrum_inputs)
x_spectrum = layers.Conv1D(filters=5, kernel_size=10,strides=1, activation='relu', name = 'conv1d_spectrum_2')(x_spectrum)
x_spectrum = layers.Flatten(name = 'flatten_spectrum')(x_spectrum)
x_spectrum = layers.Dense(16, activation="relu", name = 'dense_spectrum_1')(x_spectrum)

# sampling
z_mean = layers.Dense(latent_dim, name="z_mean")(x_spectrum)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x_spectrum)
z = Sampling(name = 'Sampling_encoder')([z_mean, z_log_var])

# build model
encoder_2 = keras.Model(encoder_spectrum_inputs, [z_mean, z_log_var, z], name="encoder_2")
encoder_2.summary()


"""
Decoder Model (latent + spectrum)
"""
latent_dim_2 = 2

decoder_latent_inputs = keras.Input(shape=(latent_dim,),name = 'decoder_latent_inputs')
x_latent = layers.Dense(64, activation="relu", name = 'dense_1')(decoder_latent_inputs)
x_latent = layers.Dense(32, activation="relu", name = 'dense_2')(x_latent)
x_latent = layers.Dense(16, activation="relu", name = 'dense_3')(x_latent)


# spectrum
decoder_spectrum_inputs = layers.Input(shape=(36,4),name = 'decoder_spectrum_inputs')
x_spectrum = layers.Conv1D(filters=5, kernel_size=10,strides=1, activation='relu', name = 'conv1d_spectrum_1')(decoder_spectrum_inputs)
x_spectrum = layers.Conv1D(filters=5, kernel_size=10,strides=1, activation='relu', name = 'conv1d_spectrum_2')(x_spectrum)
x_spectrum = layers.Flatten(name = 'flatten_spectrum')(x_spectrum)
x_spectrum = layers.Dense(16, activation="relu", name = 'dense_spectrum_1')(x_spectrum)

# merged
mergedOut_Decoder = Concatenate()([x_latent,x_spectrum])


z2_mean = layers.Dense(latent_dim_2, name="z_mean")(mergedOut_Decoder)
z2_log_var = layers.Dense(latent_dim_2, name="z_log_var")(mergedOut_Decoder)
z2 = Sampling(name = 'Sampling_decoder')([z2_mean, z2_log_var])


decoder = keras.Model([decoder_latent_inputs, decoder_spectrum_inputs], [z2_mean, z2_log_var, z2], name="decoder")
decoder.summary()





#Ref: https://keras.io/guides/customizing_what_happens_in_fit/
#Ref: https://keras.io/examples/generative/vae/
#Ref: https://github.com/hagabbar/VItamin/blob/c1ae6dfa27b8ab77193caacddd477fde0dece1c2/Models/VICI_inverse_model.py#L404
class CVAE(keras.Model):
    def __init__(self, encoder1, encoder2, decoder, **kwargs):
        super(CVAE, self).__init__(**kwargs)
        self.encoder1 = encoder1  #(parameter + spectrum)
        self.encoder2 = encoder2  #(spectrum)
        self.decoder = decoder    #(latent + spectrum)
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.reconstruction_loss_theta23_tracker = keras.metrics.Mean(name="loss_theta23_tracker")
        self.reconstruction_loss_delta_tracker = keras.metrics.Mean(name="loss_delta_tracker")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.reconstruction_loss_theta23_tracker,
            self.reconstruction_loss_delta_tracker
                ]

    def train_step(self, data):
        x, y = data
        
        with tf.GradientTape() as tape:
            z1_mean, z1_log_var, z1 = self.encoder1(x)     #(parameter + spectrum)
            
            z2_mean, z2_log_var, z2 = self.encoder2(x[1])  #(spectrum)
            
            reconstruction_mean, reconstruction_var, reconstruction = self.decoder([z1, x[1]])      #(latent + spectrum)
            
            
            reconstruction_theta23 = reconstruction[:,0]
            
            reconstruction_sin = tf.math.sin(reconstruction[:,1])
            reconstruction_sin = tf.reshape(reconstruction_sin, [reconstruction_sin.shape[0],1])
            
            reconstruction_cos = tf.math.cos(reconstruction[:,1])
            reconstruction_cos = tf.reshape(reconstruction_cos, [reconstruction_cos.shape[0],1])
            
            reconstruction_delta = tf.concat([reconstruction_sin, reconstruction_cos], 1)
                                                                 
                                                                 
            mse = tf.keras.losses.MeanSquaredError()
            
            reconstruction_loss_theta23 = tf.reduce_mean(
                                            tf.reduce_sum(
                                                        mse(y[0], reconstruction_theta23 )
                                                            )
                                            )
            reconstruction_loss_delta = tf.reduce_mean(
                                            tf.reduce_sum(
                                                        mse(y[1], reconstruction_delta )
                                                            )
                                            )
            

            
            reconstruction_loss = reconstruction_loss_theta23*reconstruction_loss_delta
        
            SMALL_CONSTANT = 1e-12 # necessary to prevent the division by zero in many operations 
            GAUSS_RANGE = 10.0     # Actual range of truncated gaussian when the ramp is 0
            
            # define the r1(z|y) mixture model
            temp_var_r1 = SMALL_CONSTANT + tf.exp(z2_log_var)
            bimix_gauss = tfp.distributions.MultivariateNormalDiag(
                          loc=z2_mean,
                          scale_diag=tf.sqrt(temp_var_r1))
            
            # GET q(z|x,y)
            temp_var_q = SMALL_CONSTANT + tf.exp(z1_log_var)
            mvn_q = tfp.distributions.MultivariateNormalDiag(
                          loc=z1_mean,
                          scale_diag=tf.sqrt(temp_var_q))
            

            
            
            q_samp = mvn_q.sample()  
            log_q_q = mvn_q.log_prob(q_samp)
            log_r1_q = bimix_gauss.log_prob(q_samp)           # evaluate the log prob of r1 at the q samples
            kl_loss = tf.reduce_mean(log_q_q - log_r1_q)      # average over batch
            
        
            total_loss = reconstruction_loss + kl_loss
            
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.reconstruction_loss_theta23_tracker.update_state(reconstruction_loss_theta23)
        self.reconstruction_loss_delta_tracker.update_state(reconstruction_loss_delta)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "theta23_loss": self.reconstruction_loss_theta23_tracker.result(),
            "delta_loss": self.reconstruction_loss_delta_tracker.result(),
        }




#======================================================#





"""
Model Training
"""
cvae = CVAE(encoder_1,encoder_2, decoder)
cvae.compile(optimizer=keras.optimizers.Adam())
# vae.compile(optimizer=keras.optimizers.Adadelta())

check_list=[]
# checkpoint = ModelCheckpoint(filepath= "./CVAE_1DCNN_checkmodel.h5",
#                             save_best_only=True,
#                             verbose=0)
csv_logger = CSVLogger("./Training_loss/CVAE_1DCNN_training_log_v1.csv")
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
check_list.append(csv_logger)
# check_list.append(earlystopping)




cvae.fit( x = [y_train, x_train_poisson],
         y = [y_train_theta23, y_train_delta],
#            validation_split = 0.1,
           batch_size=50,
           epochs=200,
           verbose=1,
#            shuffle = True,
           callbacks=check_list
         )

cvae.encoder1.save("./Model/CVAE_1DCNN_encoder_1_v1.h5")
cvae.encoder2.save("./Model/CVAE_1DCNN_encoder_2_v1.h5")
cvae.decoder.save("./Model/CVAE_1DCNN_decoder_v1.h5")


##############################################################################################################
finish = time.time()

totaltime =  finish - start
logging.info("\033[3;33mTime Cost : {:.4f} min\033[0;m".format(totaltime/60.))