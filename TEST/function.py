#!/usr/bin/python3
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
import logging

# From Cartesian coordinates transform to polar coordinates
def angle_transform(prediction_array: np.ndarray = 0)-> np.ndarray :
    tmp = np.arctan2(prediction_array[:,0],prediction_array[:,1])
    tmp = np.where(tmp >= 0 , tmp, 2*np.pi+tmp) 
    delta = tmp/np.pi*180 
    
    return delta






def load_data(path: str ="../Data/n1000000_0910_all_flat.npz" , model : str = "VAE_1DCNN_v1" )-> np.ndarray :
    
    
    def feature_norm(x : np.ndarray) -> np.ndarray : 
        if x.shape[1] != 144:
            raise ValueError("Check size of x should be (N,144)")
        x_normalized = np.column_stack([x[:,:36]/169.042, x[:,36:72]/787.425, x[:,72:108]/78.6681, x[:,108:144]/334.052])
        return x_normalized

    def feature_reshape(data,column=2):
        tmp = data.reshape(len(data),column,36)
        tmp = np.array([ element.T for element in tmp])
        return tmp
    
    # Load Data
    training_data = np.load(path)

    # Stack Data
    #======================================================#
    

    #VAE_1DCNN_v1
    if model == "VAE_1DCNN_v1":
        # Stack Data
        data_all = np.column_stack([training_data['ve_dune'][:,:36], training_data['vu_dune'][:,:36], training_data['vebar_dune'][:,:36], training_data['vubar_dune'][:,:36]])
        
        """
        Standardization
        """
        scaler = StandardScaler()
        scaler.fit(data_all)
    
        target = np.column_stack( [training_data["delta"]/180*np.pi , training_data["theta23"]])

        x_train, y_train = data_all[:900000], target[:900000]
        y_train = np.column_stack([y_train[:,1], np.sin(y_train[:,0]), np.cos(y_train[:,0])])

        x_test, y_test = data_all[900000:], target[900000:]
        y_test = np.column_stack([ y_test[:,1], np.sin(y_test[:,0]), np.cos(y_test[:,0])])


        logging.info("X train/test shape: {} / {}".format(x_train.shape, x_test.shape))
        if type(y_train) == list:
            for i in range(len(y_train)):
                logging.info("Y train["+str(i)+"]/test["+str(i)+"] shape {} / {}".format(y_train[i].shape,y_test[i].shape))
        else:
            logging.info("Y train/test shape: {} / {}".format(y_train.shape,y_test.shape))

        x_train_poisson = np.random.poisson(x_train)
#         x_train_poisson = feature_norm(x_train_poisson)
        x_train_poisson = scaler.transform(x_train_poisson)
        x_train_poisson = feature_reshape(x_train_poisson,column=4)
#         x_train_poisson = x_train_poisson.reshape(len(x_train_poisson),4,36)
#         x_train_poisson = np.array([ element.T for element in x_train_poisson])


        x_test_poisson = np.random.poisson(x_test)
#         x_test_poisson = feature_norm(x_test_poisson)
        x_test_poisson = scaler.transform(x_test_poisson)
        x_test_poisson = feature_reshape(x_test_poisson,column=4)
#         x_test_poisson = x_test_poisson.reshape(len(x_test_poisson),4,36)
#         x_test_poisson = np.array([ element.T for element in x_test_poisson])

        logging.info("\n")
        logging.info("x_train_poisson shape: {}".format(x_train_poisson.shape))
        logging.info("x_test_poisson shape: {}".format(x_test_poisson.shape))
        logging.info("\n")
    
    #VAE_1DCNN_v2
    elif model == "VAE_1DCNN_v2":
        # Stack Data
        data_all = np.column_stack([training_data['ve_dune'][:,:36], training_data['vu_dune'][:,:36], training_data['vebar_dune'][:,:36], training_data['vubar_dune'][:,:36]])

        
        """
        Standardization
        """
        scaler = StandardScaler()
        scaler.fit(data_all)
    
        target = np.column_stack( [training_data["delta"]/180*np.pi , training_data["theta23"]])

        x_train, y_train = data_all[:900000], target[:900000]
        y_train_delta = np.column_stack([np.sin(y_train[:,0]), np.cos(y_train[:,0])]) 
        y_train_theta23 = y_train[:,1]
        y_train = [y_train_theta23, y_train_delta]

        x_test, y_test = data_all[900000:], target[900000:]
        y_test_delta = np.column_stack([np.sin(y_test[:,0]), np.cos(y_test[:,0])]) 
        y_test_theta23 = y_test[:,1]
        y_test = [y_test_theta23, y_test_delta]

        logging.info("X train/test shape: {} / {}".format(x_train.shape, x_test.shape))
        if type(y_train) == list:
            for i in range(len(y_train)):
                logging.info("Y train["+str(i)+"]/test["+str(i)+"] shape {} / {}".format(y_train[i].shape,y_test[i].shape))
        else:
            logging.info("Y train/test shape: {} / {}".format(y_train.shape,y_test.shape))

        x_train_poisson = np.random.poisson(x_train)
#         x_train_poisson = feature_norm(x_train_poisson)
        x_train_poisson = scaler.transform(x_train_poisson)
        x_train_poisson = feature_reshape(x_train_poisson,column=4)
#         x_train_poisson = x_train_poisson.reshape(len(x_train_poisson),4,36)
#         x_train_poisson = np.array([ element.T for element in x_train_poisson])

        
        x_test_poisson = np.random.poisson(x_test)
#         x_test_poisson = feature_norm(x_test_poisson)
        x_test_poisson = scaler.transform(x_test_poisson)
        x_test_poisson = feature_reshape(x_test_poisson,column=4)
#         x_test_poisson = x_test_poisson.reshape(len(x_test_poisson),4,36)
#         x_test_poisson = np.array([ element.T for element in x_test_poisson])

        logging.info("\n")
        logging.info("x_train_poisson shape: {}".format(x_train_poisson.shape))
        logging.info("x_test_poisson shape: {}".format(x_test_poisson.shape))
        logging.info("\n")
        

        
    #Regression_Model_Fully_Connected_Dense_std   
    elif model == "Regression_Model_Fully_Connected_Dense_std":
        # Stack Data
        data_all = np.column_stack([training_data['ve_dune'][:,:36], training_data['vu_dune'][:,:36], training_data['vebar_dune'][:,:36], training_data['vubar_dune'][:,:36]])
        
        
        """
        Standardization
        """
        scaler = StandardScaler()
        scaler.fit(data_all)
        
        target = np.column_stack( [training_data['theta23'] , training_data["delta"]/180*np.pi])

        x_train, y_train = data_all[:900000], target[:900000]
        y_train_theta23 = y_train[:,0]
        y_train_delta = np.column_stack([np.sin(y_train[:,1]), np.cos(y_train[:,1])]) 
        y_train = [y_train_theta23, y_train_delta]

        x_test, y_test = data_all[900000:], target[900000:]
        y_test_theta23 = y_test[:,0]
        y_test_delta = np.column_stack([np.sin(y_test[:,1]), np.cos(y_test[:,1])]) 
        y_test = [y_test_theta23, y_test_delta]

        
        logging.info("X train/test shape: {} / {}".format(x_train.shape, x_test.shape))
        if type(y_train) == list:
            for i in range(len(y_train)):
                logging.info("Y train["+str(i)+"]/test["+str(i)+"] shape {} / {}".format(y_train[i].shape,y_test[i].shape))
        else:
            logging.info("Y train/test shape: {} / {}".format(y_train.shape,y_test.shape))
        

        x_train_poisson = np.random.poisson(x_train)
        x_test_poisson = np.random.poisson(x_test)
        
        x_train_poisson = scaler.transform(x_train_poisson)
        x_test_poisson = scaler.transform(x_test_poisson)

        logging.info("\n")
        logging.info("x_train_poisson shape: {}".format(x_train_poisson.shape))
        logging.info("x_test_poisson shape: {}".format(x_test_poisson.shape))
        logging.info("\n")
        
        
    #Regression_Model_Fully_Connected_Dense_app_dis   
    elif model == "Regression_Model_Fully_Connected_Dense_app_dis":
        # Stack Data
        data_all = np.column_stack([training_data['ve_dune'][:,:36], training_data['vebar_dune'][:,:36], training_data['vu_dune'][:,:36],training_data['vubar_dune'][:,:36]])
        
        
        """
        Standardization
        """
        scaler = StandardScaler()
        scaler.fit(data_all)
        
        target = np.column_stack( [training_data['theta23'] , training_data["delta"]/180*np.pi])

        x_train, y_train = data_all[:900000], target[:900000]
        y_train_theta23 = y_train[:,0]
        y_train_delta = np.column_stack([np.sin(y_train[:,1]), np.cos(y_train[:,1])]) 
        y_train = [y_train_theta23, y_train_delta]

        x_test, y_test = data_all[900000:], target[900000:]
        y_test_theta23 = y_test[:,0]
        y_test_delta = np.column_stack([np.sin(y_test[:,1]), np.cos(y_test[:,1])]) 
        y_test = [y_test_theta23, y_test_delta]

        
        logging.info("X train/test shape: {} / {}".format(x_train.shape, x_test.shape))
        if type(y_train) == list:
            for i in range(len(y_train)):
                logging.info("Y train["+str(i)+"]/test["+str(i)+"] shape {} / {}".format(y_train[i].shape,y_test[i].shape))
        else:
            logging.info("Y train/test shape: {} / {}".format(y_train.shape,y_test.shape))
        

        x_train_poisson = np.random.poisson(x_train)
        x_test_poisson = np.random.poisson(x_test)
        
        x_train_poisson = scaler.transform(x_train_poisson)
        x_test_poisson = scaler.transform(x_test_poisson)
        
        
        appearance_x_train_poisson = x_train_poisson[:,:72]
        appearance_x_train_poisson = appearance_x_train_poisson.reshape(len(appearance_x_train_poisson),2,36)
        appearance_x_train_poisson = np.array([ element.T for element in appearance_x_train_poisson])

        disappearance_x_train_poisson = x_train_poisson[:,72:]
        disappearance_x_train_poisson = disappearance_x_train_poisson.reshape(len(disappearance_x_train_poisson),2,36)
        disappearance_x_train_poisson = np.array([ element.T for element in disappearance_x_train_poisson])
        
        
        appearance_x_test_poisson = x_test_poisson[:,:72]
        appearance_x_test_poisson = appearance_x_test_poisson.reshape(len(appearance_x_test_poisson),2,36)
        appearance_x_test_poisson = np.array([ element.T for element in appearance_x_test_poisson])

        disappearance_x_test_poisson = x_test_poisson[:,72:]
        disappearance_x_test_poisson = disappearance_x_test_poisson.reshape(len(disappearance_x_test_poisson),2,36)
        disappearance_x_test_poisson = np.array([ element.T for element in disappearance_x_test_poisson])

        
        x_train_poisson = [appearance_x_train_poisson, disappearance_x_train_poisson]
        x_test_poisson = [appearance_x_test_poisson, disappearance_x_test_poisson]
        
        
        logging.info("\n")
        if type(x_train_poisson) == list:
            for i in range(len(x_train_poisson)):
                logging.info("X train["+str(i)+"]/test["+str(i)+"] shape {} / {}".format(x_train_poisson[i].shape,x_test_poisson[i].shape))
        else:
            logging.info("X train/test shape: {} / {}".format(x_train_poisson.shape,x_test_poisson.shape))
        logging.info("\n")
        
        
    
        
    return x_train_poisson, y_train, x_test_poisson, y_test







# Load Model and make the prediction
def prediction_function(x_test : np.ndarray = 0, model : str = "VAE_1DCNN_v1")-> np.ndarray:
    
    
    #VAE_1DCNN_v1
    if model == "VAE_1DCNN_v1":

        class Sampling(layers.Layer):
            """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

            def call(self, inputs):
                z_mean, z_log_var = inputs
                batch = tf.shape(z_mean)[0]
                dim = tf.shape(z_mean)[1]
                epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
                return z_mean + tf.exp(0.5 * z_log_var) * epsilon


        vae_encoder = load_model("./Model/VAE_1DCNN_encoder_v1.h5", compile=False, custom_objects={"Sampling": Sampling})
        vae_decoder = load_model("./Model/VAE_1DCNN_decoder_v1.h5", compile=False)
        z = vae_encoder.predict(x_test)
        prediction = vae_decoder.predict(z[0])
        
        if type(prediction) == list:
            for i in range(len(prediction)):
                logging.info("prediction["+str(i)+"] shape {}".format(prediction[i].shape))
        else:
            logging.info("prediction shape {}".format(prediction.shape))
        
    elif model == "VAE_1DCNN_v2":

        class Sampling(layers.Layer):
            """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

            def call(self, inputs):
                z_mean, z_log_var = inputs
                batch = tf.shape(z_mean)[0]
                dim = tf.shape(z_mean)[1]
                epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
                return z_mean + tf.exp(0.5 * z_log_var) * epsilon


        vae_encoder = load_model("./Model/VAE_1DCNN_encoder_v2.h5", compile=False, custom_objects={"Sampling": Sampling})
        vae_decoder = load_model("./Model/VAE_1DCNN_decoder_v2.h5", compile=False)
        z = vae_encoder.predict(x_test)
        prediction = vae_decoder.predict(z[0])
        
        if type(prediction) == list:
            for i in range(len(prediction)):
                logging.info("prediction["+str(i)+"] shape {}".format(prediction[i].shape))
        else:
            logging.info("prediction shape {}".format(prediction.shape))

            
            
    #Regression_Model_Fully_Connected_Dense_std 
    elif model == "Regression_Model_Fully_Connected_Dense_std":

        model = load_model("./Model/poisson_20.h5", compile=False)
        prediction = model.predict(x_test)
        
        
        if type(prediction) == list:
            for i in range(len(prediction)):
                logging.info("prediction["+str(i)+"] shape {}".format(prediction[i].shape))
        else:
            logging.info("prediction shape {}".format(prediction.shape))
            
    
    #Regression_Model_Fully_Connected_Dense_app_dis 
    elif model == "Regression_Model_Fully_Connected_Dense_app_dis":

        model = load_model("./Model/poisson_app_dis.h5", compile=False)
        prediction = model.predict(x_test)
        
        
        if type(prediction) == list:
            for i in range(len(prediction)):
                logging.info("prediction["+str(i)+"] shape {}".format(prediction[i].shape))
        else:
            logging.info("prediction shape {}".format(prediction.shape))
            
        
        
    return prediction






def load_test_data(path: str ="../Data/n1000000_0910_all_flat.npz" , model : str = "VAE_1DCNN_v1", IO_or_NO : int = 0 , N : int = 10000 )-> np.ndarray :
    
    
    def feature_norm(x : np.ndarray) -> np.ndarray : 
        if x.shape[1] != 144:
            raise ValueError("Check size of x should be (N,144)")
        x_normalized = np.column_stack([x[:,:36]/169.042, x[:,36:72]/787.425, x[:,72:108]/78.6681, x[:,108:144]/334.052])
        return x_normalized

    def feature_reshape(data,column=2):
        tmp = data.reshape(len(data),column,36)
        tmp = np.array([ element.T for element in tmp])
        return tmp
    
    # Load Data
    training_data = np.load(path)
    
    
    
    #Regression_Model_Fully_Connected_Dense_std
    if model == "Regression_Model_Fully_Connected_Dense_std":
        # Stack Data
        data_all = np.column_stack([training_data['ve_dune'][:,:36], training_data['vu_dune'][:,:36], training_data['vebar_dune'][:,:36], training_data['vubar_dune'][:,:36]])
        
        
        """
        Standardization
        """
        scaler = StandardScaler()
        scaler.fit(data_all)
        
    
    #Regression_Model_Fully_Connected_Dense_app_dis
    elif model == "Regression_Model_Fully_Connected_Dense_app_dis":
        # Stack Data
        data_all = np.column_stack([training_data['ve_dune'][:,:36], training_data['vebar_dune'][:,:36], training_data['vu_dune'][:,:36],training_data['vubar_dune'][:,:36]])
        
        
        """
        Standardization
        """
        scaler = StandardScaler()
        scaler.fit(data_all)
    
    

    test_data = np.load('../Data/sample_NuFit0911.npz')



    
    #IO_or_NO : 0 for NO and 1 for IO
    if IO_or_NO == 0:
        logging.info("NO")
        logging.info("True point: theta_23 = {:.2f} \delta_cp = {:.2f}".format(test_data['theta23'][0], test_data['delta'][0]))
        true_theta_23, trua_delta = test_data['theta23'][0], test_data['delta'][0]

        
        #Regression_Model_Fully_Connected_Dense_std
        if model == "Regression_Model_Fully_Connected_Dense_std": 
            data_mid = np.column_stack([test_data["ve_dune"][:,:36],  test_data["vu_dune"][:,:36], test_data["vebar_dune"][:,:36], test_data["vubar_dune"][:,:36]])
            data_NO_mid = data_mid[0]
            logging.info("Test NO Data Shape:{}".format(data_NO_mid.shape))

            data_poisson = np.random.poisson(data_NO_mid, size = (N, len(data_NO_mid)))
            data_poisson = scaler.transform(data_poisson)
            test_data = data_poisson
            
            if type(test_data) == list:
                for i in range(len(test_data)):
                        logging.info("X ["+str(i)+"] shape {}".format(test_data[i].shape))
            else:
                logging.info("X train/test shape: {}".format(test_data.shape))
            logging.info("\n")
            
            
        #Regression_Model_Fully_Connected_Dense_app_dis
        elif model == "Regression_Model_Fully_Connected_Dense_app_dis": 

            data_mid = np.column_stack([test_data["ve_dune"][:,:36], test_data["vebar_dune"][:,:36], test_data["vu_dune"][:,:36], test_data["vubar_dune"][:,:36]])
            data_NO_mid = data_mid[0]
            logging.info("Test NO Data Shape:{}".format(data_NO_mid.shape))

            data_poisson = np.random.poisson(data_NO_mid, size = (N, len(data_NO_mid)))
            data_poisson = scaler.transform(data_poisson)
            appearance_poisson = feature_reshape(data_poisson[:,:72],column=2)
            disappearance_poisson = feature_reshape(data_poisson[:,72:],column=2)
            
            test_data = [appearance_poisson, disappearance_poisson]
            
            if type(test_data) == list:
                for i in range(len(test_data)):
                        logging.info("X ["+str(i)+"] shape {}".format(test_data[i].shape))
            else:
                logging.info("X train/test shape: {}".format(test_data.shape))
            logging.info("\n")
            
            
    else:
        logging.info("IO")
        logging.info("True point: theta_23 = {:.2f} \delta_cp = {:.2f}".format(test_data['theta23'][1], test_data['delta'][1]))
        true_theta_23, trua_delta = test_data['theta23'][1], test_data['delta'][1]

        
        #Regression_Model_Fully_Connected_Dense_std
        if model == "Regression_Model_Fully_Connected_Dense_std": 
            data_mid = np.column_stack([test_data["ve_dune"][:,:36],  test_data["vu_dune"][:,:36], test_data["vebar_dune"][:,:36],test_data["vubar_dune"][:,:36]])
            data_IO_mid = data_mid[1]
            logging.info("Test IO Data Shape:{}".format(data_IO_mid.shape))

            data_poisson = np.random.poisson(data_IO_mid, size = (N, len(data_IO_mid)))
            data_poisson = scaler.transform(data_poisson)
            test_data = data_poisson
            
            if type(test_data) == list:
                for i in range(len(test_data)):
                        logging.info("X ["+str(i)+"] shape {}".format(test_data[i].shape))
            else:
                logging.info("X train/test shape: {}".format(test_data.shape))
            logging.info("\n")
        
        
        
        
        #Regression_Model_Fully_Connected_Dense_app_dis
        elif model == "Regression_Model_Fully_Connected_Dense_app_dis": 

            data_mid = np.column_stack([test_data["ve_dune"][:,:36], test_data["vebar_dune"][:,:36], test_data["vu_dune"][:,:36], test_data["vubar_dune"][:,:36]])
            
            data_IO_mid = data_mid[1]
            logging.info("Test IO Data Shape:{}".format(data_IO_mid.shape))

            data_poisson = np.random.poisson(data_IO_mid, size = (N, len(data_IO_mid)))
            data_poisson = scaler.transform(data_poisson)
            appearance_poisson = feature_reshape(data_poisson[:,:72],column=2)
            disappearance_poisson = feature_reshape(data_poisson[:,72:],column=2)
            
            test_data = [appearance_poisson, disappearance_poisson]
            
            if type(test_data) == list:
                for i in range(len(test_data)):
                        logging.info("X ["+str(i)+"] shape {}".format(test_data[i].shape))
            else:
                logging.info("X test shape: {} ".format(test_data.shape))
            logging.info("\n")
            
            
    return true_theta_23, trua_delta, test_data




# Accuracy Study
def accuracy_study_sample(IO_or_NO : int = 0, model : str = "VAE_1DCNN_v1" )-> np.ndarray :
    
    training_data = np.load("../Data/n1000000_0910_all_flat.npz")
    
    # 0 for NO and 1 for IO

    if IO_or_NO == 0:
        logging.info("NO")
        mc_test_data = np.load('../Data/best_fit_spectrum_deltacp_theta23_DUNE_NO.npz')

    elif IO_or_NO == 1:
        logging.info("IO")
        mc_test_data = np.load('../Data/best_fit_spectrum_deltacp_theta23_DUNE_IO.npz')

    theta23_true = mc_test_data["theta23_true"]/(2*np.pi)*360
    delta_true = mc_test_data["delta_true"]
    theta23_best_fit = mc_test_data["theta23_fit"]/(2*np.pi)*360
    delta_best_fit = mc_test_data["delta_fit"]

    parameter_range = np.where(np.logical_and(delta_true >= 0, delta_true <= 360 ))[0]#[:100]


    theta23_true = theta23_true[parameter_range]
    delta_true = delta_true[parameter_range]
    theta23_best_fit = theta23_best_fit[parameter_range]
    delta_best_fit = delta_best_fit[parameter_range]

    
    
    if model == "Regression_Model_Fully_Connected_Dense_std":
        # Stack Data
        data_all = np.column_stack([training_data['ve_dune'][:,:36], training_data['vu_dune'][:,:36], training_data['vebar_dune'][:,:36], training_data['vubar_dune'][:,:36]])
        
        """
        Standardization
        """
        scaler = StandardScaler()
        scaler.fit(data_all)
        
        mc_data = np.column_stack([mc_test_data["ve_dune_poisson"][:,1:37],
                               mc_test_data["vu_dune_poisson"][:,1:37],
                               mc_test_data["vebar_dune_poisson"][:,1:37],
                               mc_test_data["vubar_dune_poisson"][:,1:37]])

        mc_data = mc_data[parameter_range]
        mc_data = scaler.transform(mc_data)

        logging.info("MC Data Shape:{}".format(mc_data.shape))
        logging.info("theta23 true Shape:{}".format(theta23_true.shape))
        logging.info("delta true Shape:{}".format(delta_true.shape))
        logging.info("theta23 best fit Shape:{}".format(theta23_best_fit.shape))
        logging.info("delta best fit Shape:{}".format(delta_best_fit.shape))
        
        
    return mc_data, theta23_true, delta_true, theta23_best_fit, delta_best_fit