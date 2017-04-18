# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 21:59:22 2017

@author: Chris
"""

from sklearn.metrics import mean_squared_error
import numpy as np
import Utils as f
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
from time import time

#RMSE function
def RMSE(y,y_pred):
    return mean_squared_error(y, y_pred)**0.5

#function to cap predictions on a scale of 1 to 3
def post_process_preds(y_pred):
    return np.array([max(min(y[0],3.0),1.0) for y in y_pred])
    
    
print('Attempting to load pre-processed data...')
try:
    #load pre-processed training data
    X = f.load_obj('input_clean/X_train')
    Y = f.load_obj('input_clean/Y_train')
    print('Pre-processed data successfully loaded...')
except:
    print('Pre-processed data failed to load, ensure the working directory is correct...')
    
#dataframe to store random search results
try: 
    results_df = pd.read_csv('output/NeuralNetworkOptimizationV2.csv',index_col='Round')
    print('Previous optimization results successfully loaded...')
except:
    print('Previous optimization results failed to load, starting new optimization experiment...')
    results_df = pd.DataFrame(columns = ['batch_size',
                                         'num_hidden_layers',
                                         'hidden_layer_size',
                                         'hidden_activation', 
                                         'hidden_regularizer',
                                         'hidden_regularisation', 
                                         'dropout_keep_prob',
                                         'learning_rate',
                                         'decay_learning_rate',
                                         'clip_gradients',
                                         'optimizer'])

#split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.7, random_state = 1)

#number of features in data
N_feats = X_train.shape[1]
N_instances = X_train.shape[0]

regularizers = {'L1':tf.contrib.layers.l1_regularizer, 'L2':tf.contrib.layers.l2_regularizer}
activations = {'relu':tf.nn.relu, 'sigmoid':tf.nn.sigmoid,'elu':tf.nn.elu, 'softplus':tf.nn.softplus,'None': None}
optimizers = {'Adam':tf.train.AdamOptimizer, 'SGD':tf.train.GradientDescentOptimizer}

#parameters
BATCH_SIZE_ = [64,128,256,512]
num_hidden_layers_ = [1,2,3,4,5,6]
hidden_layer_size_ = [25,50,100,200,400,800]
hidden_regularizer_ = ['L1', 'L2']
hidden_regularisation_ = [0.1,0.2,0.5,1.0,2.0]
hidden_activation_ = ['relu','sigmoid', 'elu', 'softplus', 'None']
dropout_keep_prob_ = [0.5,0.6,0.7,0.8,0.9,1.0]
learning_rate_ = [0.01,0.001,0.0001]
decay_learning_rate_ = [True,False]
clip_grads_ = [True,False]
optimizer_ = ['Adam']

#palceholders for X and Y
x = tf.placeholder(tf.float64, shape = (None, N_feats))
y = tf.placeholder(tf.float64, shape = (None, 1))

for i in range(100):
    #randomly sample params from given ranges
    BATCH_SIZE = np.random.choice(BATCH_SIZE_,size=1)[0]
    num_hidden_layers = np.random.choice(num_hidden_layers_,size=1)[0]
    hidden_layer_size = np.random.choice(hidden_layer_size_,size=1)[0]
    hidden_activation = activations[np.random.choice(hidden_activation_,size=1)[0]]
    hidden_regularizer = regularizers[np.random.choice(hidden_regularizer_,size=1)[0]]
    hidden_regularisation = np.random.choice(hidden_regularisation_,size=1)[0]
    dropout_keep_prob = np.random.choice(dropout_keep_prob_,size=1)[0]
    learning_rate = np.random.choice(learning_rate_,size=1)[0]
    decay_learning_rate = np.random.choice(decay_learning_rate_,size=1)[0]
    clip_grads = np.random.choice(clip_grads_,size=1)[0]
    optimizer = optimizers[np.random.choice(optimizer_,size=1)[0]]
        
    
    with tf.variable_scope("Scope"+str(i)) as varscope:

        #hidden layers
        hidden = tf.contrib.layers.repeat(x, num_hidden_layers, tf.contrib.layers.fully_connected, 
                                          num_outputs= int(hidden_layer_size), 
                                          activation_fn= hidden_activation,
                                          weights_regularizer= hidden_regularizer(hidden_regularisation), 
                                          biases_regularizer= hidden_regularizer(hidden_regularisation))

        #dropout layer
        dropout = tf.nn.dropout(hidden, dropout_keep_prob, name='Dropout')
        
        #output (i.e. y)
        y_pred = tf.layers.dense(dropout, units = 1, activation = None, name = 'Output')
    
    #loss
    loss = tf.reduce_sum(tf.losses.mean_squared_error(y, y_pred))
    
    #Function to clip gradients when optimizing
    def ClipIfNotNone(grad):
        if grad is None:
            return grad
        return tf.clip_by_value(grad, -1, 1)
        
    #set up a decaying learning rate, starting at the given value
    #and decaying to 0.96*learning_rate every epoch
    if decay_learning_rate == True:
        global_step = tf.Variable(0, trainable=False)
        learning_rate_new = tf.train.exponential_decay(learning_rate, global_step,
                                                       N_instances//BATCH_SIZE, 
                                                       0.96, staircase=True)
    else:
        learning_rate_new = learning_rate
    
    #Optimize the loss function      
    if clip_grads == True:  
        opt = optimizer(learning_rate_new)
        grads = opt.compute_gradients(loss)
        clipped_gradients = [(ClipIfNotNone(grad), var) for grad, var in grads]
        opt_op = opt.apply_gradients(clipped_gradients)
    else:
        opt_op = optimizer(learning_rate_new).minimize(loss)    
    
    #run max 20 epochs
    with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            #find number of stories
            n = X_train.shape[0]
            
            min_test_rmse_score = 2.0
            min_test_rmse_epoch = 0
            min_test_rmse_time = 0.0
            for epoch in range(20):
                print('----- Epoch', epoch, '-----')
                total_loss = 0
                #start timer for timing training
                t0 = time()
                #sample batches randomly
                for i in range(n // BATCH_SIZE):
                    indices = np.random.choice(n,BATCH_SIZE)
                    feed_dict = {x: X_train.iloc[indices], 
                                 y: Y_train.iloc[indices]}
                    #compute loss at current batch
                    _, current_loss = sess.run([opt_op, loss], feed_dict=feed_dict)
                    #update total loss
                    total_loss += current_loss
                print(' Train loss:', total_loss / n)
                t1 = time()
                
                #define feed_dict for training data            
                train_feed_dict = {x: X_train,
                                   y: Y_train}
                #predict on training data
                Y_train_pred = sess.run(y_pred, feed_dict = train_feed_dict)
                #postprocess predictions
                Y_train_pred_pp = post_process_preds(Y_train_pred)
                #compute training rmse score
                train_rmse = RMSE(Y_train, Y_train_pred_pp)
                print(' Train rmse:', round(train_rmse,5))
                
                #define feed_dict for test data            
                test_feed_dict = {x: X_test,
                                   y: Y_test}
                #predict on training data
                Y_test_pred = sess.run(y_pred, feed_dict = test_feed_dict)
                #postprocess predictions
                Y_test_pred_pp = post_process_preds(Y_test_pred)
                #compute training rmse score
                test_rmse = RMSE(Y_test, Y_test_pred_pp)
                print(' Test rmse:', round(test_rmse,5))
                #if necessary, update the min rmse score
                if test_rmse < min_test_rmse_score:
                    min_test_rmse_score = test_rmse
                    min_test_rmse_epoch = epoch
                    min_test_rmse_time = round(t1-t0)
            #store current results
            current_results = pd.DataFrame({'batch_size':[BATCH_SIZE],
                                            'num_hidden_layers':[num_hidden_layers],
                                            'hidden_layer_size':[hidden_layer_size],
                                            'hidden_activation':[hidden_activation], 
                                            'hidden_regularizer':[hidden_regularizer],
                                            'hidden_regularisation':[hidden_regularisation], 
                                            'dropout_keep_prob':[dropout_keep_prob],
                                            'learning_rate':[learning_rate],             
                                            'decay_learning_rate':[decay_learning_rate],             
                                            'clip_gradients':[clip_grads],             
                                            'optimizer':[optimizer],             
                                            'best_test_rmse':[min_test_rmse_score],             
                                            'best_test_rmse_epoch':[min_test_rmse_epoch],             
                                            'best epoch training_time':['{}s'.format(min_test_rmse_time)]})              
            print('Current Results: \n', current_results)
            #append results to dataframe
            results_df = pd.concat([results_df,current_results])
            results_df = results_df.sort_values(by='best_test_rmse')
            results_df = results_df.reset_index(drop=True)
            #save results
            results_df.to_csv('output/NeuralNetworkOptimizationV2.csv', index_label = 'Round')





