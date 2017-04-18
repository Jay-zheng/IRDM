# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 21:59:22 2017

@author: Chris
"""

from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import Utils as f
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
import metrics as m
import tensorflow as tf

#function to cap predictions on a scale of 1 to 3
def post_process_preds(y_pred):
    return np.array([max(min(y,3.0),1.0) for y in y_pred])
    

print('Attempting to load pre-processed data...')
try:
    #load pre-processed training data
    X_train = f.load_obj('input_clean/X_train')
    X_test = f.load_obj('input_clean/X_test')
    Y_train = f.load_obj('input_clean/Y_train')
    print('Pre-processed data successfully loaded...')
except:
    print('Pre-processed data failed to load, ensure the working directory is correct...')

###################### Defining the model #######################

#number of features in data
N_feats = X_train.shape[1]
N_instances = X_train.shape[0]

#define some of the features to choose from when building the model
regularizers = {'L1':tf.contrib.layers.l1_regularizer, 'L2':tf.contrib.layers.l2_regularizer}
activations = {'relu':tf.nn.relu, 'sigmoid':tf.nn.sigmoid,'elu':tf.nn.elu, 'softplus':tf.nn.softplus,'None': None}
optimizers = {'Adam':tf.train.AdamOptimizer, 'SGD':tf.train.GradientDescentOptimizer}

#parameters of best model (chosen from optimization results)
BATCH_SIZE = 64
num_hidden_layers = 3
hidden_layer_size = 400
hidden_activation = activations['sigmoid']
hidden_regularizer = regularizers['L1']
hidden_regularisation = 2.0
dropout_keep_prob = 0.9
learning_rate = 0.0001
decay_learning_rate = False
clip_grads = True
optimizer = optimizers['Adam']
    
#palceholders for X and Y
x = tf.placeholder(tf.float64, shape = (None, N_feats))
y = tf.placeholder(tf.float64, shape = (None, 1))

with tf.variable_scope("Scope1") as varscope:
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


###################### 10 FOLD CV #############################
if True:
    #dict to store CV results
    CV_results_df = pd.DataFrame()
    #create 10-fold CV model
    folds = KFold(n_splits=10, shuffle=True, random_state=1)
    #lists to store metric values at each fold
    rmse_scores = []
    exp_var_scores = []
    mean_abs_error_scores = []
    mean_sq_error_scores = []
    r2_scores = []
    #carry out 10-fold CV on the data
    fold = 1
    print('Starting 10-fold cross validation')
    for train_index, test_index in folds.split(X_train):
        #retrieve train - test split for current fold
        x_train = X_train.iloc[train_index,:]
        x_test = X_train.iloc[test_index,:]
        y_train = Y_train.iloc[train_index]
        y_test = Y_train.iloc[test_index]
        
        #Train model for 18 epochs
        print('Training Neural Network for 18 epochs...')
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            #find number of stories
            n = x_train.shape[0]
    
            for epoch in range(18):
                print('----- Epoch', epoch, '-----')
                total_loss = 0
                #sample batches randomly
                for i in range(n // BATCH_SIZE):
                    indices = np.random.choice(n,BATCH_SIZE)
                    feed_dict = {x: x_train.iloc[indices], 
                                 y: y_train.iloc[indices]}
                    #compute loss at current batch
                    _, current_loss = sess.run([opt_op, loss], feed_dict=feed_dict)
                    #update total loss
                    total_loss += current_loss
                print(' Train loss:', total_loss / n)
                
                #define feed_dict for training data            
                train_feed_dict = {x: x_train,
                                   y: y_train}
                #predict on training data
                Y_train_pred = sess.run(y_pred, feed_dict = train_feed_dict)
                #postprocess predictions
                Y_train_pred_pp = post_process_preds(Y_train_pred)
                #compute training rmse score
                train_exp_var,train_mean_abs_err,train_mean_sq_err,train_rmse,train_r2_sc = m.metrics_regress(Y_train_pred_pp, y_train)
                print(' Train rmse:', round(train_rmse,5))
                
                #define feed_dict for test data            
                test_feed_dict = {x: x_test,
                                  y: y_test}
                #predict on training data
                Y_test_pred = sess.run(y_pred, feed_dict = test_feed_dict)
                #postprocess predictions
                Y_test_pred_pp = post_process_preds(Y_test_pred)
                #evaluate predictions
                exp_var,mean_abs_err,mean_sq_err,rmse,r2_sc = m.metrics_regress(Y_test_pred_pp, y_test)
                print(' Test rmse:', round(rmse,5))
        
        #append scores from 18th epoch to lists
        rmse_scores.append(rmse)
        exp_var_scores.append(exp_var)
        mean_abs_error_scores.append(mean_abs_err)
        mean_sq_error_scores.append(mean_sq_err)
        r2_scores.append(r2_sc)
        print('Fold: {}, \t RMSE: {:4f}'.format(fold, rmse_scores[-1]))
        #append score for current fold to results df
        for tup in [(rmse_scores,'rmse'),(exp_var_scores,'expl_var'),(mean_abs_error_scores,'mae'),(mean_sq_error_scores,'mse'),(r2_scores,'R2')]:
            scores, metric = tup   
            CV_results_df.loc[metric,'Fold '+str(fold)] = scores[-1]
        fold += 1
    
    #compute mean and standard deviation of each metric, and append these to the results df
    for tup in [(rmse_scores,'rmse'),(exp_var_scores,'expl_var'),(mean_abs_error_scores,'mae'),(mean_sq_error_scores,'mse'),(r2_scores,'R2')]:
        scores, metric = tup        
        #compute mean and std deviation of the scores
        mean = np.mean(scores)
        std = np.std(scores)
        print('10 fold CV '+metric+' mean: {:4f}'.format(mean))
        print('10 fold CV '+metric+' standard deviation: {:4f}'.format(std))
        #append mean and std to results df
        CV_results_df.loc[metric,'mean'] = mean
        CV_results_df.loc[metric,'std'] = std
    #save results to csv
    CV_results_df.to_csv('output/NeuralNetworkCVResults.csv')

#%%
###################### PREDICTING ON TEST SET #############################

print('Training new model on training dataset...')
#Train model on entire training set for 18 epochs
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    #find number of stories
    n = X_train.shape[0]

    for epoch in range(18):
        print('----- Epoch', epoch, '-----')
        total_loss = 0
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
        
        #define feed_dict for training data            
        train_feed_dict = {x: X_train,
                           y: Y_train}
        #predict on training data
        Y_train_pred = sess.run(y_pred, feed_dict = train_feed_dict)
        #postprocess predictions
        Y_train_pred_pp = post_process_preds(Y_train_pred)
        #compute training rmse score
        train_exp_var,train_mean_abs_err,train_mean_sq_err,train_rmse,train_r2_sc = m.metrics_regress(Y_train_pred_pp, Y_train)
        print(' Train rmse:', round(train_rmse,5))

    print('Predicting on test dataset...')
    #predict using the trained model
    #define feed_dict for test data            
    test_feed_dict = {x: X_test}
    #predict on training data
    predictions_test = sess.run(y_pred, feed_dict = test_feed_dict)
#rescale predictions to a 1-3 scale
predictions = pd.DataFrame(post_process_preds(predictions_test))
predictions = predictions.apply(lambda x: x[0],axis = 1)
#add index to predictions
predictions_df = pd.DataFrame(X_test.index)
predictions_df['relevance'] = predictions
print('Saving predictions to output folder...')
#save predictions
predictions_df.to_csv('output/NeuralNetworkPredictions.csv',index=False)