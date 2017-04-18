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
from sklearn.linear_model import LinearRegression
import metrics as m

#function to cap predictions on a scale of 1 to 3
def post_process_preds(y_pred):
    return np.array([max(min(y[0],3.0),1.0) for y in y_pred])

#df to store results
CV_results_df = pd.DataFrame()

""" Compare 3 baselines for prediction:
1. Using Non-cleaned data and 1 feature (natural tf on product_title, search_term)
2. Using Non-cleaned data and 68 features (to compare the effect of adding features)
3. Using Cleaned data and 1 feature (to compare the effect of cleaning data)
4. Using Cleaned data and 68 features (to compare the combined effect of adding features and cleaning data)"""
baselines = ['not-clean;1-feat', 'not-clean;68-feats', 'clean;1-feat', 'clean;68-feats']
for baseline in baselines:
    print('\n Starting script for {}'.format(baseline))    
    #specify the directory depending on the baseline we're running
    if 'not-clean' in baseline:
        data_dir = 'input'
    else:
        data_dir = 'input_clean'
    print('Attempting to load pre-processed data...')
    try:
        #load pre-processed training data
        X_train = f.load_obj(data_dir+'/X_train')
        X_test = f.load_obj(data_dir+'/X_test')
        Y_train = f.load_obj(data_dir+'/Y_train')
        print('Pre-processed data successfully loaded...')
    except:
        print('Pre-processed data failed to load, ensure the working directory is correct...')

    #if only using 1 feature, redefine the data to only include the first feature
    if '1-feat' in baseline:
        X_train = pd.DataFrame(X_train.iloc[:,0])
        X_test = pd.DataFrame(X_test.iloc[:,0])
    
    #dict to store CV results
    CV_results_dict = {'Baseline':baseline}
    #create 10-fold CV model
    folds = KFold(n_splits=10, shuffle=True, random_state=1)
    #list to store rmse value at each fold
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
        x_train = X_train.iloc[train_index]
        x_test = X_train.iloc[test_index]
        y_train = Y_train.iloc[train_index]
        y_test = Y_train.iloc[test_index]
        #create model
        model = LinearRegression()
        #train model
        model.fit(x_train, y_train)    
        #predict on the test split
        y_pred = model.predict(x_test)
        #rescale predictions
        y_pred_pp = post_process_preds(y_pred)
        #evaluate predictions and append score to lists
        exp_var,mean_abs_err,mean_sq_err,rmse,r2_sc = m.metrics_regress(y_pred_pp, y_test)
        #append scores to lists
        rmse_scores.append(rmse)
        exp_var_scores.append(exp_var)
        mean_abs_error_scores.append(mean_abs_err)
        mean_sq_error_scores.append(mean_sq_err)
        r2_scores.append(r2_sc)
        print('Fold: {}, \t RMSE: {:4f}'.format(fold, rmse_scores[-1]))
        #CV_results_dict['rmse_Fold_'+str(fold)] = rmse_scores[-1]
        fold += 1
    
    #compute mean and standard deviation of each metric, and append these to the results df
    for tup in [(rmse_scores,'rmse'),(exp_var_scores,'expl_var'),(mean_abs_error_scores,'mae'),(mean_sq_error_scores,'mse'),(r2_scores,'R2')]:
        scores, metric = tup        
        #compute mean and std deviation of the scores
        mean = np.mean(scores)
        std = np.std(scores)
        print('10 fold CV '+metric+' mean: {:4f}'.format(mean))
        print('10 fold CV '+metric+' standard deviation: {:4f}'.format(std))
        CV_results_dict[metric+'_mean'] = mean
        CV_results_dict[metric+'_std'] = std
        
    
    CV_results_df = pd.concat([CV_results_df,pd.DataFrame(CV_results_dict, index=[baseline])])
    
    print('Training new model on training dataset...')
    #train a new model on the whole training data
    model = LinearRegression()
    model.fit(X_train,Y_train)
    print('Predicting on test dataset...')
    #predict using the trained model
    predictions = model.predict(X_test)
    #rescale predictions to a 1-3 scale
    predictions = post_process_preds(predictions)
    #add index to predictions
    predictions = pd.DataFrame(predictions, index = X_test.index)
    predictions.columns = ['relevance']
    print('Saving predictions to output folder...')
    #save predictions
    predictions.to_csv('output/LinearRegression'+'('+baseline+')'+'Predictions.csv')

CV_results_df.to_csv('output/LinearRegressionBaselineCVResults.csv',index=False)
    
#%%

import seaborn as sns
from pylab import get_cmap
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['xtick.labelsize'] = 15
matplotlib.rcParams['ytick.labelsize'] = 15
matplotlib.rcParams['axes.titlesize'] = 15

axes = {}
fig, (axes['ax1'], axes['ax2'], axes['ax3'], axes['ax4'], axes['ax5']) = plt.subplots(1,5, sharey=True, sharex=True,figsize=(15,3))
#plot the four baseline scores on a heatmap, for each metric
for num,tup in enumerate([('rmse',"RdYlGn_r"),('expl_var','RdYlGn'),('mae','RdYlGn_r'),('mse','RdYlGn_r'),('R2','RdYlGn')]):
    metric, cmap = tup
    sns_df = pd.DataFrame()
    for i, row in enumerate(CV_results_df.iterrows()):
        ID, row = row
        if '68-feats' in row['Baseline']:
            if 'not-clean' in row['Baseline']:
                sns_df.loc['68 Features','Non-Clean Data'] = row[metric+'_mean']
            else:
                sns_df.loc['68 Features','Clean Data'] = row[metric+'_mean']
        else:
            if 'not-clean' in row['Baseline']:
                sns_df.loc['1 Feature','Non-Clean Data'] = row[metric+'_mean']
            else:
                sns_df.loc['1 Feature','Clean Data'] = row[metric+'_mean']
    
    #plot results on separate graphs for each metric
    plt.figure(figsize=(6,4))
    ax = plt.axes()
    plot = sns.heatmap(data=sns_df, 
                       annot = True, 
                       robust = True, 
                       cmap = get_cmap(cmap),
                       annot_kws={"size":15})
    ax.set_title('Baseline '+ metric +' Scores')
    plt.savefig('output/LinearRegressionBaselines_'+metric+'Plot.png')
    
    #plot all plots on one graph
    sns.heatmap(data=sns_df, 
                annot = True, 
                robust = True, 
                cmap = get_cmap(cmap),
                cbar=False,
                square=True,
                annot_kws={"size":15},
                xticklabels = ['Not-Clean','Clean'],
                yticklabels = ['1 Feat','68 Feats'],
                ax = axes['ax'+str(num+1)])
    axes['ax'+str(num+1)].set_title(metric)
    fig.savefig('output/LinearRegressionBaselines_PlotAllMetrics.png')