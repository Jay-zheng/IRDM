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
    return np.array(max(min(y_pred,3.0),1.0))

models=['LinearRegression',
          'Lasso',
          'Ridge',
          'ElasticNet',
          'KNN',
          'SVM',
          'XGBoost',
          'RandomForest',
          'NeuralNetwork']

CV_results = {}
predictions = {}
#read in the predictions and CV rmse means for each model
print('Attempting to load CV Results...')
for model in models:
    print(' Reading {} CV Results...'.format(model))
    CVResults_df = pd.read_csv(str('output/'+model+'CVResults.csv'),index_col='Unnamed: 0')
    preds_df = pd.read_csv(str('output/'+model+'Predictions.csv'),index_col='id')
    preds_df.columns = ['relevance']
    CV_results[model] = CVResults_df.loc['rmse','mean']
    predictions[model] = preds_df

#%%

#compute weights
weights = pd.DataFrame()
min_rmse = min(CV_results.values())
for model in models:
    weights.loc[model,'Weight'] = CV_results[model]
    
weights.Weight = weights.Weight.apply(lambda x: 1.0 - 10*(x - min_rmse))
norm = sum(weights.Weight)
weights.Weight = weights.Weight.apply(lambda x: x/norm)

#%%
print('Combining model predictions')
#predict ensemble predictions
ensemble_predictions = pd.DataFrame(index = predictions['LinearRegression'].index, columns = ['relevance'])
for i,model in enumerate(models):
    print(' Processing {} model'.format(model))
    weight = weights.loc[model,'Weight']
    if i == 0:
        ensemble_predictions.relevance = predictions[model].relevance.apply(lambda x: x *  weight)
    else:
        ensemble_predictions.relevance = ensemble_predictions.relevance + predictions[model].relevance.apply(lambda x: x * weight)

ensemble_predictions.relevance = ensemble_predictions.relevance.apply(lambda x: post_process_preds(x))
ensemble_predictions.to_csv('output/EnsemblePredictions.csv')
#%%
#check model predictions ( min & max )
for model in models:
    print('Model: {} \n Min: {:f} \t Max: {:f}'.format(model,min(predictions[model]['relevance']),max(predictions[model]['relevance'])))

print('Model: Ensemble \n Min: {:f} \t Max: {:f}'.format(min(ensemble_predictions['relevance']),max(ensemble_predictions['relevance'])))
