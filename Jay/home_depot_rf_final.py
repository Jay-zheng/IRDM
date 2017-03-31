
# coding: utf-8

# In[29]:

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import metrics
from numpy import sqrt


# In[7]:

df_all = pd.read_csv("C:/Users/JayZheng/Documents/Data Science/data mining/home-depot/data/df_all.csv",encoding="ISO-8859-1")
df_train = pd.read_csv("C:/Users/JayZheng/Documents/Data Science/data mining/home-depot/data/train.csv",encoding="ISO-8859-1")
df_test = pd.read_csv("C:/Users/JayZheng/Documents/Data Science/data mining/home-depot/data/test.csv",encoding="ISO-8859-1")


# In[8]:

numrow= df_train.shape[0]


# In[4]:

#df_all.drop(df_all.columns[[1,2,3,4,5,6,7,8]],axis=1,inplace = True)


# In[9]:

df_train = df_all.iloc[:numrow]
df_test = df_all.iloc[numrow:]
id_test = df_test['id']
y_train = df_train['relevance'].values
X_train = df_train[:]
X_test = df_test[:]


# In[10]:

X_train.drop(X_train.columns[[1,2,3,4,5,6,7,8]],axis=1,inplace = True)
X_test.drop(X_test.columns[[1,2,3,4,5,6,7,8]],axis=1,inplace = True)


# In[11]:

X_train, X_test, Y_train, Y_test = train_test_split(X_train, y_train, train_size = 0.7, random_state = 1)


# In[12]:

params = {
            'n_estimators': [160,200,250,300,350,400,450,500],
            'max_features': ['auto','sqrt','log2'],
            'max_depth': [4,8,12,16,20,30,40,50,60],
            'n_jobs': [3]
          }


# In[13]:

from time import time
#from sklearn.grid_search import GridSearchCV
#import sklearn.pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier


# In[14]:

clf = RandomForestClassifier()


# In[15]:

n_iter_search = 50
random_search = RandomizedSearchCV(clf, param_distributions=params,
                                   n_iter=n_iter_search)

start = time()
random_search.fit(X_train, Y_train.tolist())
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))


# In[17]:

result = pd.DataFrame(random_search.cv_results_)


# In[18]:

best_estimator = random_search.best_estimator_
best_score =random_search.best_score_
best_param=random_search.best_params_


# In[ ]:

CV_result= result.to_csv('C:/Users/JayZheng/Documents/Data Science/data mining/home-depot/data/CV_result.csv', index=False)


# In[55]:


result.sort("std_test_score")


# In[41]:

#restart from here 
df_train = df_all.iloc[:numrow]
df_test = df_all.iloc[numrow:]
id_test = df_test['id']
Y_train = pd.DataFrame(data = {'relevance':df_train['relevance']}, index = df_train.index)
X_train = df_train[:]
X_test = df_test[:]
X_train.drop(X_train.columns[[1,2,3,4,5,6,7,8]],axis=1,inplace = True)
X_test.drop(X_test.columns[[1,2,3,4,5,6,7,8]],axis=1,inplace = True)


# In[68]:

best_param


# In[69]:

#best parameters found after trying a few more random ones
n_estimators = 350
max_features = 'sqrt'
max_depth = 20
n_jobs = 3


# In[62]:

def evaluation(ground_truth,prediction):
    
    mean_abs_err = metrics.mean_absolute_error(ground_truth,prediction) # the lower the better
    #Mean Squared Error
    mean_sq_err = metrics.mean_squared_error(ground_truth,prediction)
    #Root Mean Squared Error
    rmse = sqrt(mean_sq_err)

    return mean_abs_err,mean_sq_err,rmse


# In[70]:

CV_results_df = pd.DataFrame()
#create 10-fold CV model
folds = KFold(n_splits=10, shuffle=True, random_state=1)
#lists to store metric values at each fold
rmse_scores = []
mean_abs_error_scores = []
mean_sq_error_scores = []

#carry out 10-fold CV on the data
fold = 1
print('Starting 10-fold cross validation')
for train_index, test_index in folds.split(X_train):
    #retrieve train - test split for current fold
    x_train = X_train.iloc[train_index,:]
    x_test = X_train.iloc[test_index,:]
    y_train = Y_train.iloc[train_index]
    y_test = Y_train.iloc[test_index]
    #create model
    model = RandomForestRegressor(n_estimators = n_estimators,
                                  max_features = max_features,
                                  max_depth = max_depth,
                                  n_jobs = n_jobs)
    #train model
    model.fit(x_train, y_train.values.ravel())    
    #predict on the test split
    y_pred = model.predict(x_test)
    mean_abs_err,mean_sq_err,rmse = evaluation(y_pred, y_test)
    rmse_scores.append(rmse)

    mean_abs_error_scores.append(mean_abs_err)
    mean_sq_error_scores.append(mean_sq_err)
 
    print('Fold: {}, \t RMSE: {:4f}'.format(fold, rmse_scores[-1]))
    #append score for current fold to results df
    for tup in [(rmse_scores,'rmse'),(mean_abs_error_scores,'mae'),(mean_sq_error_scores,'mse')]:
        scores, metric = tup   
        CV_results_df.loc[metric,'Fold '+str(fold)] = scores[-1]
    fold += 1


# In[74]:

cv_result = pd.DataFrame(CV_results_df)


# In[ ]:

cv_result.to_csv('C:/Users/JayZheng/Documents/Data Science/data mining/home-depot/data/cv_results.csv',index=False)


# In[75]:

model = RandomForestRegressor(n_estimators = n_estimators,
                              max_features = max_features,
                              max_depth = max_depth)
model.fit(X_train,Y_train.values.ravel())
print('Predicting on test dataset...')
#predict using the trained model
predict = model.predict(X_test)


# In[83]:

predictions = pd.DataFrame({"id": id_test, "relevance": predict}).to_csv('C:/Users/JayZheng/Documents/Data Science/data mining/home-depot/data/predictionv2.csv', index=False)


# In[ ]:



