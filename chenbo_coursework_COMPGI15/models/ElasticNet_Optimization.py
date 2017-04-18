<<<<<<< HEAD

# coding: utf-8

# In[2]:

# All the necessary libs
import pandas as pd
import numpy as np
import Utils as f
from time import time

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet


# In[5]:

# Load training data
X = f.load_obj('input/X_train')
y = f.load_obj('input/Y_train')

en = ElasticNet()

optimization_df = pd.DataFrame(columns=['rank', 'time', 'rmse', 'alpha', 'l1_ratio','fit_intercept', 'normalize', 'tol'])
optimized_list = []
optimized_dict = {}

# Utility function to report best scores
def report(results, n_top=50):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            rmse = np.sqrt(abs(results['mean_test_score'][candidate]))
            optimized_dict = {'rank': i, 'time': results['mean_fit_time'][candidate], 'rmse': rmse,
                              'alpha': results['param_alpha'].data[candidate],
                             'l1_ratio': results['param_l1_ratio'].data[candidate],
                             'fit_intercept ': results['param_fit_intercept'].data[candidate],
                              'normalize': results['param_normalize'].data[candidate],
                             'tol': results['param_tol'].data[candidate]}
            optimized_list.append(optimized_dict)
            print("Model with rank: {0}".format(i))
            print("Mean rmse score: {0:.3f})".format(
                  np.sqrt(abs(results['mean_test_score'][candidate]))))
            print("Parameters: {0}".format(results['params'][candidate]))            

# specify parameters and distributions to sample from
params = {
            'alpha': [0.0001,0.001,0.01,0.1,1.0,10.0,100.0],
            'fit_intercept': [True,False],
            'l1_ratio': [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
            'normalize': [True,False],
            'tol': [0.0001,0.001,0.01,0.1],
          }

k_fold = KFold(3, random_state=1)
# run randomized search
n_iter_search = 50
random_search = RandomizedSearchCV(en, param_distributions=params,scoring='neg_mean_squared_error',
                                   cv=k_fold,n_iter=n_iter_search)
start = time()
random_search.fit(X, y)
print("Randomized Search CV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.cv_results_)

# Save params
optimization_df = optimization_df.append(optimized_list)
optimization_df.to_csv('output/ElasticNetOptimization.csv')


# In[1]:

# Save notebook as py 
get_ipython().system(u'jupyter nbconvert --to=python ElasticNet_Optimization.ipynb')


# In[ ]:



=======

# coding: utf-8

# In[2]:

# All the necessary libs
import pandas as pd
import numpy as np
import Utils as f
from time import time

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet


# In[5]:

# Load training data
X = f.load_obj('input/X_train')
y = f.load_obj('input/Y_train')

en = ElasticNet()

optimization_df = pd.DataFrame(columns=['rank', 'time', 'rmse', 'alpha', 'l1_ratio','fit_intercept', 'normalize', 'tol'])
optimized_list = []
optimized_dict = {}

# Utility function to report best scores
def report(results, n_top=50):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            rmse = np.sqrt(abs(results['mean_test_score'][candidate]))
            optimized_dict = {'rank': i, 'time': results['mean_fit_time'][candidate], 'rmse': rmse,
                              'alpha': results['param_alpha'].data[candidate],
                             'l1_ratio': results['param_l1_ratio'].data[candidate],
                             'fit_intercept ': results['param_fit_intercept'].data[candidate],
                              'normalize': results['param_normalize'].data[candidate],
                             'tol': results['param_tol'].data[candidate]}
            optimized_list.append(optimized_dict)
            print("Model with rank: {0}".format(i))
            print("Mean rmse score: {0:.3f})".format(
                  np.sqrt(abs(results['mean_test_score'][candidate]))))
            print("Parameters: {0}".format(results['params'][candidate]))            

# specify parameters and distributions to sample from
params = {
            'alpha': [0.0001,0.001,0.01,0.1,1.0,10.0,100.0],
            'fit_intercept': [True,False],
            'l1_ratio': [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
            'normalize': [True,False],
            'tol': [0.0001,0.001,0.01,0.1],
          }

k_fold = KFold(3, random_state=1)
# run randomized search
n_iter_search = 50
random_search = RandomizedSearchCV(en, param_distributions=params,scoring='neg_mean_squared_error',
                                   cv=k_fold,n_iter=n_iter_search)
start = time()
random_search.fit(X, y)
print("Randomized Search CV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.cv_results_)

# Save params
optimization_df = optimization_df.append(optimized_list)
optimization_df.to_csv('output/ElasticNetOptimization.csv')


# In[1]:

# Save notebook as py 
get_ipython().system(u'jupyter nbconvert --to=python ElasticNet_Optimization.ipynb')


# In[ ]:



>>>>>>> b87fc3811e42ab51738dd141760a7221e7902d47
