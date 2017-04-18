from sklearn import svm
import pickle as p
import pandas as pd
from sklearn.model_selection import KFold, RandomizedSearchCV,train_test_split
from collections import defaultdict
from sklearn import metrics
from numpy import sqrt
import numpy as np
import metrics as m
from sklearn.decomposition import PCA
from  sklearn import preprocessing

def post_process_preds(y_pred):

    return np.array([max(min(y,3.0),1.0) for y in y_pred])

def load_data():
    #Loads data and outputs teaining and test sets
    print("Loading dataset...")
    try:
        train_x = p.load(open('input_clean/X_train.pkl', 'rb'))
        train_y = p.load(open('input_clean/Y_train.pkl', 'rb'))
        test_x = p.load(open('input_clean/X_test.pkl', 'rb'))
    except:
        print("Loading failed")
        return
    print("Dataset successfully loaded!")
    return train_x, train_y, test_x

def score_rmse(model,x,y):
    pre_predicts = model.predict(x)
    predicts = post_process_preds(pre_predicts)
    mean_sq_err = metrics.mean_squared_error(y, predicts)
    rmse = sqrt(mean_sq_err)
    return -rmse  # Returns negative so that grid search chooses the model with the lowest score


def predict_to_csv(model,x_test):
    print("Starting predictions...")
    predicts = model.predict(x_test)
    predicts = post_process_preds(predicts)
    df = pd.DataFrame(predicts, index=x_test.index)
    df.to_csv('output/SVMPredictions.csv')
    print("Predictions done and written on output/SVMPredictions.csv")

    return

def cross_validation(x,y,model,n_folds=10): # Not finished
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=1)
    results = []
    i=0
    for train_index, test_index in kf.split(x):
        i+=1
        print("Performing {0} fold out of {1}".format(i,n_folds))
        scores = defaultdict(float)
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        #model.fit(x_train,y_train.values.ravel())
        model.fit(x_train, y)
        predicts = model.predict(x_test)
        predicts = post_process_preds(predicts)
        exp_var, mean_abs_err, mean_sq_err, rmse, r2_sc = m.metrics_regress(predicts,y_test)
        scores['rmse'] = rmse
        scores['expl_var'] = exp_var
        scores['mae'] = mean_abs_err
        scores['mse'] = mean_sq_err
        scores['R2'] = r2_sc
        results.append(scores)
    print("Cross validation finished!")
    return results

def cv_tocsv(cv):
    n_folds = len(cv)
    index_folds = ["Fold {}".format(i) for i in range(1,n_folds+1)]
    df = pd.DataFrame(cv,index = index_folds)
    df = df.T
    df['mean']=df.mean(axis=1)
    df['sd'] = df.std(axis=1)
    df.to_csv('output/SVMCVResults.csv')
    print("Cross validation results written to SVMResults.csv")
    return

def random_grid_search(x, y,n_iter = 10,kernel = 'rbf', n_splits_cv = 3):

    #Performs a Randomized Grid Search over the parameters defined in 'parameters' variable.
    #train_pca = PCA(n_components=20)
    #x = train_pca.fit_transform(x)
    model = svm.SVC(verbose=False)
    print("Starting grid search with {} runs".format(n_iter))
    parameters_all = [{'kernel': 'rbf', 'gamma': [0.1, 0.5, 'auto', 1], 'C': [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50,100]},  #Parameters for a complete GridSearch
                  {'kernel': 'linear', 'C': [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100]},
                  {'kernel': 'poly', 'degree': [2, 3, 5], 'C': [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100]}
                  ]
    parameters_linear = {'C': [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100]} # RandomizedSearchCV doesn't allow for lists of dictionaries
    parameters_rbf = {'kernel': ['rbf'], 'gamma': list(np.logspace(-3,3)), 'C': list(np.logspace(-3,3))}
    parameters_poly = {'kernel': ['poly'], 'degree': [2, 3, 5], 'C': [0.001]}#, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100]}

    if kernel == 'rbf':
        parameters = parameters_rbf
    elif kernel == 'linear':
        model = svm.LinearSVR()
        parameters = parameters_linear
    elif kernel == 'poly':
        parameters = parameters_poly

    else:
        print("Kernel not valid")
        return

    kf = KFold(n_splits=n_splits_cv, shuffle=True, random_state=1)
    grid_search = RandomizedSearchCV(model, param_distributions=parameters, verbose=3,n_iter=n_iter, cv=kf, scoring=score_rmse)
    #grid_search.fit(x, y.values.ravel())
    grid_search.fit(x, y)
    print("Grid Search finished")
    return grid_search


def main():
    train_x, train_y, test_x = load_data()
    train_x,_,train_y,_ = train_test_split(train_x,train_y,train_size = 0.25,random_state=1)
    label_encoder = preprocessing.LabelEncoder()
    train_y = label_encoder.fit_transform(train_y)
    results = random_grid_search(train_x, train_y, kernel='rbf',n_iter = 1)
    df_results = pd.DataFrame(results.cv_results_)
    cv = cross_validation(train_x, train_y, results.best_estimator_, n_folds=10)
    cv_tocsv(cv)
    #print(df_results)
    df_results.to_csv("output/SVMOptimization.csv")
    model = results.best_estimator_
    #model.fit(train_x,train_y.values.ravel())
    model.fit(train_x, train_y)
    predict_to_csv(model, test_x)

    return

if __name__ == "__main__":
    main()




