import csv
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from time import time
df_train = pd.read_csv('/Users/Hermione/MasterUCL/Info retrieval and data mining/coursework/dataset/train.csv', encoding="ISO-8859-1")
num_train = df_train.shape[0]

data = pd.read_csv("/Users/Hermione/MasterUCL/Info retrieval and data mining/coursework/feats.csv")
df_train = data.iloc[:num_train]
df_test = data.iloc[num_train:]
X_train = df_train.drop(['id','relevance','product_uid'],axis=1).values

y_train = df_train['relevance'].values

t0 = time()
params = {'n_estimators': 500, 'max_depth': 6, 'min_samples_split': 2, 'min_samples_leaf':15, 'learning_rate': 0.035, 'loss': 'ls', 'verbose':1}
clf = GradientBoostingRegressor(**params)

clf.fit(X_train, y_train)

feature_importance = clf.feature_importances_
print(feature_importance)
print("The total feature number is:",clf.feature_importances_.shape)

index = np.where(feature_importance > 0)
index = list(index[0])
print("The number of important feature is:", len(index))


with open('importantFeatureInd', 'w') as myfile:
    wr = csv.writer(myfile,  dialect='excel')
    wr.writerow(index)