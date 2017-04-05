from sklearn.linear_model import Ridge
import pandas as pd

# Get the data
data = pd.read_csv("big_feat.csv")

# Number of training points
n_train = 74067

# Training and test data
train = data.iloc[:n_train]
test = data.iloc[n_train:]

# Make a dataframe to store the results
results = pd.DataFrame(columns=["id","relevance"])
results["id"] = test["id"]

# Set the X and Y training and test
train_y = train["relevance"]
train_x = train.drop(["id","relevance"],axis=1)

test_x = test.drop(["id","relevance"],axis=1)

# Make the regressor
reg = Ridge()

# Fit the model
reg.fit(train_x.values, train_y.values)

# Make predictions
poo = reg.predict(test_x.values)

# Round any values larger than 3.0 to be 3.0
poo[poo > 3.0] = 3.0

# Write to results
results["relevance"] = poo

# Save to a CSV
results.to_csv("big_result.csv",index=False)

"""
df_all = pd.read_csv("df_all.csv",encoding="ISO-8859-1")
df_training = pd.read_csv("train.csv",encoding="ISO-8859-1")
df_testing = pd.read_csv("test.csv",encoding="ISO-8859-1")

rows = df_training.shape[0]

df_train = df_all.iloc[:rows]
df_test = df_all.iloc[rows:]
id_test = df_test['id']
y_train = df_train['relevance'].values
X_train = df_train
X_test = df_test

X_train.drop(X_train.columns[[1,2,3,4,5,6,7,8]],axis=1,inplace = True)
X_test.drop(X_test.columns[[1,2,3,4,5,6,7,8]],axis=1,inplace = True)

X_train = X_train.values
X_test = X_test.values

print(X_train)

poo = Ridge()

print("Fitting...")
poo.fit(X_train,y_train)
print("Predicting...")
thing = poo.predict(X_test)

print(thing)

thing[thing > 3.00] = 3.00

ids = df_test["id"]


df = pd.DataFrame(data = {"id":ids,"relevance":thing})

df.to_csv("result.csv", index=False, encoding="utf-8")
"""
















