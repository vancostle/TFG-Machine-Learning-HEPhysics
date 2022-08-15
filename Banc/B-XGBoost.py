#`3-Bank-Afiliation.csv` with `XGBoost-ScikitLean` Model
# https://www.section.io/engineering-education/machine-learning-with-xgboost-and-scikit-learn/
# Subscriptions on bank's term deposit predictor classification model.
# The used dataset contains costumers' information and important attributes, but before we'll need to clean it a little bit.

import xgboost as xgb
import pandas as pd     # loading and cleaning the dataset
import numpy as np      # manipulating it
import sklearn
from sklearn.model_selection import train_test_split    # split data intro train and test
from sklearn.metrics import accuracy_score

datacsv = "3-bank-additional-modified.csv"

df = pd.read_csv(datacsv, sep=";")     # default csv separator is ","

df.columns[df.dtypes == 'object']       # retrieving all data to object type

df = pd.get_dummies(df,df.columns[df.dtypes == 'object'])    # encoded data into numbers

X = df.iloc[:, 0:62]              
y = df.iloc[:, 63] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
xgb_classifier = xgb.XGBClassifier() 
xgb_classifier.fit(X_train,y_train)  

predictions = xgb_classifier.predict(X_test)        # making predictions
pred_prob = xgb_classifier.predict_proba(X_test)    # getting the probability of the predictions

print("Accuracy of Model::",accuracy_score(y_test,predictions))  # accuracy of the model

############### Saving data and don't need to compile all this everytime you need to make a change in the figures

actual = pd.Series(y_test).values

with open('XGB-Bank-Actual-array.txt', 'w') as f:
    for item in actual:
        f.write("%s\n" % item)
f.close

with open('XGB-Bank-Pred_prob-array.txt', 'w') as f:
    for item in pred_prob[:,1]:
        f.write("%s\n" % item)
f.close

actual_f = np.loadtxt("XGB-Bank-Actual-array.txt", dtype=float)
pred_prob_f = np.loadtxt("XGB-Bank-Pred_prob-array.txt", dtype=float)


# As easy as a pie folks
