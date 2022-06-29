# `3-Bank-Afiliation.csv` with `Tensor Flow` model
# Visit https://www.section.io/engineering-education/machine-learning-with-xgboost-and-scikit-learn/, or my script on XGB for further details
# Used https://www.tensorflow.org/tutorials/load_data/csv for help with the pd and the split of data
# Useful for future https://www.tensorflow.org/tutorials/keras/regression
# Useful little theory https://towardsdatascience.com/building-a-deep-learning-model-using-keras-1548ca149d37

import pandas as pd                                         # loading and cleaning the dataset
import numpy as np                                          # manipulating it
import tensorflow as tf
from tensorflow import keras
tf.random.set_seed(100)

######################################## prepare the file

datacsv = "3-bank-additional-modified.csv"

df = pd.read_csv(datacsv, sep=";")                          # default csv separator is ","
df.columns[df.dtypes == 'object']                           # retrieving all data to object type
print('before dummies', len(df.columns))

df = pd.get_dummies(df,df.columns[df.dtypes == 'object'])   # encoded data into numbers

size_train = int(df.shape[0]//(100/80))                     # split the 80% of df for train                     

X_train = df.iloc[:size_train, :(len(df.columns)-2)]        # the csv file has 20 columns and 41189 rows
X_test = df.iloc[size_train:, :(len(df.columns)-2)]         # for the train we use only the 80% of data = 80% rows
y_train = df.iloc[:size_train, len(df.columns)-1]           # the y data is the 'label', is what we try to predict
y_test = df.iloc[size_train:, len(df.columns)-1]            # in this case is the last column 'yes' or 'not

####################################### Defining the model

model = keras.Sequential([                                
    keras.layers.Dense(100, activation='relu', input_shape=(X_train.shape[1],)),          
    keras.layers.Dense(100, activation='relu'), 
    keras.layers.Dense(2, activation='softmax')
])

# the last layer need to be 2, affiliation yes (1) or not (0), but with softmax we get their probabilities

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.00001), # optimizer='adam', # for predetermined parameters
              loss='sparse_categorical_crossentropy',
            #   loss = 'mean_squared_error',               
              metrics=['accuracy'])

######################################## Training and evaluating the mdeo

model.fit(X_train, y_train, validation_split=0.2, epochs=30) 
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc) 

######################################## Making predictions

pred_prob = model.predict(X_test)
predictions = []

for i in range (0,len(pred_prob)):
    predictions.append(np.argmax(pred_prob[i]))
    
print(f'Predicted: "{np.argmax(pred_prob[0])}", Actual: "{(y_test.values)[0]}"')

######################################## Saving the data to not compile everyday

with open('TF-Bank-Actual-array.txt', 'w') as f:
    for item in actual:
        f.write("%s\n" % item)
f.close

with open('TF-Bank-Pred_prob-array.txt', 'w') as f:
    for item in pred_prob[:,1]:
        f.write("%s\n" % item)
f.close

actual_f = np.loadtxt("TF-Bank-Actual-array.txt", dtype=float)
pred_prob_f = np.loadtxt("TF-Bank-Pred_prob-array.txt", dtype=float)

# That's all folks

