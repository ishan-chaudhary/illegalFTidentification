# Artificial Neural Network
# ANN Selection

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('datasetFIX_ann_selection.csv')
X = dataset.iloc[:, 0:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Feature Scaling with Standardization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Part 2 - Train a Model

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import Adam

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu', input_dim = 2))
# classifier.add(Dropout(p = 0.1))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
opt = Adam(learning_rate = 0.01)
classifier.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Evaluate a train model
train_acc = classifier.evaluate(X_train, y_train, verbose = 0)


# Part 3 - Evaluate a model wih 5-Cross Validation

import keras
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

def build_classifier():
    classifier_kfold = Sequential()
    classifier_kfold.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu', input_dim = 2))
    classifier_kfold.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
   #learning rate = 0.0001, 0.0005, 0.001, 0.005, 0.01
    opt = Adam(learning_rate = 0.01)
    classifier_kfold.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier_kfold
classifier_kfold = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier_kfold, X = X_train, y = y_train, cv = 5)
mean_accuracies = accuracies.mean()
variance = accuracies.std()


# Part 4 - Making predictions and evaluating the model on the Test set

# Predicting the Test set results
y_pred = classifier.predict(X_test)
final_pred = (y_pred > 0.5)

# Making the Confusion Matrix and evaluate with final accuracy
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, final_pred)

from sklearn.metrics import accuracy_score
final_accuracy = accuracy_score(y_test, final_pred)








