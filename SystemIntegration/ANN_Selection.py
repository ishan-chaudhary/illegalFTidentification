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

# Feature Scaling with Standardization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


# Part 2 - Train, Test and Evaluate wih 5-Cross Validation

import keras
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu', input_dim = 2))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
   #learning rate = 0.0001, 0.0005, 0.001, 0.005, 0.01
    opt = Adam(learning_rate = 0.01)
    classifier.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X, y = y, cv = 5)
mean_accuracies = accuracies.mean()
variance = accuracies.std()


# After did the Part 2, restart kernel and run only the Part 3


# Part 3 - Making Prediction for Classification using The Best Parameter from 5 Cross Validation Implementation

import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('datasetFIX_ann_selection.csv')
X = dataset.iloc[:, 0:2].values
y = dataset.iloc[:, 2].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

# Feature Scaling with Standardization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


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


# replace 'a' in code below with delta heading value input
# replace 'b' in code below with distance value input
new_prediction = classifier.predict(sc.transform(np.array([[a, b]])))
#new_prediction = (new_prediction > 0.5)
