#importing required libraries
from FinalMultiPerceptron import MultiPerceptron
import pickle
from numpy import genfromtxt
import numpy as np
import math

#Loading the data
X_TrainData = genfromtxt('train_data.csv', delimiter=',')
y_TrainData = genfromtxt('train_labels.csv', delimiter=',')

# Creating a object of the model Multiperceptron
Object = MultiPerceptron(Epochs=40, lr=0.01, Input_Size =len(X_TrainData[0]))
# Splitting the Data into Train and Test
X_TraData, X_TestData, y_TraData, y_TestData = Object.train_test_split(X_TrainData,y_TrainData)

# Adding neurons to the hidden and output layer
Object.AddNeurons(20,4)
#Initializing Weights at the hidden and output layer
Object.initializeweights()
#Fitting the data
Object.fit(X_TraData, y_TraData)

#Calculating accuracy for the test data
y_pred = Object.predict(X_TestData)
Accuracy = Object.accuracy(y_pred, y_TestData)
print("Accuracy of Test Data" , Accuracy)

# Storing the Trained Model
pickle_file= "FinalMultiperceptron.pkl"
with open(pickle_file, 'wb') as file:
    pickle.dump(Object, file)




