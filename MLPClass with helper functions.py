#importing required libraries
import numpy as np

np.random.seed(1)
# Creating a class MultuPerceptron
class MultiPerceptron():
    # --init-- function initializes Input, Number of Epochs and Learning Rate
    def __init__(self, Epochs=10, lr=0.01, Input_Size=1):
        self.Epochs = Epochs
        self.lr = lr
        self.Input_Size = Input_Size
        self.Hidden_Neurons = 0
        self.Output_Neurons = 0
        self.h_weights = []
        self.o_weights = []
        self.InputBias_Matrix = []
        self.Forward_hidden_activated = []
        self.Forward_out_activated = []
        self.hidden_layer_output = []

    # AddNeurons function adds neurons to hidden layer and output layer
    def AddNeurons(self, H_neurons, O_neurons):
        self.Hidden_Neurons = H_neurons
        self.Output_Neurons = O_neurons

    # initializeweights function initializes weight matrices at hidden and output layer
    def initializeweights(self):
        self.h_weights = np.random.rand(self.Input_Size + 1, self.Hidden_Neurons)*(10*np.exp(-4))
        self.o_weights = np.random.rand(self.Hidden_Neurons + 1, self.Output_Neurons)*(10*np.exp(-4))
        # print("Weights at the Output Layer", self.o_weights)
        # print("Weights at the Hidden Layer", self.h_weights)

    #ForwardPass
    def Forward_pass(self, X):
        # Appending bias to the input
        self.InputBias_Matrix = np.append(X, [-1])
        # Multiplying Input Matrix with the weights at the input layer
        Forward_hidden = np.matmul(self.InputBias_Matrix, self.h_weights)
        # Applying Sigmoid activation function to the output of hidden layer
        self.Forward_hidden_activated = self.Sigmoid_func(Forward_hidden)
        # Appending bias to the hidden layer
        self.hidden_layer_output = np.append(self.Forward_hidden_activated, [-1])
        # Multiplying Hidden layer output matrix with the weights at the output layer
        Forward_out = np.matmul(self.hidden_layer_output, self.o_weights)
        # Appplying Softmax function to the output layer
        self.Forward_out_activated = self.softmax_func(Forward_out)
        return self.Forward_out_activated

    def backprop(self, y):
        # Here we subtract Expected from Output to find the back propogated error at the output because here we use the derivative
        # of the cross Entropy function multiplied by the derivative of Softmax Function which is nothing but (Output - expected)
        d_out = self.soft_der(y)
        d_out = np.asarray(d_out)
        d_out = d_out.reshape(-1, len(d_out))
        self.hidden_layer_output = self.hidden_layer_output.reshape(-1, len(self.hidden_layer_output))
        # We find the delta weight at output layer by multiplying the backpropogated error to the output of the hidden layer
        dw_out = np.matmul(np.transpose(d_out), self.hidden_layer_output)
        # We find the back propogated error at the hidden layer by multiplying output of the output layer , output  layer weights, input to output layer and (1-input to output layer)
        d_hidden = np.matmul(d_out, np.transpose(self.o_weights[:len(self.o_weights) - 1]))
        d_hidden = d_hidden * self.sig_der(self.Forward_hidden_activated)
        self.InputBias_Matrix = np.asarray(self.InputBias_Matrix)
        self.InputBias_Matrix = self.InputBias_Matrix.reshape(-1, len(self.InputBias_Matrix))
        # We find the delta weight at the hidden layer by multiplying Input to the hidden layer with back propogated error
        dw_hidden = np.matmul(np.transpose(d_hidden), self.InputBias_Matrix)
        # print("Updated Weights at Hidden Layer",self.h_weights)
        self.h_weights = self.h_weights - self.lr * (np.transpose(dw_hidden))
        # print("Updated Weights at Output Layer", self.o_weights)
        self.o_weights = self.o_weights - self.lr * (np.transpose(dw_out))

    # Shuffle function to shuffle input data and label data
    def shuffle(self,a):
        indx = np.arange(len(a))
        np.random.shuffle(indx)
        a = a[indx]
        return a
    # Calculating the cross entropy loss
    def calError(self, Out, Expec):
        sum = np.sum(np.multiply(Expec, np.log(Out + np.exp(-8))))
        m = len(Out)
        L = -(1 / m) * sum
        return L

    # predict function gives the prediction of the output for given weights
    def predict(self, X):
        self.PredictArray = []
        final_value = 0
        for i in range(len(X)):
            Output = self.Forward_pass(X[i])
            final_value = np.argmax(Output)
            predict = np.zeros(len(Output))
            predict[final_value] = 1
            self.PredictArray.append(predict)
        self.PredictArray = np.asarray(self.PredictArray)
        return self.PredictArray

    # accuracy function
    def accuracy(self, y_p, y):
        s = 0
        for i in range(len(y_p)):
            y_pred = np.argmax(y_p[i])
            y_actual = np.argmax(y[i])
            if (y_pred == y_actual):
                s += 1
        accuracy = s / len(y_p)
        return accuracy

    #train_test_split function splits the data into train and validation set
    def train_test_split(self,X_dataset, y_dataset, s=0.80):
        split = int(len(X_dataset)*s)
        X_T = X_dataset[0:split]
        X_V = X_dataset[split + 1:]
        y_T = y_dataset[0:split]
        y_V = y_dataset[split + 1:]
        return  X_T , X_V , y_T, y_V

# Fit function shuffles the data, splits the data into train and validate data, reports the cross entropy loss and also calcultates the accuracy for train and validate data
    def fit(self, X, y):
        Error = 0
        self.previousVacc = 0
        self.Vacc = 0
        self.Tacc = 0
        for i in range(self.Epochs):
            self.X = self.shuffle(X)
            self.y = self.shuffle(y)
            self.X_TraData, self.X_ValData, self.y_TraData, self.y_ValData   = self.train_test_split(X,y)
            sigmaError = 0
            self.y_predict_val = []
            self.y_predict_tra = []
            for j in range(len(self.X_TraData)):
                Output = self.Forward_pass(self.X_TraData[j])
                self.backprop(self.y_TraData[j])
                last = self.y_TraData[j]
                Error = self.calError(Output, last)
                sigmaError += Error
            E = sigmaError / len(self.X_TraData)
            self.y_predict_val = self.predict(self.X_ValData)
            self.y_predict_tra = self.predict(self.X_TraData)
            self.previousVacc = self.Vacc
            self.Vacc = self.accuracy(self.y_predict_val, self.y_ValData)
            self.Tacc = self.accuracy(self.y_predict_tra, self.y_TraData)
            print("Epoch", i,"Train Accuracy", self.Tacc, "Validation Accuracy", self.Vacc, "Loss", E)

    # Functions of activation functions and derivatives of the activation functions
    def Sigmoid_func(self, a):
        sig = 1 / (1 + np.exp(-a))
        return sig

    def sig_der(self, der):
        d = der * (1 - der)
        return d

    def softmax_func(self, a):
        soft_activate = np.exp(a)
        return soft_activate / np.sum(soft_activate, axis=0)

    def soft_der(self, x):
        y = self.Forward_out_activated - x
        return y




