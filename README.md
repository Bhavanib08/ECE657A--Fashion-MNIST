# Implementing Backpropogation Algorithm
The Dataset considered has 24754 training dataset and 784 features which can be classified into 4 classes. 
Also the labels are one – hot encoded
## Splitting Of Data
I split the data into train and validation set in the ratio 80:20.  We are splitting the train data into train and validation data (split from the train dataset) to tune hyper-parameters and threshold. This ratio is chosen as for training the model we need a large data and a small percent of data to test.
## Stochastic/Online Gradient Descent
The error gradient in Online Gradient Descent is estimated from a single randomly selected data point from the training dataset and the model weights are then updated.
## Shuffling of the Data
We Shuffle the data so that the networks learn faster from the most unexpected sample. Shuffling of the training set is also done so that successive training data points never (rarely) belong to the same class.
It will also present input data points that produce larger error more frequently than data points that produce a small error.
##Initializing the weights
The initial assumption of weights can make a significant effect on the training. Weights should be chosen randomly. If the neurons start with the same weights, then the neurons will follow the same gradient and will always end up doing the same thing as one another.
## Choosing of Learning Rate
Learning rate is the amount by which the model weights get updated each iteration. Choosing the Learning rate is required because a small learning rate can cause slower convergence but a better result and a larger learning rate can result in a faster convergence but a less optimal result.
I chose 0.01 to be my learning rate considering the above reasons and considering this my training happened faster and loss stabilized.
## Activation Functions
I used two different activation functions at the hidden layer and output layer. For the Hidden layer I applied sigmoid activation function and for the Output Layer I applied Softmax Activation functions.
### Sigmoid Activation Function
This allows for a smooth curve of real values number from [0, 1]. This would ease the calculation and tuning of errors in a such a way that in the next forward pass iteration, it will output predictions from [0,1]. 
### Softmax Activation and cross entropy Function
Our data is one- hot encoded and there exits 4 classes and for multi class classification it is preferred to use Softmax and cross Entropy Function. 

## Number of Nodes
At Input layer the number of nodes will be equal to input + bias and the output layer in our case the number of nodes will be 4 because of the data in one – hot encoded and there exits 4 classes. The number of nodes at the hidden layer depends on a variety of variables and there are a number of configuration designs to choose from.
The general consensus is that, if the number of inputs are larger than the number of outputs, the number nodes in the hidden layer should be in between the number of nodes in the input and Output layer but never greater than the number of nodes at the input, this is because in this way the network can better generalize and avoid over fitting or memorizing of the training data.
I have Considered 20 nodes in addition to a bias node at the hidden layer performing various trials with different number of nodes.
## Number of Epochs
Number of Epochs is a hyper parameter that states the number of times the learning will work through the data to fit the model. Ideally higher number of epochs must give higher accuracy until to a point the model is not over fitting the data.
Considering this I gave the number of Epochs to be 40 after various trials with other number of Epochs and 40 gave the best accuracy.
Accuracies
### Training Accuracy: 98.44
### Validation Accuracy: 96.86
### Testing Accuracy: 97.03
