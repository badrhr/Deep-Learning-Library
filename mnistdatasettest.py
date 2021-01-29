import numpy as np

from BDLibrary.train import train
from BDLibrary.nn import NeuralNet
from BDLibrary.layers import Linear, Activation_Function
from BDLibrary.data import DataIterator, BatchIterator
from BDLibrary.optim import Optimizer, SGD



data_path = "data/mnist/"
train_data = np.array(np.loadtxt(data_path + "mnist_test.csv", 
                        delimiter=","))

targets = train_data[:,[0,]]

inputs = train_data[:,1:]


net = NeuralNet([
    Linear(input_size=784, output_size=256),
    Activation_Function('relu'),
    Linear(input_size=256, output_size=128),
    Activation_Function('sigmoid'),
    Linear(input_size=128, output_size=64),
    Activation_Function('relu'),
    Linear(input_size=64, output_size=32),
    Activation_Function('sigmoid'),
    Linear(input_size=32, output_size=10)
])
    
train(net, inputs, targets,num_epochs = 5000, optimizer=SGD(lr=0.0001))

for x, y in zip(inputs, targets):
    predicted = net.forward(x)
    #print(x, predicted, y)
    print(predicted, y)

