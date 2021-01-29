"""
The canonical example of a function that can't be
learned with a simple linear model is XOR
"""
import numpy as np
from BDLibrary.optim import Optimizer,Adam


from BDLibrary.train import train
from BDLibrary.nn import NeuralNet
from BDLibrary.layers import Linear, Activation_Function

inputs = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
])

targets = np.array([
    [1, 0],
    [0, 1],
    [0, 1],
    [1, 0]
])

net = NeuralNet([
    Linear(input_size=2, output_size=2),
    Activation_Function('Segmoid'),
    Linear(input_size=2, output_size=2)
])

train(net, inputs, targets)

for x, y in zip(inputs, targets):
    predicted = net.forward(x)
    print(x, predicted, y)
