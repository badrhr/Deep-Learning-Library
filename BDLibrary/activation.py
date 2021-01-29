import numpy as np

#from BDLibrary.tensor import Tensor
from tensor import Tensor
from typing import Dict, Callable


class Layer:
    def __init__(self) -> None:
        self.params: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Produce the outputs corresponding to these inputs
        """
        raise NotImplementedError

    def backward(self, grad: Tensor) -> Tensor:
        """
        Backpropagate this gradient through the layer
        """
        raise NotImplementedError

F = Callable[[Tensor], Tensor]

class Activation(Layer):
    """
    An activation layer just applies a function
    elementwise to its inputs
    """
    def __init__(self, f: F, f_prime: F) -> None:
        super().__init__()
        self.f = f
        self.f_prime = f_prime

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return self.f(inputs)

    def backward(self, grad: Tensor) -> Tensor:
        """
        if y = f(x) and x = g(z)
        then dy/dz = f'(x) * g'(z)
        """
        return self.f_prime(self.inputs) * grad
    
    
    
# Sigmoid 
def Sigmoid(z: Tensor) -> Tensor:
    return 1.0 / (1.0+ np.exp(-z))

def Sigmoid_prime(z: Tensor) -> Tensor:
    s=1.0 / (1.0+ np.exp(-z))
    return s*(1-s) 

#Tanh
def tanh(z: Tensor) -> Tensor:
    return np.tanh(z)

def tanh_prime(z: Tensor) -> Tensor:
    return (1 - (np.tanh(z)** 2))

# relu
def relu(z: Tensor) -> Tensor:
    return np.maximum(0, z)

def relu_prime(z: Tensor) -> Tensor:
        if z==0 or z<0 : 
            z=0
        else:
            z=1
        return z



# leaky relu
def leakyrelu(z: Tensor) -> Tensor:
    return np.maximum(0.01*z, z)

def leakyrelu_prime(z: Tensor) -> Tensor:
        if z==0 or z<0 : 
            z=0.01
        else:
            z=1
        return z




class Activation_Function(Activation):
    
    def __init__(self):
        super().__init__(tanh, tanh_prime)
    
    def __init__(self, activation_function):
        super().__init__(tanh, tanh_prime)
        
    def activation_fn(self, z,activation= "tanh"):
        if activation == "leaky_relu":
            return np.maximum(0.01 * z, z)
        elif activation == "relu":
            return np.maximum(0, z)
        elif activation == "sigmoid":
            return 1.0 / (1.0+ np.exp(-z))
        elif activation == "tanh":
            return np.tanh(z)
    
    
    def activation_fn_prime(self, z,activation="tanh"):
        if activation == "leaky_relu":
            if z==0 or z<0 : 
                z=0.01
            else:
                z=1
            return z
        elif activation == "relu":
            if z==0 or z<0 : 
                z=0
            else:
                z=1
            return z
        elif activation == "sigmoid":
            return self.activation_fn(z) * (1 - self.activation_fn(z))
        elif activation == "tanh":
            return (1 - (np.tanh(z)** 2))