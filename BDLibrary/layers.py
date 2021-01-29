"""
Our neural nets will be made up of layers.
Each layer needs to pass its inputs forward
and propagate gradients backward. For example,
a neural net might look like

inputs -> Linear -> Tanh -> Linear -> output
"""
from typing import Dict, Callable

import numpy as np

from BDLibrary.tensor import Tensor


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


class Linear(Layer):
    """
    computes output = inputs @ w + b
    """
    def __init__(self, input_size: int, output_size: int) -> None:
        # inputs will be (batch_size, input_size)
        # outputs will be (batch_size, output_size)
        super().__init__()
        self.params["w"] = np.random.randn(input_size, output_size)
        self.params["b"] = np.random.randn(output_size)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        outputs = inputs @ w + b
        """
        self.inputs = inputs
        return inputs @ self.params["w"] + self.params["b"]

    def backward(self, grad: Tensor) -> Tensor:
        """
        if y = f(x) and x = a * b + c
        then dy/da = f'(x) * b
        and dy/db = f'(x) * a
        and dy/dc = f'(x)

        if y = f(x) and x = a @ b + c
        then dy/da = f'(x) @ b.T
        and dy/db = a.T @ f'(x)
        and dy/dc = f'(x)
        """
        self.grads["b"] = np.sum(grad, axis=0)
        self.grads["w"] = self.inputs.T @ grad
        return grad @ self.params["w"].T


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


class Activation_Function(Activation):
    
    def __init__(self, activation = "tanh"):
        self.activation = self.tanh
        self.activation_prime = self.relu_prime
        
        if activation == "leaky_relu":
            self.activation = self.leakyrelu
            self.activation_prime = self.leakyrelu_prime
        elif activation == "relu":
            self.activation = self.relu
            self.activation_prime = self.relu_prime
        elif activation == "sigmoid":
            self.activation = self.Sigmoid
            self.activation_prime = self.Sigmoid_prime            
        
        super().__init__(self.activation, self.activation_prime)
            
       
    # Sigmoid 
    def Sigmoid(self,z: Tensor) -> Tensor:
        return 1.0 / (1.0+ np.exp(-z))
    
    def Sigmoid_prime(self,z: Tensor) -> Tensor:
        s=1.0 / (1.0+ np.exp(-z))
        return s*(1-s) 
    
    #Tanh
    def tanh(self,z: Tensor) -> Tensor:
        return np.tanh(z)
    
    def tanh_prime(self,z: Tensor) -> Tensor:
        return (1 - (np.tanh(z)** 2))
    
    # relu
    def relu(self,z: Tensor) -> Tensor:
        return np.maximum(0, z)
    
    def relu_prime(self,z: Tensor) -> Tensor:
            if z.all()==0 or z.all()<0 : 
                z=0
            else:
                z=1
            return z
       
    # leaky relu
    def leakyrelu(self,z: Tensor) -> Tensor:
        return np.maximum(0.01*z, z)
    
    def leakyrelu_prime(self,z: Tensor) -> Tensor:
            if z.all()==0 or z.all()<0 : 
                z=0.01
            else:
                z=1
            return z
    
