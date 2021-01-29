"""
A loss function measures how good our predictions are,
we can use this to adjust the parameters of our network
"""
import numpy as np

from BDLibrary.tensor import Tensor

class Loss:
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError



class Cost(Loss):
    
    def __init__ (self,loss='MSE'):
        self.cost_func=loss
        
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        if (self.cost_func=='Linear'):
            return np.sum(predicted-actual)/len(predicted)
        elif (self.cost_func=='quadratic'):
            return np.sum(0.5*((predicted-actual)**2))/len(predicted)

        elif (self.cost_func == "cross_entropy"):
            return np.sum(np.nan_to_num(-actual*np.log(predicted)-(1-actual)*np.log(1-predicted)))/len(predicted)
        elif (self.cost_func == "MSE"):
            return np.sum((predicted - actual) ** 2)


    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        if (self.cost_func=='Linear'):
            return predicted / predicted # return 1 in all cases
        elif (self.cost_func == 'quadratic'):
            return (predicted - actual)/len(predicted)
        elif (self.cost_func == 'cross_entropy'):
            return (np.nan_to_num(-(actual-predicted) / ((1 - predicted) * predicted)))/len(predicted)
        elif (self.cost_func == "MSE"):
            return 2 * (predicted - actual)
        
'''       
class MSE(Loss):
    """
    MSE is mean squared error, although we're
    just going to do total squared error
    """
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return np.sum((predicted - actual) ** 2)

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return 2 * (predicted - actual)
'''