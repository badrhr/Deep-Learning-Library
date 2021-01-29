"""
We use an optimizer to adjust the parameters
of our network based on the gradients computed
during backpropagation
"""
from BDLibrary.nn import NeuralNet
import numpy as np

class Optimizer:
    def step(self, net: NeuralNet) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr

    def step(self, net: NeuralNet) -> None:
        for param, grad in net.params_and_grads():
            param -= self.lr * grad



class Adam(Optimizer):
    def __init__(self, eta=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.m_dw, self.v_dw = 0, 0
        self.m_db, self.v_db = 0, 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.eta = eta
        
        
    def step(self, net: NeuralNet) -> None:
        #w, b, dw, db 
        t = 1
        for param, grad in net.params_and_grads():
            w = param["w"]
            b = param["b"]
            dw = grad["w"]
            db = grad["b"]
                        
            ## dw, db are from current minibatch
            ## momentum beta 1
            # *** weights *** #
            self.m_dw = self.beta1*self.m_dw + (1-self.beta1)*dw
            # *** biases *** #
            self.m_db = self.beta1*self.m_db + (1-self.beta1)*db
    
            ## rms beta 2
            # *** weights *** #
            self.v_dw = self.beta2*self.v_dw + (1-self.beta2)*(dw**2)
            # *** biases *** #
            self.v_db = self.beta2*self.v_db + (1-self.beta2)*(db)
    
            ## bias correction
            m_dw_corr = self.m_dw/(1-self.beta1**t)
            m_db_corr = self.m_db/(1-self.beta1**t)
            v_dw_corr = self.v_dw/(1-self.beta2**t)
            v_db_corr = self.v_db/(1-self.beta2**t)
    
            ## update weights and biases
            param["w"] = w - self.eta*(m_dw_corr/(np.sqrt(v_dw_corr)+self.epsilon))
            param["b"] = b - self.eta*(m_db_corr/(np.sqrt(v_db_corr)+self.epsilon))