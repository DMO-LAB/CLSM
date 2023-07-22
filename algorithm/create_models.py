# -*- coding: utf-8 -*-
"""
Created on Thu May 25 16:56:02 2023

@author: oowoyele
"""

import torch
import sys

def initialize(shape=None, dtype=torch.float64):
    """Initializes a matrix with random numbers."""
    inp_shape = shape[0]
    out_shape = shape[1]
    scale = 1 / (inp_shape + out_shape)
    
    var = torch.rand((inp_shape, out_shape), dtype=dtype) * 0.2 - 0.1
    var.requires_grad = True
    return var

def initialize_1d(shape=None, dtype=torch.float64):
    """Initializes a vector with random numbers."""
    inp_shape = shape[0]
    scale = 1 / (inp_shape)
    
    var = torch.rand((inp_shape,), dtype=dtype) * 0.2 - 0.1
    var.requires_grad = True
    return var

class CreateModel:
    """
    Create the structure of Neural Network (or Lasso) models and manage their operations.
    
    Attributes:
    - ann_struct: List of integers denoting structure of the neural network.
    - n_targets: Number of targets.
    - n_layers: Number of hidden layers.
    - lasso_reg bool: If model is LASSO Regression model and not a NN
    """
    def __init__(self, input_tensor=None, output_tensor=None, lasso_reg=False, lambda_reg=0.01, 
                 alpha=0.01, ann_struct=None, activation='sigmoid', lin_output=False, dtype=torch.float64, 
                 weights=None, biases=None):
        
        self.ann_struct = [torch.tensor(ann_str) for ann_str in ann_struct]  # structure of ann as a list
        self.n_targets = self.ann_struct[-1]  # number of targets
        self.n_layers = len(self.ann_struct) - 2  # number of hidden layers
        self.act_layers = [[]] * (self.n_layers + 1)  # list to hold activated layers
        self.layers = [[]] * (self.n_layers + 1)  # list to hold layers before activation
        self.lin_output = lin_output
        self.dtype = dtype
        self.weights = weights  # network weights
        self.biases = biases  # network biases
        self.activation_fn = torch.nn.Sigmoid()  # using torch's built-in sigmoid
        self.x = torch.from_numpy(input_tensor)
        self.y = torch.from_numpy(output_tensor)
        self.lasso_reg = lasso_reg
        self.num_samples = self.x.shape[0]
        self.alpha_l = alpha
        self.lambda_reg = lambda_reg

        if self.lasso_reg == True:
            ones = torch.ones(self.x.size(0), 1)
            self.x = torch.cat((self.x, ones), dim=1)
            if weights is None:
                self.weights = torch.rand((self.ann_struct[0]+1, 1), dtype=self.dtype) * 2 - 1
            else:
                self.weights = torch.from_numpy(weights)
            
            self.weights.requires_grad = True
        
        # initialize weights. store as a list
        elif self.weights == None:
            self.weights = [initialize(shape = (self.ann_struct[0], self.ann_struct[1]))]
            for ii in range(self.n_layers-1):
                self.weights += [initialize(shape = (self.ann_struct[ii+1], self.ann_struct[ii+2]))]
            self.weights += [initialize(shape = (self.ann_struct[-2], self.n_targets))]
        
        # initialize biases. store as a list
        if self.biases == None:
            self.biases = [initialize_1d(shape = (self.ann_struct[1],))]
            for ii in range(self.n_layers-1):
                self.biases += [initialize_1d(shape = (self.ann_struct[ii+2],))]
            self.biases += [initialize_1d(shape = (self.n_targets,))]
            
        # Get initial predictions
        self.predict()

    def sigma(self, x):
        """Apply the activation function."""
        return self.activation_fn(x)
    
    def d_sigma(self, x):
        """Apply the derivative of the activation function."""
        sigmoid_x = 1 / (1 + torch.exp(-x))
        return sigmoid_x * (1 - sigmoid_x)

    def predict(self):
        """Forward pass to get predictions from the model."""
        # For Lasso regression
        if self.lasso_reg:
            self.output = torch.matmul(self.x, self.weights)
            self.parameters = self.weights
            return self.output
        
        # For Neural Network
        self.layers[0] = torch.matmul(self.x, self.weights[0]) + self.biases[0]
        self.act_layers[0] = self.sigma(self.layers[0])
        
        for ii in range(1, self.n_layers):
            self.layers[ii] = torch.matmul(self.act_layers[ii-1], self.weights[ii]) + self.biases[ii]
            self.act_layers[ii] = self.sigma(self.layers[ii])
            
        self.layers[self.n_layers] = torch.matmul(self.act_layers[self.n_layers - 1], self.weights[self.n_layers]) + self.biases[self.n_layers]
        
        if self.lin_output:
            self.act_layers[self.n_layers] = self.layers[self.n_layers]
        else:
            self.act_layers[self.n_layers] = self.sigma(self.layers[self.n_layers])
            
        self.output = self.act_layers[self.n_layers]
        self.parameters = self.weights + self.biases

        return self.output

    def predict_new(self, X):
        """Predict using new input data."""
        # compute first layer and activate it
        X = torch.from_numpy(X)
        self.layers[0] = torch.matmul(X, self.weights[0]) +  self.biases[0]
        self.act_layers[0] = self.sigma(self.layers[0])
        
        # loop over all remaining hidden units and activate them
        for ii in range(1, self.n_layers):
            self.layers[ii] = torch.matmul(self.act_layers[ii-1], self.weights[ii]) +  self.biases[ii]
            self.act_layers[ii] = self.sigma(self.layers[ii])
        
        # compute last layer 
        ii = self.n_layers
        self.layers[ii] = torch.matmul(self.act_layers[ii-1], self.weights[ii]) +  self.biases[ii]
        
        # we have an activated hidden unit, apply activation, and assign result to list containing activated layers
        if self.lin_output == False:
            self.act_layers[ii] = self.sigma(self.layers[ii])
        else:
            self.act_layers[ii] = self.layers[ii]
        
        out = self.act_layers[ii] # assign to output variable
        return out.detach().numpy()
    
    def mse(self, update_y=True):
        """Compute Mean Squared Error."""
        if update_y:
            self.predict()
        return torch.mean((self.y - self.output) ** 2)

    def get_gradients(self):
        """Compute the gradients of the error with respect to weights and biases."""
            # function to analytically compute gradients of error wrt weights and biases. assumes you have one target, the network is fully connected.
            
        self.num_samples = self.x.shape[0] # number of samples
        actLayers_ = [self.x] + self.actLayers[:-1] # list of activated layers
        actLayers_ = [nl[:,None,:] for nl in actLayers_] # expand arrays by adding another dimension
        
        # we start from last layer and back propagate to first
        
        # last layer
        if self.lin_output == False:
            sig = [-self.d_sigma(self.Layers[self.n_layers])[:,None,:]]
        else:
            qwe = -1*torch.eye(self.n_targets, dtype=self.dtype)[None, :]
            sig = [torch.tile(qwe, [self.num_samples, 1, 1])]
        
        gradW = [torch.matmul(torch.permute(actLayers_[self.n_layers], [0,2,1]), sig[-1])]

        # loop over all layers
        for i in range(1, self.n_layers+1):
            a = sig[-1]
            b = self.weights[self.n_layers - i + 1].T
            c = self.d_sigma(self.Layers[self.n_layers-i])
            d = torch.tensordot(a, b, dims=[[2],[0]])
            sig += [d * c[:,None,:]]
            gradW += [torch.matmul(torch.permute(actLayers_[self.n_layers-i], [0,2,1]), sig[-1])]

        # reverse lists so they go from first to last (dW1 ... dW_last)
        sig.reverse()
        gradb = sig
        gradW.reverse()
        grad_vars = []
        
        # combine gradients from weights and biases in a single list
        for j in range(len(gradW)):
            grad_vars += [gradW[j][:,None, :, :]]
        for j in range(len(gradb)):
            grad_vars += [gradb[j][:,None, :, :]]
    
        return grad_vars # return gradients
    
