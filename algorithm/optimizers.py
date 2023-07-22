# -*- coding: utf-8 -*-
"""
Created on Thu May 25 17:52:56 2023

@author: oowoyele
"""

import torch
import sys
import numpy as np
from scipy.optimize import differential_evolution
from torch.optim.lr_scheduler import StepLR


class OptimizerAdam:
    """ Optimizer using Adam for neural networks."""

    def __init__(self, fcn=None, parameters=None, learning_rate=0.01):
        if parameters is None:
            self.parameters = fcn.parameters
        else:
            self.parameters = parameters
        self.optim = torch.optim.Adam(parameters, lr=learning_rate)

    def step(self, loss):
        """Take an optimization step."""
        self.optim.zero_grad()
        loss.backward(retain_graph=True)
        self.optim.step()


class OptimizerCLSMAdam:
    """Optimizer using Adam for Competitive Learning Specialized Models."""

    def __init__(self, fcn_list=None, parameters=None, learning_rate=0.05):
        if fcn_list is not None:
            fcn_given = True
            self.num_experts = len(fcn_list)
        elif parameters is not None:
            params_given = True
            self.num_experts = len(parameters)

        self.optim = []
        self.scheduler = []
        self.mse_list = [[]] * len(fcn_list)

        for iexp in range(self.num_experts):
            if fcn_given:
                self.parameters = fcn_list[iexp].parameters
            elif params_given:
                self.parameters = parameters[iexp]
            
            self.optim += [torch.optim.Adam(self.parameters, lr=learning_rate)]
            
            self.scheduler += [StepLR(self.optim[-1], step_size=1000, gamma=0.999)]
            
            if fcn_given:
                self.mse_list[iexp] = fcn_list[iexp].mse()

    def step(self, loss_list, iter, moe):
        """Take an optimization step for all models."""
        for i in range(self.num_experts):
            self.optim[i].zero_grad()
            loss_list[i].backward(retain_graph=True)
            self.optim[i].step()


class OptimizerCLSMNewton:
    """Optimizer using Newton's method for Competitive Learning Specialized Models."""

    def __init__(self, fcn_list=None):
        self.fcn_list = fcn_list
        if fcn_list is not None:
            self.num_experts = len(fcn_list)

    @staticmethod
    def compute_grads(model, alpha):
        """Compute the gradient for a model."""
        error = torch.matmul(model.x, model.weights) - model.y
        n = model.x.size(0)
        gradient_mse = (2 / n) * torch.matmul(model.x.t(), alpha * error)
        gradient_l1 = model.lambda_reg * torch.sign(model.weights)
        return gradient_mse + gradient_l1

    @staticmethod
    def compute_hessian(model, alpha):
        """Compute the Hessian matrix for a model."""
        n = model.x.size(0)
        return (2 / n) * torch.matmul((model.x.t() * alpha.t()), model.x)

    def step(self, alpha, learning_rate):
        """Take an optimization step using Newton's method."""
        for i in range(len(self.fcn_list)):
            model = self.fcn_list[i]
            alpha_ = alpha[:,i:i+1]
            H = self.compute_hessian(model,alpha_)
            g = self.compute_grads(model,alpha_)
            diagonal_values = torch.rand(H.shape[0]) * 0.1 #- 1
            diagonal_matrix = torch.diag(diagonal_values)
            update = torch.matmul(torch.linalg.inv(H + diagonal_matrix), g)
            model.weights = model.weights - torch.minimum(torch.tensor(0.001), learning_rate*update)
            model.outpt = model.predict()


class GlobalOptimizerCLSM:
    """Global optimizer for Competitive Learning Specialized Models."""

    def __init__(self, fcn_list=None):
        self.fcn_list = fcn_list
        if fcn_list is not None:
            self.num_experts = len(fcn_list)

    def err(self, weights):
        """Compute the total error for the given weights."""
        total_error = 0
        start_index = 0
        for model in self.fcn_list:
            num_weights = np.prod(model.weights.shape)
            model_weights = weights[start_index: start_index + num_weights]
            model_weights = model_weights.reshape(model.weights.shape)
            model.weights = torch.from_numpy(model_weights).type(torch.float64)
            total_error += model.mse(update_y=True)
            start_index += num_weights
        return total_error.detach().numpy()

    def global_optimization(self):
        """Perform global optimization on model weights."""
        x0 = np.concatenate([model.weights.detach().numpy().ravel() for model in self.fcn_list])
        bounds = [(-1, 1) for _ in range(len(x0))]
        result = differential_evolution(self.err, bounds, popsize=50, maxiter=3000, disp=False)

        optimized_weights = result.x
        start_index = 0
        for model in self.fcn_list:
            num_weights = np.prod(model.weights.shape)
            model_weights = optimized_weights[start_index: start_index + num_weights]
            model.weights = torch.from_numpy(model_weights).type(torch.float64).reshape(model.weights.shape)
            start_index += num_weights

    def step(self, alpha, learning_rate):
        """Optimization step for the global optimizer."""
        for model in self.fcn_list:
            H = OptimizerCLSMNewton.compute_hessian(model, alpha)
            g = OptimizerCLSMNewton.compute_grads(model, alpha)
            update = torch.matmul(torch.linalg.inv(H), g)
            model.weights -= learning_rate * update
            model.outpt = model.pred()
