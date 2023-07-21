# -*- coding: utf-8 -*-
"""
Created on Thu May 25 17:52:56 2023

@author: oowoyele
"""

import torch
import sys
from torch.optim.lr_scheduler import StepLR

class optimizer(): # fully connected neural network class
    def __init__(self, fcn = None, parameters = None, learning_rate = 0.01):
        
        if parameters == None:
            self.parameters = fcn.parameters
        else:
            self.parameters = parameters
            
        self.optim = torch.optim.Adam(parameters, lr=learning_rate)
        #self.fcn = fcn
        
        
    def step(self, loss):
        #self.fcn.pred()
        #mse = self.fcn.mse()
        #torch.autograd.set_detect_anomaly(True)
        self.optim.zero_grad()
        loss.backward(retain_graph=True)
        self.optim.step()
        
    
class optimizerMoE(): # fully connected neural network class
    def __init__(self, fcn_list = None, parameters = None, learning_rate = 0.05):

        if fcn_list is not  None:
            fcn_given = True
            self.num_experts = len(fcn_list)
        elif parameters is not None:
            params_given = True
            self.num_experts = len(parameters)
            
        self.optim = []
        self.scheduler = []
        #self.fcn_list = fcn_list
        self.mse_list = [[]]*len(fcn_list)
        
        for iexp in range(self.num_experts):
            if fcn_given:
                self.parameters = fcn_list[iexp].parameters
            elif params_given:
                self.parameters = parameters[iexp]
            
            self.optim += [torch.optim.Adam(self.parameters, lr=learning_rate)]
            
            self.scheduler += [StepLR(self.optim[-1], step_size=1000, gamma=0.999)]
            
            if fcn_given:
                self.mse_list[iexp] = fcn_list[iexp].mse()
            
        
    def step(self, loss_list,iter,moe):
        #if iter < 1000:
        #    self.optim[0].zero_grad()
        #    loss_list[0].backward(retain_graph=True)
        #    self.optim[0].step()
        #else:
        for iexp in range(self.num_experts):
            self.optim[iexp].zero_grad()
            loss_list[iexp].backward(retain_graph=True)
            self.optim[iexp].step()
            #self.scheduler[iexp].step()
            #fcn.pred()
            #loss_list[iexp] = fcn.mse()
            
class optimizerMoE2():
    def __init__(self, fcn_list = None):
        self.fcn_list = fcn_list
        if fcn_list is not  None:
            fcn_given = True
            self.num_experts = len(fcn_list)
    
    def compute_grads(self,model,alpha):
        error = torch.matmul(model.x, model.weights) - model.y
        n = model.x.size(0)  # number of examples
        gradient_mse = (2/n) * torch.matmul(model.x.t(), alpha * error)
        gradient_l1 = model.lambda_reg * torch.sign(model.weights)
        gradient = gradient_mse + gradient_l1
        return gradient
    
    def compute_hessian(self,model,alpha):
        n = model.x.size(0)
        # Calculate the Hessian for the squared error term:
        hessian_mse = (2/n) * torch.matmul((model.x.t() * alpha.t()), model.x)
        return hessian_mse
    
    def step(self,alpha,learning_rate):
        for i in range(len(self.fcn_list)):
            model = self.fcn_list[i]
            alpha_ = alpha[:,i:i+1]
            H = self.compute_hessian(model,alpha_)
            g = self.compute_grads(model,alpha_)
            #update = torch.matmul(torch.linalg.inv(H), g)
            #model.weights = model.weights - learning_rate*update
            #model.outpt = model.pred()
            diagonal_values = torch.rand(H.shape[0]) * 0.1 #- 1

            # Create the diagonal matrix using torch.diag()
            diagonal_matrix = torch.diag(diagonal_values) #* 0.000002 - 0.000001
            update = torch.matmul(torch.linalg.inv(H + diagonal_matrix), g)
            #print(i, torch.max(update).detach().numpy())
            model.weights = model.weights - torch.minimum(torch.tensor(0.001), learning_rate*update)
            model.outpt = model.pred()
        
    
    # def compute_grads(self, model, alpha):
    #     error = torch.matmul(model.x, model.weights) - model.y
    #     n = model.x.size(0)  # number of examples
    #     gradient_mse = (2/n) * torch.matmul(model.x.t(), alpha * error)
    #     gradient_l2 = 2 * model.lambda_reg * model.weights
    #     gradient = gradient_mse + gradient_l2
    #     return gradient

    # def compute_hessian(self, model, alpha):
    #     n = model.x.size(0)
    #     # Calculate the Hessian for the squared error term:
    #     hessian_mse = (2/n) * torch.matmul((model.x.t() * alpha.t()), model.x)
    #     # Add the regularization term:
    #     hessian_l2 = 2 * model.lambda_reg * torch.eye(model.weights.size(0))
    #     hessian = hessian_mse + hessian_l2
    #     return hessian        
    
import numpy as np
from scipy.optimize import differential_evolution
from scipy.optimize import dual_annealing

class optimizerMoE3():
    def __init__(self, fcn_list = None):
        self.fcn_list = fcn_list
        if fcn_list is not  None:
            fcn_given = True
            self.num_experts = len(fcn_list)
    
    def compute_grads(self,model,alpha):
        error = torch.matmul(model.x, model.weights) - model.y
        n = model.x.size(0)  # number of examples
        gradient_mse = (2/n) * torch.matmul(model.x.t(), alpha * error)
        gradient_l1 = model.lambda_reg * torch.sign(model.weights)
        gradient = gradient_mse + gradient_l1
        return gradient
    
    def compute_hessian(self,model,alpha):
        n = model.x.size(0)
        hessian_mse = (2/n) * torch.matmul((model.x.t() * alpha.t()), model.x)
        return hessian_mse

    def err(self, weights):
        total_error = 0
        start_index = 0
        self.list = []
        for model in self.fcn_list:
            
            # Determine size of weights
            num_weights = np.prod(model.weights.shape)
            
            # Extract the corresponding weights from the flattened array
            model_weights = weights[start_index : start_index + num_weights]
            
            # Reshape the weights to match the model's weights
            model_weights = model_weights.reshape(model.weights.shape)
            
            model.weights = torch.from_numpy(model_weights).type(torch.float64)
            
            self.list += [model.mse(update_y = True)]
            
            total_error += model.mse(update_y = True)
            
            # Update the start index for the next model
            start_index += num_weights
        
        # Return numpy error
        return total_error.detach().numpy()

    def glo_opt(self):
        # Initial guess (flattened)
        x0 = np.concatenate([model.weights.detach().numpy().ravel() for model in self.fcn_list])

        # Bounds
        bounds = [(-1, 1) for _ in range(len(x0))]

        # Optimizing
        result = differential_evolution(self.err, bounds,popsize = 50, maxiter=3000,disp=False)
        # result = dual_annealing(self.err, bounds,maxiter=3000)

        # Updating weights
        optimized_weights = result.x
        start_index = 0
        for model in self.fcn_list:
            # Determine size of weights
            num_weights = np.prod(model.weights.shape)
            
            # Extract the corresponding weights from the flattened array
            model_weights = optimized_weights[start_index : start_index + num_weights]
            
            # Reshape the weights to match the model's weights
            model_weights = model_weights.reshape(model.weights.shape)
            
            # Update weights
            model.weights = torch.from_numpy(model_weights).type(torch.float64)
            
            # Update the start index for the next model
            start_index += num_weights

    
    def step(self,alpha,learning_rate):
        for i in range(len(self.fcn_list)):
            model = self.fcn_list[i]
            alpha_ = alpha[:,i:i+1]
            H = self.compute_hessian(model,alpha_)
            g = self.compute_grads(model,alpha_)
            update = torch.matmul(torch.linalg.inv(H), g)
            model.weights = model.weights - learning_rate*update
            model.outpt = model.pred()
