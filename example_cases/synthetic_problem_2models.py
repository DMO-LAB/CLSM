# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 20:55:11 2023

@author: oowoyele
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
from algorithm.create_models import CreateModel
from algorithm.clsm import CLSM
from algorithm.optimizers import OptimizerCLSMNewton

def save_obj(obj, filename):
    """Saves a Python object to a file using pickle."""
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def load_obj(filename):
    """Loads a Python object from a file using pickle."""
    with open(filename, 'rb') as f:
        return pickle.load(f)


def step_function(step):
    """Generate a function based on the given step."""
    x = np.arange(-15, 30, step)
    y = np.zeros_like(x)
    for i, j in enumerate(x):
        if j <= 0:
            y[i] = 0.2*np.sin(x[i])
        else:
            y[i] = 0.1*x[i]*np.cos(x[i])
    return x, y

# Generate data
x, y = step_function(0.5)

x1 = x
x2 = np.sin(x)
x3 = np.cos(x)
x4 = x1*x2
x5 = x2*x3
x6 = x1*x3

# Create additional features based on the data
features_name = ['x','sinx','cosx', 'xsinx', 'sinxcosx' ,'xcosx','bias']

X = np.column_stack([x1,x2,x3,x4,x5,x6])
y = y.reshape(-1,1) 

# Add noise to the data
noise_factor = 1e-2 # Adjust the noise factor as desired
noise = np.random.normal(0, noise_factor, size=y.shape)
y = y + noise

####################################################################################

# Training parameters
num_inputs = len(X)
num_targets = len(y)
inp = X
out = y
lam = 1e-4


learning_rate = 1

best_overall_error = 1e4

for itrial in range(5):
    
    fcn1 = CreateModel(inp, out,lasso_reg = True, lambda_reg = lam, ann_struct = [X.shape[1],1], dtype = torch.float64)
    fcn2 = CreateModel(inp, out,lasso_reg = True, lambda_reg =lam, ann_struct = [X.shape[1],1], dtype = torch.float64)
    fcn_list = [fcn1, fcn2]
    optimizer = OptimizerCLSMNewton(fcn_list = [fcn1,fcn2])
    moe = CLSM(fcn_list, kappa = 0.1, smoothen_alpha = True, n_neighbors = 10, states = X)
    
    for it in range(10000):
    
        loss_list = moe.compute_weighted_mse()
        loss_ = [loss.detach().numpy() for loss in loss_list]
        
        # Check if any value in loss list is NaN
        if any(np.isnan(loss_value) for loss_value in loss_):
            print(f"Error: Loss value is NaN at iteration {it}")
            break
        
        wp_np = [nwp for nwp in moe.get_num_winning_points()]
        alpha = moe.compute_alpha()
        
    
        if moe.smoothen_alpha == True:
            alpha = moe.alpha_smooth
        else:
            alpha = moe.alpha
    
        wp_np = [nwp for nwp in moe.get_num_winning_points()]
        
        optimizer.step(alpha,learning_rate)
        
        overall_error = np.sum([wp_np[ii]*loss_[ii] for ii in np.arange(moe.num_experts)])/inp.shape[0]
        if it%1000== 0:
            print(it, loss_, overall_error, wp_np)
    
    print("######################################################",  )
    print("overall error from trial ", str(itrial+1), " = ",  overall_error)
    print("######################################################",  )
    
    if overall_error < best_overall_error:
        print("updating models since better trial was found...")
        filename = 'saved_models/synthetic_2models/fcn_list.pkl'
        save_obj(fcn_list, filename)
        
        filename = 'saved_models/synthetic_2models/opt.pkl'
        save_obj(optimizer, filename)
        
        filename = 'saved_models/synthetic_2models/moe.pkl'
        save_obj(moe, filename)
        
        best_overall_error = overall_error
    
    ################################################################################################