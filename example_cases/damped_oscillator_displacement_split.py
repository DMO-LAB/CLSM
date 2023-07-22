# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 19:02:34 2023
@author: oowoyele
"""


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
import pickle
from algorithm.create_models import CreateModel
from algorithm.clsm import CLSM
from algorithm.optimizers import OptimizerCLSMNewton
from sklearn.preprocessing import MinMaxScaler
from scipy.integrate import odeint


def save_obj(obj, filename):
    """Saves a Python object to a file using pickle."""
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def load_obj(filename):
    """Loads a Python object from a file using pickle."""
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Physical properties for the spring-mass system
m = 1.0
c1 = 0.25
k1 = 10
delta = 0.25
c2 = 0.15
k2 = 5

# System of differential equations
def ode1(y, t):
    y1, y2 = y
    dy1dt = y2
    dy2dt = -c1/m * y2 - k1/m * y1 + 0.1 * t

    if y1 > delta:
        dy2dt += -c2/m * y2 - k2/m * (y1 - delta)
    
    return [dy1dt, dy2dt]

# Initial conditions
y0 = [1, 0]
# Time points
t = np.linspace(0, 20, 200)
# Solve ODE
y = odeint(ode1, y0, t)
y1 = y[:, 0]
y2 = y[:, 1]

# Adjusting y2dot based on certain conditions
yl0 = np.where(y1 > delta)[0]
y2dot = -c1/m * y2 - k1/m * y1 + 0.1 * t
y2dot[yl0] += -c2/m * y2[yl0] - k2/m * (y1[yl0] - delta)

# Feature generation for the model
X = np.column_stack([y1, y2, y1*y2, y1**2, y2**2, t, y1**3, y2**3])
y2dot = y2dot.reshape(-1, 1) 

data_ = np.concatenate([y[:, 0:1], np.array(t)[:, None]], axis=1)
scaler = MinMaxScaler()
data_n = scaler.fit_transform(data_)

# Training parameters
num_inputs = len(X)
num_targets = len(y2dot)
lam = 1e-6
learning_rate = 1
best_overall_error = 10000

# Training loop
for itrial in range(5):
    fcn1 = CreateModel(X, y2dot, lasso_reg=True, lambda_reg=lam, ann_struct=[X.shape[1], 1], dtype=torch.float64)
    fcn2 = CreateModel(X, y2dot, lasso_reg=True, lambda_reg=lam, ann_struct=[X.shape[1], 1], dtype=torch.float64)

    fcn_list = [fcn1, fcn2]
    optimizer = OptimizerCLSMNewton(fcn_list=fcn_list)
    moe = CLSM(fcn_list, kappa=0.1, smoothen_alpha=True, n_neighbors=10, states=data_n)
    
    for iteration in range(2000):
        # Random permutation for stochastic optimization
        kk = np.random.permutation(200)[:200]
        
        loss_list = moe.compute_weighted_mse()
        loss_values = [loss.detach().numpy() for loss in loss_list]
        
        if any(np.isnan(value) for value in loss_values):
            print(f"Error: Loss value is NaN at iteration {iteration}")
            break

        winning_points = [points for points in moe.get_num_winning_points()]
        alpha = moe.compute_alpha()

        if moe.smoothen_alpha:
            alpha = moe.alpha_smooth
            
        optimizer.step(alpha, learning_rate)

        overall_error = np.sum([winning_points[i] * loss_values[i] for i in range(moe.num_experts)]) / len(X)
        if iteration % 5000 == 0:
            print(iteration, loss_values, overall_error, winning_points)

    # Save models if current trial has the best result so far
    if overall_error < best_overall_error:
        print("updating models since better trial was found...")
        filenames = [
            'saved_models/spring_mass_displacement_models/fcn_list.pkl',
            'saved_models/spring_mass_displacement_models/opt.pkl',
            'saved_models/spring_mass_displacement_models/moe.pkl'
        ]
        
        objects_to_save = [fcn_list, optimizer, moe]
        
        for obj, fname in zip(objects_to_save, filenames):
            save_obj(obj, fname)

        best_overall_error = overall_error
