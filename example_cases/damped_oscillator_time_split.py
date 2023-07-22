# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 13:45:27 2023
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


# System parameters
m = 0.75
c = 0.05
k = 2.4


def ode1(y, t):
    """System of differential equations."""
    y1, y2 = y
    dy1dt = y2

    f2 = 0
    if t < 2:
        f2 = 2*t

    dy2dt = -c/m*y2 - k/m*y1 + f2
    
    return [dy1dt, dy2dt]


# Parameter names
param = ['displacement', 'velocity']

# Initial conditions
y0 = [2, 0]

# Time points
t = np.linspace(0, 25, 400)

# Solve ODE
y = odeint(ode1, y0, t)

y = np.array(y)
y1 = y[:, 0]
y2 = y[:, 1]

tl0 = np.where(t < 2)[0]
y2dot = -c/m*y2 - k/m*y1
y2dot[tl0] = y2dot[tl0] + t[tl0]*2

# Feature extraction
x1 = y1
x2 = y2
x3 = y1*y2
x4 = y1**2
x5 = y2**2
x6 = t
x7 = y1**3
x8 = y2**3
X = np.column_stack([x1, x2, x3, x4, x5, x6, x7, x8])
y2dot = y2dot.reshape(-1, 1) 

features_name = ['y1', 'y2', 'y1y2', 'y1_2', 'y2_2', 't', 'y1_3', 'y2_3', 'bias']

data_ = np.concatenate([y[:, 0:1], np.array(t)[:, None]], axis=1)
scaler = MinMaxScaler()
data_n = scaler.fit_transform(data_)

num_inputs = len(X)
num_targets = len(y2dot)
inp = X
out = y2dot
lam = 1e-6

learning_rate = 1
best_overall_error = 10000

for trial in range(5):
    fcn1 = CreateModel(inp, out, lasso_reg=True, lambda_reg=lam, ann_struct=[X.shape[1], 1], dtype=torch.float64)
    fcn2 = CreateModel(inp, out, lasso_reg=True, lambda_reg=lam, ann_struct=[X.shape[1], 1], dtype=torch.float64)
    
    fcn_list = [fcn1, fcn2]
    opt2 = OptimizerCLSMNewton(fcn_list=fcn_list)
    
    moe = CLSM(fcn_list, kappa=0.1, smoothen_alpha=True, n_neighbors=10, states=data_n)

    for iteration in range(2000):
        kk = np.random.permutation(400)[:400]
        
        loss_list = moe.compute_weighted_mse()
        loss_ = [loss.detach().numpy() for loss in loss_list]

        # Check NaN in loss
        if any(np.isnan(loss_value) for loss_value in loss_):
            print(f"Error: Loss value is NaN at iteration {iteration}")
            break

        wp_np = [nwp for nwp in moe.get_num_winning_points()]
        
        if moe.smoothen_alpha:
            alpha = moe.alpha_smooth
        else:
            alpha = moe.alpha
        
        opt2.step(alpha, learning_rate)
        
        overall_error = np.sum([wp_np[ii]*loss_[ii] for ii in range(moe.num_experts)])/inp.shape[0]
        if iteration % 5000 == 0:
            print(iteration, loss_, overall_error, wp_np)
    
    print("######################################################")
    print(f"Overall error from trial {trial + 1} = {overall_error}")
    print("######################################################")
    
    if overall_error < best_overall_error:
        print("Updating models since a better trial was found...")

        filenames = [
            'saved_models/spring_mass_time_models/fcn_list.pkl',
            'saved_models/spring_mass_time_models/opt.pkl',
            'saved_models/spring_mass_time_models/moe.pkl'
        ]
        
        objects_to_save = [fcn_list, opt2, moe]
        
        for f_name, obj in zip(filenames, objects_to_save):
            save_obj(obj, f_name)
        
        best_overall_error = overall_error
