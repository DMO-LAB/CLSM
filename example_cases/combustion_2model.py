# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 16:34:34 2023

@author: oowoyele
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 21:35:32 2023

@author: oowoyele
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
from algorithm.model import MLP
from algorithm.clsm import CLSM
from algorithm.optimize import optimizerMoE,optimizerMoE2,optimizerMoE3 
import pandas as pd
from scipy.integrate import odeint
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,mean_squared_error

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

def save_obj(obj, filename):
    """Saves a Python object to a file using pickle."""
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def load_obj(filename):
    """Loads a Python object from a file using pickle."""
    with open(filename, 'rb') as f:
        return pickle.load(f)

step = 2

x1 = np.loadtxt('flamespeed_data/flameSpeed.txt')[0::step,].T.ravel()
x2 = np.loadtxt('flamespeed_data/flameSpeed_350')[0::step,].T.ravel()
x3 = np.loadtxt('flamespeed_data/flameSpeed_400')[0::step,].T.ravel()
x4 = np.loadtxt('flamespeed_data/flameSpeed_450')[0::step,].T.ravel()
x5 = np.loadtxt('flamespeed_data/flameSpeed_500')[0::step,].T.ravel()

y = np.vstack([x1,x2,x3,x4,x5]).ravel()
phi = np.array(list(np.arange(0.6, 1.6, 0.01)[0::step])*10*5) #10 rep pressure, 5 rep T
P0 = np.array(list(np.repeat(np.linspace(1,10,10),len(x1)/10))*5) 
T0 = np.array([298,350,400,450,500])
T0 = np.repeat(T0,len(x1))
X = np.vstack([P0,T0,phi]).T
y = y.reshape(-1,1)

xmin = X.min(axis=0)
xmax = X.max(axis=0)
Xn = (X- xmin) / (xmax - xmin)  # normalizing to range 0-1

ymax = np.max(y, axis=0)
ymin = np.min(y, axis=0)
Y = (y - ymin) / (ymax - ymin)  # normalize source term to range 0-1

Xn = Xn
noise_factor = 3e-2# Adjust the noise factor as desired
noise = np.random.normal(0, noise_factor, size=Y.shape)
Y = Y #+ noise



step = 3

x1 = np.loadtxt('flamespeed_data/flameSpeed.txt')[1::step,].T.ravel()
x2 = np.loadtxt('flamespeed_data/flameSpeed_350')[1::step,].T.ravel()
x3 = np.loadtxt('flamespeed_data/flameSpeed_400')[1::step,].T.ravel()
x4 = np.loadtxt('flamespeed_data/flameSpeed_450')[1::step,].T.ravel()
x5 = np.loadtxt('flamespeed_data/flameSpeed_500')[1::step,].T.ravel()

y = np.vstack([x1,x2,x3,x4,x5]).ravel()
phi = np.array(list(np.arange(0.6, 1.6, 0.01)[1::step])*10*5) #10 rep pressure, 5 rep T
P0 = np.array(list(np.repeat(np.linspace(1,10,10),len(x1)/10))*5) 
T0 = np.array([298,350,400,450,500])
T0 = np.repeat(T0,len(x1))
X_test = np.vstack([P0,T0,phi]).T
y_test = y.reshape(-1,1)

X_test = (X_test- xmin) / (xmax - xmin)  # normalizing to range 0-1

Y_test = (y_test - ymin) / (ymax - ymin)  # normalize source term to range 0-1

#noise_factor = 3e-3# Adjust the noise factor as desired
noise = np.random.normal(0, noise_factor, size=Y_test.shape)
y_test = Y_test# + noise

num_inputs = len(X)
num_targets = len(y)
inp = Xn
out = Y

best_overall_error = 10000
for itrial in range(5):
    fcn1 = MLP(inp, out, annstruct = [3, 3, 1], activation = 'sigmoid', lin_output = True, dtype = torch.float64)
    fcn2 = MLP(inp, out, annstruct = [3, 3, 1], activation = 'sigmoid', lin_output = True, dtype = torch.float64)
    fcn_list = [fcn1,fcn2]
    moe = CLSM(fcn_list, kappa = 1, smoothen_alpha = False)
    
    opt = optimizerMoE(fcn_list, learning_rate = 0.01)
    
    for it in range(200000):
        
        # alpha_weighted = smoothen_alpha(moe.compute_alpha(), Xn, n_neighbors=5)
        loss_list = moe.compute_weighted_mse(it)
        loss_ = [loss.detach().numpy() for loss in loss_list]
        wp_np = [nwp for nwp in moe.get_num_winning_points()]
        opt.step(loss_list,iter=it,moe=moe)
        
        overall_error = np.sum([wp_np[ii]*loss_[ii] for ii in np.arange(moe.num_experts)])/inp.shape[0]
        
        if it%10000== 0:
            print(it, loss_, overall_error, wp_np)
    
    print("######################################################",  )
    print("overall error from trial ", str(itrial+1), " = ",  overall_error)
    print("######################################################",  )
    
    if overall_error < best_overall_error:
        print("updating models since better trial was found...")
        filename = 'saved_models/flamespeed_2models/fcn_list.pkl'
        save_obj(fcn_list, filename)
        
        filename = 'saved_models/flamespeed_2models/opt.pkl'
        save_obj(opt, filename)
        
        filename = 'saved_models/flamespeed_2models/moe.pkl'
        save_obj(moe, filename)
        
        best_overall_error = overall_error
    
################################################################################################