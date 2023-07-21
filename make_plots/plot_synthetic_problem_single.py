# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 20:25:20 2023

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


def save_obj(obj, filename):
    """Saves a Python object to a file using pickle."""
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def load_obj(filename):
    """Loads a Python object from a file using pickle."""
    with open(filename, 'rb') as f:
        return pickle.load(f)


def step_function(step):
    x = np.arange(-15, 30, step)
    y = np.zeros_like(x)
    for i, j in enumerate(x):
        if j <= 0:
            y[i] = 0.2*np.sin(x[i])
        else:
            y[i] = 0.1*x[i]*np.cos(x[i])
    return x, y

x, y = step_function(0.5)

x1 = x
x2 = np.sin(x)
x3 = np.cos(x)
x4 = x1*x2
x5 = x2*x3
x6 = x1*x3

features_name = ['x','sinx','cosx', 'xsinx', 'sinxcosx' ,'xcosx','bias']

X = np.column_stack([x1,x2,x3,x4,x5,x6])
y = y.reshape(-1,1) 

noise_factor = 1e-2 # Adjust the noise factor as desired
noise = np.random.normal(0, noise_factor, size=y.shape)
y = y + noise

####################################################################################

num_inputs = len(X)
num_targets = len(y)
inp = X
out = y
lam = 1e-4

#########################################################################################
filename = 'saved_models/synthetic_1model/fcn_list.pkl'
fcn_list = load_obj(filename)

filename = 'saved_models/synthetic_1model/opt.pkl'
opt = load_obj(filename)

filename = 'saved_models/synthetic_1model/moe.pkl'
moe = load_obj(filename)

if moe.smoothen_alpha == True:
    alpha = moe.alpha_smooth
else:
    alpha = moe.alpha
#########################################################################################

# Set up the figure and axis
fig = plt.figure(figsize = (12,8))
ax = fig.gca()

# Create scatter plots for the two regimes with different colors and markers
#scatter1 = ax.plot(X[inds1,0], fcn1.pred()[inds1], marker='o', markeredgewidth = 1.5, markersize = 12, color='lightsteelblue', markeredgecolor='black', label="Regime 1 (specialized model)")
#scatter2 = ax.plot(X[inds2,0], fcn2.pred()[inds2], marker='s', markeredgewidth = 1.5, markersize = 12, color='violet', markeredgecolor='black', label="Regime 2 (specialized model)")
ypred = fcn_list[0].pred().detach().numpy()
scatter3 = ax.plot(X[:,0], ypred, '^', markeredgewidth = 1.5, markersize = 20, color='wheat', markeredgecolor='black', label="global model")

actual = ax.plot(x, y, "-", color = "midnightblue", linewidth = 4.5, label = "true function")

# Create a legend to label the different regimes
#legend = plt.legend(fontsize=28, loc='upper left')
leg = plt.legend(fontsize = 28)
leg.get_frame().set_edgecolor('k')

font = {'fontname':'Times New Roman',
        'color':  'black',
        'weight': 'normal',
        'style': 'italic',
        'size': 48,
        }

plt.yticks([-3, -1, 0, 1, 3], fontsize = 36)
plt.xticks([-30, -20, -10, 0, 10, 20, 30], fontsize = 36)
plt.xlabel('x', fontsize = 48, fontdict=font)
plt.ylabel('y', fontsize = 48, fontdict=font)
plt.xlim(-16, 31)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2.5)


plt.tight_layout()
plt.show()