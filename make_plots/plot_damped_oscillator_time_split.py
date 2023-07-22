# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 18:33:36 2023

@author: oowoyele
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from scipy.integrate import odeint

# Add the parent directory to the system path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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


# Define constants for the spring-mass system
m = 0.75
c = 0.05
k = 2.4

def system_ode(y, t):
    """ODE system for the spring-mass system."""
    y1, y2 = y
    dy1dt = y2
    f2 = 0
    if t < 2:
        f2 = 2 * t

    dy2dt = -c/m * y2 - k/m * y1 + f2
    return [dy1dt, dy2dt]

# Initial conditions and time points
y0 = [2, 0]
t = np.linspace(0, 25, 400)
y = odeint(system_ode, y0, t)

# Compute additional parameters from the ODE results
y2dot = -c/m * y[:,1] - k/m * y[:,0]
tl0 = np.where(t < 2)[0]
y2dot[tl0] = y2dot[tl0] + t[tl0]*2

# Feature extraction from the data
x = [y[:,0], y[:,1], y[:,0]*y[:,1], y[:,0]**2, y[:,1]**2, t, y[:,0]**3, y[:,1]**3]
X = np.column_stack(x)
y2dot = y2dot.reshape(-1, 1)

# Feature names for dataframe representation
features_name = ['y1','y2','y1y2', 'y1_2', 'y2_2' ,'t','y1_3','y2_3','bias']

# Normalizing data
data = np.concatenate([y[:,0:1], np.array(t)[:, None]], axis=1)
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data)

# Model related configurations
lam = 1e-6
fcn1 = CreateModel(X, y2dot, lasso_reg=True, lambda_reg=lam, ann_struct=[X.shape[1], 1], dtype=torch.float64)
fcn2 = CreateModel(X, y2dot, lasso_reg=True, lambda_reg=lam, ann_struct=[X.shape[1], 1], dtype=torch.float64)
fcn_list = [fcn1, fcn2]
optimizer = OptimizerCLSMNewton(fcn_list=fcn_list)

# CLSM configuration
moe = CLSM(fcn_list, kappa=0.1, smoothen_alpha=True, n_neighbors=10, states=data_normalized)
moe.kappa = 0.1

# Load pre-trained models
fcn_list = load_obj('saved_models/spring_mass_time_models/fcn_list.pkl')
optimizer = load_obj('saved_models/spring_mass_time_models/opt.pkl')
moe = load_obj('saved_models/spring_mass_time_models/moe.pkl')

# Determine the alpha values
alpha = moe.alpha_smooth if moe.smoothen_alpha else moe.alpha

# Extract the weights for each model
weights = []
for model in fcn_list:
    weight = model.weights.detach().numpy().reshape(-1)
    weights.append(weight)
    df = pd.DataFrame([weight], columns=features_name)
    print(df)

# Get indices of the winning points for each model
inds = moe.get_winning_points_inds()

# Classifier to predict the regime based on the states
clf = RandomForestClassifier()
alpha_np = alpha.detach().numpy()
labels = np.argmax(alpha_np, axis=1)
clf.fit(data_normalized, labels)

def predicted_ode(y, t):
    """ODE system using the model predictions."""
    state = np.concatenate(([[y[0]]], [[t]]), axis=1)
    state_normalized = scaler.transform(state)
    y1, y2 = y
    dy1dt = y2

    # Compute the weighted sum of inputs
    idx = clf.predict(state_normalized)[0]
    dy2dt = sum([weights[idx][i] * var for i, var in enumerate([y1, y2, y1*y2, y1**2, y2**2, t, y1**3, y2**3])])
    dy2dt += weights[idx][-1]  # Add the bias
    return [dy1dt, dy2dt]

# Solve ODE with the predicted model

ypred = odeint(predicted_ode, y0, t)



font = {'fontname':'Times New Roman',
        'color':  'black',
        'weight': 'normal',
        'style': 'italic',
        'size': 28,
        }
lw = 5

ax = plt.figure(figsize = (10,10)).add_subplot(projection='3d')

param = ["regime 1 (predicted)", "regime 2 (predicted)"]
colors = ['deepskyblue','magenta']

data_pred = np.concatenate([ypred[:,0:1], np.array(t)[:,None]], axis = 1)
data_pred_n = scaler.fit_transform(data_pred)
labels = clf.predict(data_pred_n)


for ii in range(2):
    r1_inds = np.where(labels == ii)[0]
    plt.plot(t[r1_inds], ypred[r1_inds, 1],ypred[r1_inds, 0], '.', markersize = 20, markeredgecolor = "black",linewidth = lw, color=colors[ii], label=param[ii])

plt.plot(t, y[:, 1],y[:, 0], '-', linewidth = 3.5, color = "black", label="true dynamics")

leg = ax.legend(fontsize = 18, loc="upper center")
leg.get_frame().set_edgecolor('k')


ax.set_xticks(ticks = [0, 10, 20], labels = [0, 10, 20], fontsize  = 18)
ax.set_yticks(ticks = [-2, 0, 2], labels = [-2, 0, 2], fontsize  = 18)
ax.set_zticks(ticks = [-1.0, 0.5, 2.0], labels = [-1.0, 0.5, 2.0], fontsize  = 18)

ax.set_xlabel('t', fontdict=font, labelpad=18)
ax.set_zlabel('y', fontdict=font, labelpad=18)
ax.set_ylabel('y', fontdict=font, labelpad=18)

ax.xaxis.set_tick_params(color='white')
ax.grid(False)
ax.xaxis.pane.set_edgecolor('#D0D0D0')
ax.yaxis.pane.set_edgecolor('#D0D0D0')
ax.zaxis.pane.set_edgecolor('#D0D0D0')
ax.xaxis.pane.set_alpha(0.8)
ax.yaxis.pane.set_alpha(0.8)
ax.zaxis.pane.set_alpha(0.8)
ax.dist = 12
plt.show()



fig = plt.figure(figsize = (12,8))
ax = fig.gca()

param = ["regime 1 (predicted)", "regime 2 (predicted)"]


for ii in range(2):
    r1_inds = np.where(labels == ii)[0]
    ax.plot(t[r1_inds], ypred[r1_inds, 0], '.', markersize = 20, markeredgecolor = "black",linewidth = lw, color=colors[ii], label=param[ii])


ax.plot(t, y[:, 0], '-', linewidth = 3.5, color = "black", label="true dynamics")

# Create a legend to label the different regimes
#legend = plt.legend(fontsize=28, loc='upper left')

leg = plt.legend(fontsize = 24)
leg.get_frame().set_edgecolor('k')

font = {'fontname':'Times New Roman',
        'color':  'black',
        'weight': 'normal',
        'style': 'italic',
        'size': 48,
        }

ax.axvspan(np.min(t)-1, 1.994747375, facecolor='lightsteelblue', alpha=0.3)
ax.axvspan(1.994747375, np.max(t)+2, facecolor='violet', alpha=0.3)

plt.yticks([-2, -1, 0, 1, 2], fontsize = 36)
plt.xticks([0, 5, 10, 15, 20], fontsize = 36)
plt.xlabel('t', fontsize = 48, fontdict=font)
plt.ylabel('y', fontsize = 48, fontdict=font)

plt.xlim(-0.1, 25)

for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2.5)

plt.show()

