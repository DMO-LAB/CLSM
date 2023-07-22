# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 15:59:18 2023

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

# Include parent directory in path
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


# Constants
m = 1.0
c1 = 0.25
k1 = 10
delta = 0.25
c2 = 0.15
k2 = 5


def ode1(y, t):
    """ODE system definition."""
    y1, y2 = y
    dy1dt = y2
    dy2dt = -c1/m * y2 - k1/m * y1 + 0.1 * t
    if y1 > delta:
        dy2dt += -c2/m * y2 - k2/m * (y1 - delta)
    return [dy1dt, dy2dt]


# Initial conditions and time points
y0 = [1, 0]
t = np.linspace(0, 20, 200)

# Solve ODE
y = odeint(ode1, y0, t)
y = np.array(y)
y1 = y[:, 0]
y2 = y[:, 1]

# Calculate y2dot based on conditions
yl0 = np.where(y1 > delta)[0]
y2dot = -c1/m * y2 - k1/m * y1 + 0.1 * t
y2dot[yl0] = y2dot[yl0] + -c2/m * y2[yl0] - k2/m * (y1[yl0] - delta)

# Feature construction
x1 = y1
x2 = y2
x3 = y1 * y2
x4 = y1 ** 2
x5 = y2 ** 2
x6 = t
x7 = y1 ** 3
x8 = y2 ** 3
X = np.column_stack([x1, x2, x3, x4, x5, x6, x7, x8])
y2dot = y2dot.reshape(-1, 1)

features_name = ['y1', 'y2', 'y1y2', 'y1_2', 'y2_2', 't', 'y1_3', 'y2_3', 'bias']
data_ = np.concatenate([y[:, 0:1], np.array(t)[:, None]], axis=1)
scaler = MinMaxScaler()
data_n = scaler.fit_transform(data_)

filename = 'saved_models/spring_mass_displacement_models/fcn_list.pkl'
fcn_list = load_obj(filename)

filename = 'saved_models/spring_mass_displacement_models/opt.pkl'
opt = load_obj(filename)

filename = 'saved_models/spring_mass_displacement_models/moe.pkl'
moe = load_obj(filename)

alpha = moe.alpha_smooth if moe.smoothen_alpha else moe.alpha

# Extract model weights
weights = []
for ii in range(moe.num_experts):
    weights.append(fcn_list[ii].weights.detach().numpy().reshape(-1))
    df = pd.DataFrame([weights[-1]], columns=[features_name])
    print(df)

inds = moe.get_winning_points_inds()

alpha_np = alpha.detach().numpy()
lab = np.argmax(alpha_np, axis=1)
clf = RandomForestClassifier()
clf.fit(data_n, lab)


def ode_pred(y, t):
    """Prediction using ODE with different dynamics based on classifier."""
    state = np.concatenate(([[y[0]]], [[t]]), axis=1)
    state = scaler.transform(state)
    y1, y2 = y
    dy1dt = y2

    x1 = y1
    x2 = y2
    x3 = y1 * y2
    x4 = y1 ** 2
    x5 = y2 ** 2
    x6 = t
    x7 = y1 ** 3
    x8 = y2 ** 3

    if clf.predict(state)[0] == 0:
        dy2dt = sum([weights[0][i] * x for i, x in enumerate([x1, x2, x3, x4, x5, x6, x7, x8])]) + weights[0][8]
    else:
        dy2dt = sum([weights[1][i] * x for i, x in enumerate([x1, x2, x3, x4, x5, x6, x7, x8])]) + weights[1][8]

    return [dy1dt, dy2dt]


# Plotting
ypred = odeint(ode_pred, y0, t)

font = {'fontname':'Times New Roman',
        'color':  'black',
        'weight': 'normal',
        'style': 'italic',
        'size': 28,
        }
lw = 5
ax = plt.figure(figsize = (10,10)).add_subplot(projection='3d')

param = ["regime 1 (predicted)", "regime 2 (predicted)"]
colors = ['magenta', 'deepskyblue']

data_pred = np.concatenate([ypred[:,0:1], np.array(t)[:,None]], axis = 1)
data_pred_n = scaler.fit_transform(data_pred)
labels = clf.predict(data_pred_n)


for ii in range(0,2,1):

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


for ii in range(0,2,1):
    #r1_inds = np.where(np.argmax(labels, axis = 1) == ii)[0]
    r1_inds = np.where(labels == ii)[0]
    ax.plot(t[r1_inds], ypred[r1_inds, 0], '.', markersize = 25, markeredgecolor = "black",linewidth = lw, color=colors[ii], label=param[ii])
    
ax.plot(t, y[:, 0], '-', linewidth = 4.5, color = "black", label="true dynamics")

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

ax.axhspan(np.min(t)-2, 0.25, facecolor='violet', alpha=0.3)
ax.axhspan(0.25, np.max(t)+2, facecolor='lightsteelblue', alpha=0.3)

plt.yticks([-1, -0.5, 0, 0.5, 1], fontsize = 36)
plt.xticks([0, 5, 10, 15, 20], fontsize = 36)
plt.xlabel('t', fontsize = 48, fontdict=font)
plt.ylabel('y', fontsize = 48, fontdict=font)

plt.xlim(0, 20)
plt.ylim(-1.2, 1.2)

for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2.5)

plt.show()


