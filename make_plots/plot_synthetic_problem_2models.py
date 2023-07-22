# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 22:36:33 2023

@author: oowoyele
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# Update sys path to include parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algorithm.create_models import CreateModel
from algorithm.clsm import CLSM
from algorithm.optimizers import OptimizerCLSMNewton


def load_obj(filename):
    """Loads a Python object from a file using pickle."""
    with open(filename, 'rb') as f:
        return pickle.load(f)


def step_function(step):
    """Generates the x and y values based on the step function."""
    x = np.arange(-15, 30, step)
    y = np.zeros_like(x)
    for i, j in enumerate(x):
        if j <= 0:
            y[i] = 0.2 * np.sin(x[i])
        else:
            y[i] = 0.1 * x[i] * np.cos(x[i])
    return x, y


# Generate step function data
x, y = step_function(0.5)

# Compute features
x1 = x
x2 = np.sin(x)
x3 = np.cos(x)
x4 = x1 * x2
x5 = x2 * x3
x6 = x1 * x3

features_name = ['x', 'sinx', 'cosx', 'xsinx', 'sinxcosx', 'xcosx', 'bias']

X = np.column_stack([x1, x2, x3, x4, x5, x6])
y = y.reshape(-1, 1)

# Add noise to the y values
noise_factor = 1e-2
noise = np.random.normal(0, noise_factor, size=y.shape)
y = y + noise

# Load models and optimizer
filename = 'saved_models/synthetic_2models/fcn_list.pkl'
fcn_list = load_obj(filename)

filename = 'saved_models/synthetic_2models/opt.pkl'
opt = load_obj(filename)

filename = 'saved_models/synthetic_2models/moe.pkl'
moe = load_obj(filename)

alpha = moe.alpha_smooth if moe.smoothen_alpha else moe.alpha

# Extract weights from models and display in DataFrame
w = []
for ii in range(moe.num_experts):
    w.append(fcn_list[ii].weights.detach().numpy().reshape(-1))
    df = pd.DataFrame([w[-1]], columns=features_name)
    print(df)

# Compute predictions based on trained models
alpha_np = alpha.detach().numpy()
lab = np.argmax(alpha_np, axis=1)
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X, lab)

ypred = np.zeros_like(y)
lab = clf.predict(X)
ypred[lab == 0, 0] = np.sum(w[0][:6] * X[lab == 0], axis=1) + w[0][6]
ypred[lab == 1, 0] = np.sum(w[1][:6] * X[lab == 1], axis=1) + w[1][6]

# Plotting the results
fig, ax = plt.subplots(figsize=(12, 8))

ypred1 = fcn_list[0].predict().detach().numpy()
ypred2 = fcn_list[1].predict().detach().numpy()

ax.plot(X[lab == 1, 0], ypred[lab == 1], marker='s', markeredgewidth=1.5, markersize=15, color='violet', markeredgecolor='black', label="Regime 1 (specialized model)")
ax.plot(X[lab == 0, 0], ypred[lab == 0], marker='o', markeredgewidth=1.5, markersize=20, color='lightsteelblue', markeredgecolor='black', label="Regime 2 (specialized model)")
ax.plot(x, y, "-", color="midnightblue", linewidth=4.5, label="true function")

leg = ax.legend(fontsize=28)
leg.get_frame().set_edgecolor('k')
ax.axvspan(np.min(x)-1, 0, facecolor='violet', alpha=0.4)
ax.axvspan(0, np.max(x)+2, facecolor='lightsteelblue', alpha=0.4)

font = {
    'fontname': 'Times New Roman',
    'color': 'black',
    'weight': 'normal',
    'style': 'italic',
    'size': 48,
}

plt.yticks([-3, -1, 0, 1, 3], fontsize=36)
plt.xticks([-30, -20, -10, 0, 10, 20, 30], fontsize=36)
plt.xlabel('x', fontsize=48, fontdict=font)
plt.ylabel('y', fontsize=48, fontdict=font)
plt.xlim(-16, 31)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(2.5)

plt.tight_layout()
plt.show()
