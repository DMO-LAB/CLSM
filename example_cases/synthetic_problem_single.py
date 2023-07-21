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
from algorithm.optimize import optimizerMoE2


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
            y[i] = 0.2 * np.sin(x[i])
        else:
            y[i] = 0.1 * x[i] * np.cos(x[i])
    return x, y


x, y = step_function(0.5)

x1 = x
x2 = np.sin(x)
x3 = np.cos(x)
x4 = x1 * x2
x5 = x2 * x3
x6 = x1 * x3

features_name = ['x', 'sinx', 'cosx', 'xsinx', 'sinxcosx', 'xcosx', 'bias']

X = np.column_stack([x1, x2, x3, x4, x5, x6])
y = y.reshape(-1, 1)

noise_factor = 1e-2
noise = np.random.normal(0, noise_factor, size=y.shape)
y += noise

lam = 1e-4
learning_rate = 1
best_overall_error = 10000

for itrial in range(5):
    fcn = MLP(X, y, Lasso_reg=True, lambda_reg=lam, annstruct=[X.shape[1], 1], dtype=torch.float64)
    fcn_list = [fcn]
    opt = optimizerMoE2(fcn_list=[fcn])
    moe = CLSM(fcn_list, kappa=0.1, smoothen_alpha=True, n_neighbors=10, states=X)

    for it in range(5000):
        loss_list = moe.compute_weighted_mse()
        loss_ = [loss.detach().numpy() for loss in loss_list]

        if any(np.isnan(loss_value) for loss_value in loss_):
            print(f"Error: Loss value is NaN at iteration {it}")
            break

        wp_np = [nwp for nwp in moe.get_num_winning_points()]
        alpha = moe.compute_alpha()

        if moe.smoothen_alpha:
            alpha = moe.alpha_smooth
        else:
            alpha = moe.alpha

        opt.step(alpha, learning_rate)

        overall_error = np.sum([wp_np[ii] * loss_[ii] for ii in np.arange(moe.num_experts)]) / X.shape[0]
        if it % 500 == 0:
            print(it, loss_, overall_error, wp_np)

    print("######################################################")
    print("Overall error from trial ", str(itrial + 1), " = ", overall_error)
    print("######################################################")

    if overall_error < best_overall_error:
        print("Updating models since better trial was found...")
        filename = 'saved_models/synthetic_1model/fcn_list.pkl'
        save_obj(fcn_list, filename)

        filename = 'saved_models/synthetic_1model/opt.pkl'
        save_obj(opt, filename)

        filename = 'saved_models/synthetic_1model/moe.pkl'
        save_obj(moe, filename)

        best_overall_error = overall_error

# Plotting
fig = plt.figure(figsize=(12, 8))
ax = fig.gca()

ypred = fcn.pred().detach().numpy()
scatter3 = ax.plot(X[:, 0], ypred, '^', markeredgewidth=1.5, markersize=12, color='wheat', markeredgecolor='black',
                   label="global model")

ax.plot(x, y, "-.", color="midnightblue", linewidth=2.5, label="true function")

leg = plt.legend(fontsize=28)
leg.get_frame().set_edgecolor('k')

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
