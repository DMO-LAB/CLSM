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
from algorithm.create_models import CreateModel
from algorithm.clsm import CLSM
from algorithm.optimizers import OptimizerCLSMNewton


def save_object(obj, filename):
    """Save a Python object to a file using pickle."""
    with open(filename, 'wb') as file:
        pickle.dump(obj, file)


def load_object(filename):
    """Load a Python object from a file using pickle."""
    with open(filename, 'rb') as file:
        return pickle.load(file)


def generate_step_function(step):
    """Generate a function based on the given step."""
    x = np.arange(-15, 30, step)
    y = np.zeros_like(x)
    for i, value in enumerate(x):
        if value <= 0:
            y[i] = 0.2 * np.sin(value)
        else:
            y[i] = 0.1 * value * np.cos(value)
    return x, y


# Generate data
x, y = generate_step_function(0.5)

# Create additional features based on the data
x1, x2, x3 = x, np.sin(x), np.cos(x)
x4, x5, x6 = x1 * x2, x2 * x3, x1 * x3

features_names = ['x', 'sinx', 'cosx', 'xsinx', 'sinxcosx', 'xcosx', 'bias']
X = np.column_stack([x1, x2, x3, x4, x5, x6])
y = y.reshape(-1, 1)

# Add noise to the data
noise_factor = 1e-2
noise = np.random.normal(0, noise_factor, size=y.shape)
y += noise

# Training parameters
lambda_reg = 1e-4
learning_rate = 1
best_overall_error = 1e4

for trial in range(5):
    fcn = CreateModel(input_tensor=X, output_tensor=y, lasso_reg=True, lambda_reg=lambda_reg, ann_struct=[X.shape[1], 1], dtype=torch.float64)
    fcn_list = [fcn]
    optimizer = OptimizerCLSMNewton(fcn_list=[fcn])
    moe_model = CLSM(fcn_list, kappa=0.1, smoothen_alpha=True, n_neighbors=10, states=X)

    for iteration in range(5000):
        loss_values = moe_model.compute_weighted_mse()
        loss_list = [loss.detach().numpy() for loss in loss_values]

        if any(np.isnan(loss) for loss in loss_list):
            print(f"Error: Loss value is NaN at iteration {iteration}")
            break

        winning_points = [nwp for nwp in moe_model.get_num_winning_points()]
        alpha = moe_model.compute_alpha()

        if moe_model.smoothen_alpha:
            alpha = moe_model.alpha_smooth

        optimizer.step(alpha, learning_rate)

        overall_error = np.sum([winning_points[i] * loss_list[i] for i in range(moe_model.num_experts)]) / X.shape[0]
        if iteration % 500 == 0:
            print(iteration, loss_list, overall_error, winning_points)

    print("######################################################")
    print(f"Overall error from trial {trial + 1} = {overall_error}")
    print("######################################################")

    if overall_error < best_overall_error:
        print("Updating models since a better trial was found...")
        
        # Save models
        save_object(fcn_list, 'saved_models/synthetic_1model/fcn_list.pkl')
        save_object(optimizer, 'saved_models/synthetic_1model/opt.pkl')
        save_object(moe_model, 'saved_models/synthetic_1model/moe.pkl')

        best_overall_error = overall_error

# Plot results
# Set up the figure and axis
fig = plt.figure(figsize = (12,8))
ax = fig.gca()
ypred = fcn.predict().detach().numpy()
scatter3 = ax.plot(X[:,0], ypred, '^', markeredgewidth = 1.5, markersize = 12, color='wheat', markeredgecolor='black', label="global model")

actual = ax.plot(x, y, "-.", color = "midnightblue", linewidth = 2.5, label = "true function")

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