# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 16:22:00 2023

@author: oowoyele
"""
# Import necessary modules and libraries
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
from algorithm.create_models import CreateModel
from algorithm.clsm import CLSM
from algorithm.optimizers import OptimizerCLSMAdam
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import warnings

# Ignore warning messages
warnings.filterwarnings('ignore')


def save_obj(obj, filename):
    """Save a Python object to a file using pickle."""
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def load_obj(filename):
    """Load a Python object from a file using pickle."""
    with open(filename, 'rb') as f:
        return pickle.load(f)


def preprocess_data(step, prefix='flamespeed_data/'):
    """Load and preprocess the flame speed data based on a given step."""
    x1 = np.loadtxt(prefix + 'flameSpeed.txt')[::step, ].T.ravel()
    x2 = np.loadtxt(prefix + 'flameSpeed_350')[::step, ].T.ravel()
    x3 = np.loadtxt(prefix + 'flameSpeed_400')[::step, ].T.ravel()
    x4 = np.loadtxt(prefix + 'flameSpeed_450')[::step, ].T.ravel()
    x5 = np.loadtxt(prefix + 'flameSpeed_500')[::step, ].T.ravel()

    y = np.vstack([x1, x2, x3, x4, x5]).ravel()
    phi = np.array(list(np.arange(0.6, 1.6, 0.01)[::step]) * 10 * 5)  # 10 represents pressure, 5 represents T
    p0 = np.array(list(np.repeat(np.linspace(1, 10, 10), len(x1) / 10)) * 5)
    t0 = np.array([298, 350, 400, 450, 500])
    t0 = np.repeat(t0, len(x1))
    x = np.vstack([p0, t0, phi]).T
    y = y.reshape(-1, 1)

    return x, y


# Load and preprocess training data
X, y = preprocess_data(step=2)
# Normalize data
xmin = X.min(axis=0)
xmax = X.max(axis=0)
X_normalized = (X - xmin) / (xmax - xmin)

ymin = y.min(axis=0)
ymax = y.max(axis=0)
y_normalized = (y - ymin) / (ymax - ymin)

# Load and preprocess testing data
X_test, y_test = preprocess_data(step=3)
# Normalize test data
X_test_normalized = (X_test - xmin) / (xmax - xmin)
y_test_normalized = (y_test - ymin) / (ymax - ymin)

# Model training parameters
num_inputs = len(X)
num_targets = len(y)
best_overall_error = float('inf')

# Training loop for multiple trials
for trial in range(5):

    # Create the model and initialize parameters
    model = CreateModel(X_normalized, y_normalized, ann_struct=[3, 3, 1], activation='sigmoid',
                        lin_output=True, dtype=torch.float64)
    clsm = CLSM([model], kappa=1, smoothen_alpha=False)
    optimizer = OptimizerCLSMAdam([model], learning_rate=0.01)

    # Optimization loop
    for iteration in range(2000):
        loss_list = clsm.compute_weighted_mse(iteration)
        losses = [loss.detach().numpy() for loss in loss_list]
        num_winning_points = [points for points in clsm.get_num_winning_points()]
        optimizer.step(loss_list, iter=iteration, moe=clsm)

        overall_error = np.sum([num_winning_points[i] * losses[i] for i in range(clsm.num_experts)]) / X_normalized.shape[0]
        
        # Display progress
        if iteration % 10000 == 0:
            print(iteration, losses, overall_error, num_winning_points)

    # Display trial results
    print("######################################################")
    print(f"Overall error from trial {trial + 1} = {overall_error}")
    print("######################################################")

    # Check and save the best model
    if overall_error < best_overall_error:
        print("Updating models since better trial was found...")
        save_obj([model], 'saved_models/flamespeed_1model/model.pkl')
        save_obj(optimizer, 'saved_models/flamespeed_1model/optimizer.pkl')
        save_obj(clsm, 'saved_models/flamespeed_1model/clsm.pkl')
        best_overall_error = overall_error
