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


# Load data from files
def load_data(file_path, step):
    """Load flame speed data."""
    return np.loadtxt(file_path)[::step,].T.ravel()


# Preprocess the input data
step = 2
x1 = load_data('flamespeed_data/flameSpeed.txt', step)
x2 = load_data('flamespeed_data/flameSpeed_350', step)
x3 = load_data('flamespeed_data/flameSpeed_400', step)
x4 = load_data('flamespeed_data/flameSpeed_450', step)
x5 = load_data('flamespeed_data/flameSpeed_500', step)

y = np.vstack([x1, x2, x3, x4, x5]).ravel()
phi = np.array(list(np.arange(0.6, 1.6, 0.01)[::step]) * 10 * 5)  # 10 rep pressure, 5 rep T
p0 = np.array(list(np.repeat(np.linspace(1, 10, 10), len(x1) / 10)) * 5)
t0 = np.array([298, 350, 400, 450, 500])
t0 = np.repeat(t0, len(x1))
X = np.vstack([p0, t0, phi]).T
y = y.reshape(-1, 1)

# Normalize the data
xmin = X.min(axis=0)
xmax = X.max(axis=0)
Xn = (X - xmin) / (xmax - xmin)  # normalizing to range 0-1

ymax = np.max(y, axis=0)
ymin = np.min(y, axis=0)
Y = (y - ymin) / (ymax - ymin)  # normalize source term to range 0-1

noise_factor = 3e-2
noise = np.random.normal(0, noise_factor, size=Y.shape)

# Test data preprocessing
step = 3
x1_test = load_data('flamespeed_data/flameSpeed.txt', step)
x2_test = load_data('flamespeed_data/flameSpeed_350', step)
x3_test = load_data('flamespeed_data/flameSpeed_400', step)
x4_test = load_data('flamespeed_data/flameSpeed_450', step)
x5_test = load_data('flamespeed_data/flameSpeed_500', step)

y_test = np.vstack([x1_test, x2_test, x3_test, x4_test, x5_test]).ravel()
phi_test = np.array(list(np.arange(0.6, 1.6, 0.01)[::step]) * 10 * 5)  # 10 rep pressure, 5 rep T
p0_test = np.array(list(np.repeat(np.linspace(1, 10, 10), len(x1_test) / 10)) * 5)
t0_test = np.array([298, 350, 400, 450, 500])
t0_test = np.repeat(t0_test, len(x1_test))
X_test = np.vstack([p0_test, t0_test, phi_test]).T
y_test = y_test.reshape(-1, 1)

# Normalize the test data
X_test = (X_test - xmin) / (xmax - xmin)
Y_test = (y_test - ymin) / (ymax - ymin)

noise = np.random.normal(0, noise_factor, size=Y_test.shape)
y_test = Y_test

num_inputs = len(X)
num_targets = len(y)
inp = Xn
out = Y

best_overall_error = 10000

for trial in range(5):
    fcn1 = CreateModel(inp, out, ann_struct=[3, 3, 1], activation='sigmoid', lin_output=True, dtype=torch.float64)
    fcn2 = CreateModel(inp, out, ann_struct=[3, 3, 1], activation='sigmoid', lin_output=True, dtype=torch.float64)
    fcn_list = [fcn1, fcn2]
    moe = CLSM(fcn_list, kappa=1, smoothen_alpha=False)

    opt = OptimizerCLSMAdam(fcn_list, learning_rate=0.01)

    for iteration in range(2000):
        loss_list = moe.compute_weighted_mse(iteration)
        loss_ = [loss.detach().numpy() for loss in loss_list]
        wp_np = [nwp for nwp in moe.get_num_winning_points()]
        opt.step(loss_list, iter=iteration, moe=moe)

        overall_error = np.sum([wp_np[ii] * loss_[ii] for ii in np.arange(moe.num_experts)]) / inp.shape[0]

        if iteration % 10000 == 0:
            print(iteration, loss_, overall_error, wp_np)

    print("######################################################")
    print("Overall error from trial", str(trial + 1), "=", overall_error)
    print("######################################################")

    if overall_error < best_overall_error:
        print("Updating models since better trial was found...")
        filename = 'saved_models/flamespeed_2models/fcn_list.pkl'
        save_obj(fcn_list, filename)

        filename = 'saved_models/flamespeed_2models/opt.pkl'
        save_obj(opt, filename)

        filename = 'saved_models/flamespeed_2models/moe.pkl'
        save_obj(moe, filename)

        best_overall_error = overall_error
    
################################################################################################