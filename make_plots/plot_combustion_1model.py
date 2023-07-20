# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 16:36:54 2023

@author: oowoyele
"""

import sys

sys.path.append('../')

import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from MLP import MLP
from ut.MLP_2 import MLP
from ut.optimize import optimizerMoE, optimizerMoE2, optimizerMoE3
from clsm import MoE
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint
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

x1 = np.loadtxt('../flamespeed_data/flameSpeed.txt')[0::step,].T.ravel()
x2 = np.loadtxt('../flamespeed_data/flameSpeed_350')[0::step,].T.ravel()
x3 = np.loadtxt('../flamespeed_data/flameSpeed_400')[0::step,].T.ravel()
x4 = np.loadtxt('../flamespeed_data/flameSpeed_450')[0::step,].T.ravel()
x5 = np.loadtxt('../flamespeed_data/flameSpeed_500')[0::step,].T.ravel()

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

x1 = np.loadtxt('../flamespeed_data/flameSpeed.txt')[1::step,].T.ravel()
x2 = np.loadtxt('../flamespeed_data/flameSpeed_350')[1::step,].T.ravel()
x3 = np.loadtxt('../flamespeed_data/flameSpeed_400')[1::step,].T.ravel()
x4 = np.loadtxt('../flamespeed_data/flameSpeed_450')[1::step,].T.ravel()
x5 = np.loadtxt('../flamespeed_data/flameSpeed_500')[1::step,].T.ravel()

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

filename = '../saved_models/flamespeed_1model/fcn_list.pkl'
fcn_list = load_obj(filename)

filename = '../saved_models/flamespeed_1model/opt.pkl'
opt = load_obj(filename)

filename = '../saved_models/flamespeed_1model/moe.pkl'
moe = load_obj(filename)


################################################################################################
df = pd.DataFrame(X,columns = ['P','T','phi'])
df['Sl'] = Y


df['P'] = Xn[:,0]
df['T'] = Xn[:,1]
df['phi'] = Xn[:,2]

## Test result for 2 experts
ypred = []
for i in range (0,len(X_test)):
    ypred.append(fcn_list[0].pred_new(X_test[i,:].reshape(-1, 3)))
    
ypred = np.array(ypred).reshape(-1,1)
MSE = mean_squared_error(ypred,y_test)
print(MSE)

###################################################################
df = pd.DataFrame(Xn,columns = ['P','T','phi'])
df['Sl'] = Y
df[['P','T','phi']] = X

df = df.iloc[0::1]

df1 = df[~df['P'].isin([1,3,5,7,9])]
df1 = df1[~df1['T'].isin([350,450])]

unique_P = np.unique(df1['P'])
unique_T = np.unique(df1['T'])[::-1]

def predict(X_test):
    ypred = []
    for i in range (0,len(X_test)):
        ypred.append(fcn_list[0].pred_new(X_test[i,:].reshape(-1, 3)))
        
    return np.array(ypred).reshape(-1,1)

#print(X_test[i,:].reshape(-1, 3))
Xt_unnorm = X_test*(xmax - xmin) + xmin

#print(Xt_unnorm)

ax = plt.figure(figsize = (10,10)).add_subplot(projection='3d')

cl = ['lightsteelblue', 'violet', 'greenyellow']
lb = ["partition 1", "partition 2", "partition 3"]

font = {'fontname':'Times New Roman',
        'color':  'black',
        'weight': 'normal',
        'style': 'italic',
        'size': 28,
        }

first_plot = True
first_plot_ = True
for T in unique_T:
    for P in unique_P:
        #print(P,T)
        plt_bool = np.logical_and(Xt_unnorm[:,0] == P, Xt_unnorm[:,1] == T)
        y_pred = predict(X_test[plt_bool])
        #print(y_test)

        X_test_ = X_test[plt_bool]; y_test_ = y_test[plt_bool,0]
        
        ii = 0 # only one expert 
        X_test_unn = X_test_*(xmax - xmin) + xmin
        y_pred_unn = y_pred*(ymax - ymin) + ymin
        y_test_unn = y_test_*(ymax - ymin) + ymin

        if first_plot:
            plt.plot(X_test_unn[:,2], y_pred_unn[:], X_test_unn[:,1], 'o', markeredgewidth = 1.0, markersize = 8, color=cl[ii], markeredgecolor='black', label="NN prediction")
            make_plot_false = True  
        else:
            plt.plot(X_test_unn[:,2], y_pred_unn[:], X_test_unn[:,1], 'o', markeredgewidth = 1.0, markersize = 8, color=cl[ii], markeredgecolor='black')

        if make_plot_false:
            first_plot = False
        
        if first_plot_:
            plt.plot(X_test_unn[:,2], y_test_unn, X_test_unn[:,1], "-", color = "blue", linewidth = 2.5, label = r"true $s_L$")
            first_plot_ = False
        else:
            plt.plot(X_test_unn[:,2], y_test_unn, X_test_unn[:,1], "-", color = "blue", linewidth = 2.5)
            
leg = ax.legend(fontsize = 18, loc="upper left")
leg.get_frame().set_edgecolor('k')


ax.set_xticks(ticks = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6], labels = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6], fontsize  = 18)
ax.set_yticks(ticks = [0, 20, 40, 60], labels = [0, 20, 40, 60], fontsize  = 18)
ax.tick_params(axis='z', which='major', pad=8)
ax.set_zticks(ticks = [300, 350, 400, 450, 500], labels = [300, 350, 400, 450, 500], fontsize  = 18)

ax.set_xlabel(r'$\phi$', fontdict=font, labelpad=18)
ax.set_zlabel('T (K)', fontdict=font, labelpad=18)
ax.set_ylabel(r'$s_L$ (cm/s)', fontdict=font, labelpad=18)

ax.xaxis.set_tick_params(color='white')
ax.grid(False)
ax.xaxis.pane.set_edgecolor('#D0D0D0')
ax.yaxis.pane.set_edgecolor('#D0D0D0')
ax.zaxis.pane.set_edgecolor('#D0D0D0')
ax.xaxis.pane.set_alpha(0.4)
ax.yaxis.pane.set_alpha(0.5)
ax.zaxis.pane.set_alpha(0.5)

ax.dist = 12

#plt.savefig("flamespeed_class.png", dpi=600, format="png", pad_inches=0.0)
plt.show()
#############################################################
#print(X_test[i,:].reshape(-1, 3))
Xt_unnorm = X_test*(xmax - xmin) + xmin

#print(Xt_unnorm)

ax = plt.figure(figsize = (10,10))

cl = ['lightsteelblue', 'violet', 'greenyellow']
lb = ["partition 1", "partition 2", "partition 3"]

font = {'fontname':'Times New Roman',
        'color':  'black',
        'weight': 'normal',
        'size': 28,
        }

first_plot = True
first_plot_ = True
y_actual_all = []
y_pred_all = []
for T in unique_T:
    for P in unique_P:
        #print(P,T)
        plt_bool = np.logical_and(Xt_unnorm[:,0] == P, Xt_unnorm[:,1] == T)
        y_pred = predict(X_test[plt_bool])
        #print(y_test)
        #plt.plot(y_pred[plt_bool,0], y_test[plt_bool,0])
        X_test_ = X_test[plt_bool]; y_test_ = y_test[plt_bool,0]

        X_test_unn = X_test_*(xmax - xmin) + xmin
        y_test_unn = y_test_*(ymax - ymin) + ymin
        y_pred_unn = y_pred*(ymax - ymin) + ymin

        y_actual_all += [y_test_unn[:]]
        y_pred_all += [y_pred_unn[:,0]]
        
        plt.plot(y_pred_unn[:], y_test_unn[:], 'o', markeredgewidth = 1.8, markersize = 20, color=cl[ii], markeredgecolor='black')

y_actual_all = np.hstack(y_actual_all)
y_pred_all = np.hstack(y_pred_all)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy import stats

model = LinearRegression()
model.fit(y_pred_all.reshape(-1, 1), y_actual_all.reshape(-1, 1))
y_actual_all_pred = model.predict(y_pred_all.reshape(-1, 1))

# Calculate R-squared value
slope, intercept, r_value, p_value, std_err = stats.linregress(y_pred_all, y_actual_all)

plt.plot(y_pred_all, y_actual_all_pred, color='blue', linewidth = 6.5)#, label='Line of Best Fit')

print("R2 value = ", r_value**2)

plt.yticks([0, 15, 30, 45, 60], fontsize = 36)
plt.xticks([0, 15, 30, 45, 60], fontsize = 36)
plt.xlabel('predicted ' + r'$s_L$ (cm/s)', fontsize = 48, fontdict=font)
plt.ylabel('true ' + r'$s_L$ (cm/s)', fontsize = 48, fontdict=font)
plt.xlim(y_actual_all.min(),y_actual_all.max())
plt.ylim(y_actual_all.min(),y_actual_all.max())
plt.show()