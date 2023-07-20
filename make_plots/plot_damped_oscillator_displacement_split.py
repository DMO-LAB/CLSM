# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 15:59:18 2023

@author: oowoyele
"""

# In[1]:

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
import pickle

def save_obj(obj, filename):
    """Saves a Python object to a file using pickle."""
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def load_obj(filename):
    """Loads a Python object from a file using pickle."""
    with open(filename, 'rb') as f:
        return pickle.load(f)

# In[2]:


m = 1.0
c1 = 0.25
k1 = 10
delta = 0.25

c2 = 0.15
k2 = 5
# sYstem of differential equations
def ode1(y, t):
    y1, y2 = y
    dy1dt = y2

    dy2dt = -c1/m*y2 - k1/m*y1 + 0.1*t

    if y1 > delta:
        dy2dt += -c2/m*y2 - k2/m*(y1-delta)
    
    return [dy1dt, dy2dt]



# In[3]:


param = ['displacement', 'velocity']

# initial conditions
y0 = [1, 0]

# time points
t = np.linspace(0, 20, 200)

# solve ODE
y = odeint(ode1, y0, t)


# In[5]:


y = np.array(y)
y1 = y[:,0]
y2 = y[:,1]

yl0 = np.where(y1 > delta)[0]
y2dot = -c1/m*y2 - k1/m*y1 + 0.1*t
y2dot[yl0] = y2dot[yl0] + -c2/m*y2[yl0] - k2/m*(y1[yl0]-delta)
#y1dot = y2

x1 = y1
x2 = y2
x3 = y1*y2
x4 = y1**2
x5 = y2**2
x6 = t
x7 = y1**3
x8 = y2**3
X = np.column_stack([x1,x2,x3,x4,x5,x6,x7,x8])
y2dot = y2dot.reshape(-1,1) 

features_name = ['y1','y2','y1y2', 'y1_2', 'y2_2' ,'t','y1_3','y2_3','bias']


# In[6]:

data_ = np.concatenate([y[:,0:1], np.array(t)[:,None]], axis = 1)
scaler = MinMaxScaler()

data_n = scaler.fit_transform(data_)

# In[11]:


num_inputs = len(X)
num_targets = len(y2dot)
inp = X
out = y2dot
lam = 1e-6


# In[ ]:



# In[19]:

filename = '../saved_models/spring_mass_displacement_models/fcn_list.pkl'
fcn_list = load_obj(filename)

filename = '../saved_models/spring_mass_displacement_models/opt.pkl'
opt = load_obj(filename)

filename = '../saved_models/spring_mass_displacement_models/moe.pkl'
moe = load_obj(filename)

if moe.smoothen_alpha == True:
    alpha = moe.alpha_smooth
else:
    alpha = moe.alpha

################################################################################################

# In[35]:

w = []
for ii in range(moe.num_experts):
    w += [fcn_list[ii].weights.detach().numpy().reshape(-1)]
    df = pd.DataFrame([w[-1]], columns=[features_name])
    print(df)

inds = moe.get_winning_points_inds()

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

alpha_np = alpha.detach().numpy()
#alpha_np = np.round(alpha_np)
lab = np.argmax(alpha_np, axis = 1)
#clf = AdaBoostClassifier()
#clf = KNeighborsClassifier(n_neighbors=5)
clf = RandomForestClassifier()
clf.fit(data_n, lab)


# In[ ]:


# sYstem of differential equations
def ode_pred(y, t):
    #print(clf.predict(y[None,:]))
    
    state = np.concatenate(([[y[0]]], [[t]]), axis = 1)
    state = scaler.transform(state)
    
    #print(state)
    #print(clf.predict(state))
    #cqc=ece
    y1, y2 = y
    dy1dt = y2
    
    x1 = y1
    x2 = y2
    x3 = y1*y2
    x4 = y1**2
    x5 = y2**2
    x6 = t
    x7 = y1**3
    x8 = y2**3
    
    #w1 = w[0]
    #w2 = w[1]
    #if t < 2:
    #if clf.predict(state)[0][0] == 1: 
    if clf.predict(state)[0] == 0: 
        dy2dt = w[0][0]*x1 + w[0][1]*x2 + w[0][2]*x3 + w[0][3]*x4 + w[0][4]*x5 + w[0][5]*x6 + w[0][6]*x7 + w[0][7]*x8 + w[0][8]
    else:
        dy2dt = w[1][0]*x1 + w[1][1]*x2 + w[1][2]*x3 + w[1][3]*x4 + w[1][4]*x5 + w[1][5]*x6 + w[1][6]*x7 + w[1][7]*x8 + w[1][8]
    
    #print(x1,x2,x3,x4,x5,x6,x7)
    return [dy1dt, dy2dt]


font = {'fontname':'Times New Roman',
        'color':  'black',
        'weight': 'normal',
        'style': 'italic',
        'size': 28,
        }
lw = 5
# solve ODE
ypred = odeint(ode_pred, y0, t)

ax = plt.figure(figsize = (10,10)).add_subplot(projection='3d')

param = ["regime 1 (predicted)", "regime 2 (predicted)"]
colors = ['magenta', 'deepskyblue']

data_pred = np.concatenate([ypred[:,0:1], np.array(t)[:,None]], axis = 1)
data_pred_n = scaler.fit_transform(data_pred)
labels = clf.predict(data_pred_n)


for ii in range(0,2,1):
    #r1_inds = np.where(np.argmax(labels, axis = 1) == ii)[0]
    r1_inds = np.where(labels == ii)[0]
    plt.plot(t[r1_inds], ypred[r1_inds, 1],ypred[r1_inds, 0], '.', markersize = 20, markeredgecolor = "black",linewidth = lw, color=colors[ii], label=param[ii])

#r2_inds = np.where(np.argmax(labels, axis = 1) == 1)[0]
#plt.plot(t[r2_inds], ypred[r2_inds, 1],ypred[r2_inds, 0], '.', markersize = 15, markeredgecolor = "black", linewidth = lw,  color=colors[ii], label=param[ii])

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
#plt.tight_layout()
ax.dist = 12
plt.show()



fig = plt.figure(figsize = (12,8))
ax = fig.gca()

param = ["regime 1 (predicted)", "regime 2 (predicted)"]


for ii in range(0,2,1):
    #r1_inds = np.where(np.argmax(labels, axis = 1) == ii)[0]
    r1_inds = np.where(labels == ii)[0]
    ax.plot(t[r1_inds], ypred[r1_inds, 0], '.', markersize = 25, markeredgecolor = "black",linewidth = lw, color=colors[ii], label=param[ii])
    
#ii = 0
#ax.plot(t[inds[ii]], ypred[inds[ii], 0], '.', markersize = 15, markeredgecolor = "black",linewidth = lw, color=colors[ii], label=param[ii])

#ii = 1
#ax.plot(t[inds[ii]], ypred[inds[ii], 0], '.', markersize = 15, markeredgecolor = "black", linewidth = lw,  color=colors[ii], label=param[ii])


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


