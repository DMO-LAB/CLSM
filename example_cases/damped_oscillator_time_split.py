# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 13:45:27 2023

@author: oowoyele
"""


# In[1]:


import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from MLP import MLP
from ut.MLP_2 import MLP
from ut.optimize import optimizerMoE, optimizerMoE2, optimizerMoE3
from clsm import MoE
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


m = 0.75
c = 0.05
k = 2.4

# sYstem of differential equations
def ode1(y, t):
    y1, y2 = y
    dy1dt = y2

    f2 = 0
    if t < 2:
        f2 = 2*t
    #elif t > 7:
    #    f2 = 2
    
    dy2dt = -c/m*y2 - k/m*y1 + f2
    
    return [dy1dt, dy2dt]



# In[3]:


param = ['displacement', 'velocity']

# initial conditions
y0 = [2, 0]

# time points
t = np.linspace(0, 25, 400)

# solve ODE
y = odeint(ode1, y0, t)



# In[5]:


y = np.array(y)
y1 = y[:,0]
y2 = y[:,1]

tl0 = np.where(t < 2)[0]
#t20 = np.where(t > 7)[0]
y2dot = -c/m*y2 - k/m*y1
y2dot[tl0] = y2dot[tl0] + t[tl0]*2
#y2dot[t20] = y2dot[t20] + 2
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

learning_rate = 1

best_overall_error = 10000
for itrial in range(5):

    fcn1 = MLP(inp, out,Lasso_reg = True, lambda_reg = lam, annstruct = [X.shape[1],1], dtype = torch.float64)
    fcn2 = MLP(inp, out,Lasso_reg = True, lambda_reg = lam, annstruct = [X.shape[1],1], dtype = torch.float64)
    
    fcn_list = [fcn1, fcn2]
    
    opt2 = optimizerMoE2(fcn_list = fcn_list)
    
    moe = MoE(fcn_list, kappa = 0.1, smoothen_alpha = True, n_neighbors = 10, states = data_n)

    for it in range(200000):
        kk = np.random.permutation(400)[:400]
        
        loss_list = moe.compute_weighted_mse()
        loss_ = [loss.detach().numpy() for loss in loss_list]
        
        # Check if any value in loss list is NaN
        if any(np.isnan(loss_value) for loss_value in loss_):
            print(f"Error: Loss value is NaN at iteration {it}")
            break
        
    
        
        wp_np = [nwp for nwp in moe.get_num_winning_points()]
        alpha = moe.compute_alpha()#*0 #+ 1
        
    
        #alpha = smoothen_alpha(alpha, data_n, n_neighbors = 10)
        if moe.smoothen_alpha == True:
            alpha = moe.alpha_smooth
        else:
            alpha = moe.alpha
            
        wp_np = [nwp for nwp in moe.get_num_winning_points()]
        
        opt2.step(alpha,learning_rate)
        
        overall_error = np.sum([wp_np[ii]*loss_[ii] for ii in np.arange(moe.num_experts)])/inp.shape[0]
        if it%5000== 0:
            print(it, loss_, overall_error, wp_np)
    
    print("######################################################",  )
    print("overall error from trial ", str(itrial+1), " = ",  overall_error)
    print("######################################################",  )
    
    if overall_error < best_overall_error:
        print("updating models since better trial was found...")
        filename = './spring_mass_time_models/fcn_list.pkl'
        save_obj(fcn_list, filename)
        
        filename = './spring_mass_time_models/opt.pkl'
        save_obj(opt2, filename)
        
        filename = './spring_mass_time_models/moe.pkl'
        save_obj(moe, filename)
        
        best_overall_error = overall_error

################################################################################################

