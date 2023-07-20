# -*- coding: utf-8 -*-
"""
Created on Thu May 25 19:54:23 2023

@author: oowoyele
"""

import torch
import sys
import numpy as np

class CLSM(): # fully connected neural network class
    def __init__(self, fcn_list, kappa = 5, min_ratio = 0.05, smoothen_alpha = True, n_neighbors = 5, states = None):
        # fcn_list = list of fully connected neural network objects, created using the MLP() class.
        # kappa is a parameter that controls how strongly we want to separate the experts
        #self.num_experts = num_experts
        self.num_experts = len(fcn_list)
        self.kappa = kappa
        self.fcn_list = fcn_list
        self.min_ratio = min_ratio
        self.alpha = None
        self.smoothen_alpha = smoothen_alpha
        if self.smoothen_alpha == True:
            self.n_neighbors = n_neighbors
            self.states = states
            self.num_observations, self.num_features = self.states.shape
        
    def compute_SE(self, fcn):
        return (fcn.output - fcn.y)**2

    def compute_MSE(self, fcn):
        return torch.mean((fcn.output - fcn.y)**2)
    
    def compute_alpha(self):
        # computes the weights for the MSE (stored as alpha)
        with torch.no_grad():
            
            errors = [self.compute_SE(fcn) for fcn in self.fcn_list]
            
            errors_mat = torch.concatenate(errors, axis = 1)
            c = torch.amin(errors_mat, axis = 1)[:,None]
            errors_mat_norm = errors_mat/c  #torch.amin(errors_mat, axis = 1)[:,None]
            denum = torch.sum(torch.exp(-self.kappa*errors_mat_norm), axis = 1)
    
            self.alpha = torch.exp(-self.kappa*errors_mat_norm)/denum[:,None]
        # print(self.alpha)        
        return self.alpha
    
    def get_alpha_avg(self):
        dist = np.zeros((self.num_observations, self.num_observations))
        for dim in range(self.num_features):
            dist += (self.states[:,dim:dim+1] - self.states[:,dim:dim+1].T)**2
        
        order = np.argsort(dist, axis=1)[:,:self.n_neighbors]
        
        alpha_avg = []
        for iexp in range(self.num_experts):
            alpha_ = self.alpha[:,iexp:iexp+1]
            #print(order)
            #print(alpha_.T[:,tuple(order)])
            alpha_neighbors = alpha_.T[:,np.array(tuple(order))][0]
            alpha_avg += [torch.mean(alpha_neighbors, dim = 1)[:,None]]
            
        self.alpha_avg = torch.cat(alpha_avg, dim = 1)
    
    def compute_weighted_MSE(self, fcn, alpha, update_y = True):
        # computes and returns the weighted (each sample is weighted using alpha)
        # if update_y is true, it updates the model predictions using latest weights before computing the MSE
        if update_y:
            fcn.pred()
        if fcn.Lasso_reg == True:
            loss = torch.mean(alpha*(fcn.output - fcn.y)**2) + fcn.lambda_reg*torch.sum(torch.abs(fcn.weights))
        else:
            loss = torch.mean(alpha*(fcn.output - fcn.y)**2)
        return loss
    
    def get_num_winning_points(self):
        # computes and returns the number of points "won" by each model

        num_wp = [len(torch.where(torch.argmax(self.alpha, axis=1) == iexp)[0].detach().numpy()) for iexp in torch.arange(self.num_experts)]

        return num_wp
    
    def get_winning_points_inds(self):
        # computes and returns the indices of points "won" by each model
        
        inds_exp = [torch.where(torch.argmax(self.alpha, axis=1) == iexp)[0] for iexp in torch.arange(self.num_experts)]
        return inds_exp
    
    
    def compute_weighted_mse(self, alpha_weighted = None, update_y = True):
        
        if self.smoothen_alpha:
            self.compute_alpha()
            self.get_alpha_avg()
            self.alpha_smooth = self.alpha * self.alpha_avg**5
            self.wmse = [self.compute_weighted_MSE(fcn, self.alpha_smooth[:,iexp:iexp+1], update_y) for iexp, fcn in enumerate(self.fcn_list)]
        else:
            self.alpha = self.compute_alpha()
            self.wmse = [self.compute_weighted_MSE(fcn, self.alpha[:,iexp:iexp+1], update_y) for iexp, fcn in enumerate(self.fcn_list)]

        return self.wmse