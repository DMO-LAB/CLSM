# -*- coding: utf-8 -*-
"""
Created on Thu May 25 19:54:23 2023
@author: oowoyele
"""

import torch
import numpy as np

class CLSM:
    """A fully connected neural network class (CLSM) which manages multiple experts (MLPs) and their predictions."""

    def __init__(self, fcn_list, kappa=5, min_ratio=0.05, smoothen_alpha=True, n_neighbors=5, states=None):
        """
        Initialize the CLSM instance.
        
        Args:
        - fcn_list (list): List of fully connected neural network objects, created using the MLP() class.
        - kappa (float): Controls the separation strength between the experts.
        - min_ratio (float): Minimal ratio.
        - smoothen_alpha (bool): Flag to determine if alpha values should be smoothened.
        - n_neighbors (int): Number of neighbors to consider for smoothening alpha.
        - states (torch.Tensor): Matrix of states, used for smoothening.
        """

        self.num_experts = len(fcn_list)
        self.kappa = kappa
        self.fcn_list = fcn_list
        self.min_ratio = min_ratio
        self.alpha = None
        self.smoothen_alpha = smoothen_alpha
        
        if self.smoothen_alpha:
            self.n_neighbors = n_neighbors
            self.states = states
            self.num_observations, self.num_features = self.states.shape

    def compute_SE(self, fcn):
        """Compute squared error for the given model."""
        return (fcn.output - fcn.y) ** 2

    def compute_MSE(self, fcn):
        """Compute mean squared error for the given model."""
        return torch.mean((fcn.output - fcn.y) ** 2)

    def compute_alpha(self):
        """
        Compute and return the alpha values (weights for the MSE) for all models.
        Updates the instance's alpha values.
        """

        with torch.no_grad():
            errors = [self.compute_SE(fcn) for fcn in self.fcn_list]
            errors_mat = torch.cat(errors, dim=1)
            c = torch.amin(errors_mat, dim=1, keepdim=True)
            errors_mat_norm = errors_mat / c
            denum = torch.sum(torch.exp(-self.kappa * errors_mat_norm), dim=1)
            self.alpha = torch.exp(-self.kappa * errors_mat_norm) / denum.unsqueeze(1)
            
        return self.alpha

    def get_alpha_avg(self):
        """Compute and update the average alpha values based on neighbors."""

        # Calculate pairwise distance
        dist = np.zeros((self.num_observations, self.num_observations))
        for dim in range(self.num_features):
            dist += (self.states[:, dim:dim+1] - self.states[:, dim:dim+1].T) ** 2

        # Find the closest neighbors for each observation
        order = np.argsort(dist, axis=1)[:, :self.n_neighbors]
        
        alpha_avg = []
        for iexp in range(self.num_experts):
            alpha_ = self.alpha[:, iexp:iexp+1]
            alpha_neighbors = alpha_.T[:, order][0]
            alpha_avg.append(torch.mean(alpha_neighbors, dim=1, keepdim=True))

        self.alpha_avg = torch.cat(alpha_avg, dim=1)

    def compute_weighted_MSE(self, fcn, alpha, update_y=True):
        """
        Compute the weighted mean squared error for the given model.
        
        Args:
        - fcn (MLP): The neural network model.
        - alpha (torch.Tensor): The alpha values for the current model.
        - update_y (bool): Flag to update the model's prediction.
        
        Returns:
        - torch.Tensor: The weighted mean squared error.
        """
        
        if update_y:
            fcn.pred()
        if fcn.Lasso_reg:
            return torch.mean(alpha * (fcn.output - fcn.y) ** 2) + fcn.lambda_reg * torch.sum(torch.abs(fcn.weights))
        else:
            return torch.mean(alpha * (fcn.output - fcn.y) ** 2)

    def get_num_winning_points(self):
        """Return the number of data points "won" by each model."""
        return [len(torch.where(torch.argmax(self.alpha, dim=1) == iexp)[0].detach().numpy()) for iexp in torch.arange(self.num_experts)]

    def get_winning_points_inds(self):
        """Return the indices of data points "won" by each model."""
        return [torch.where(torch.argmax(self.alpha, dim=1) == iexp)[0] for iexp in torch.arange(self.num_experts)]

    def compute_weighted_mse(self, alpha_weighted=None, update_y=True):
        """Compute and return the weighted mean squared error for all models."""

        if self.smoothen_alpha:
            self.compute_alpha()
            self.get_alpha_avg()
            self.alpha_smooth = self.alpha * self.alpha_avg**5
            self.wmse = [self.compute_weighted_MSE(fcn, self.alpha_smooth[:, iexp:iexp+1], update_y) for iexp, fcn in enumerate(self.fcn_list)]
        else:
            self.alpha = self.compute_alpha()
            self.wmse = [self.compute_weighted_MSE(fcn, self.alpha[:, iexp:iexp+1], update_y) for iexp, fcn in enumerate(self.fcn_list)]
        
        return self.wmse
