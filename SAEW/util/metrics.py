import torch


def batch_wasserstein_distance(u_values, v_values):
    u_sorter = torch.argsort(u_values, dim = 1)
    v_sorter = torch.argsort(v_values, dim = 1)    

    all_values = torch.cat((u_values, v_values), axis = 1)
    all_values = torch.sort(all_values, dim=1)
    
    deltas = torch.diff(all_values[0])
    u_cdf_indices = []
    v_cdf_indices = []

    if all_values[0].shape[0] > 1:
        for i in range(all_values[0].shape[0]):
            u_cdf_indices.append(torch.searchsorted(u_values[i][u_sorter[i]], all_values[0][i][:-1], side='right' ))
            v_cdf_indices.append(torch.searchsorted(v_values[i][v_sorter[i]], all_values[0][i][:-1],  side='right'))
        u_cdf_indices = torch.stack(u_cdf_indices, dim = 0)
        v_cdf_indices = torch.stack(v_cdf_indices, dim = 0)
    else:
        u_cdf_indices = torch.searchsorted(u_values[0][u_sorter[0]], all_values[0][0][:-1], side='right').unsqueeze(0)
        v_cdf_indices = torch.searchsorted(v_values[0][v_sorter[0]], all_values[0][0][:-1], side='right').unsqueeze(0)
        
    u_cdf = u_cdf_indices / u_values.shape[1]
    v_cdf = v_cdf_indices / v_values.shape[1]
    
    return torch.sum(torch.multiply(torch.abs(u_cdf - v_cdf), deltas), dim=1)
    


import math
import torch
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def median_heuristic(X, beta=0.5):
    max_n = min(30000, X.shape[0])
    D2 = euclidean_distances(X[:max_n], squared=True)
    med_sqdist = np.median(D2[np.triu_indices_from(D2, k=1)])
    beta_list = [beta**2, beta**1, 1, (1.0/beta)**1, (1.0/beta)**2]
    return [med_sqdist * b for b in beta_list]

def batch_mmd2_loss(X_p_enc, X_f_enc, sigma_var):

    n_basis = 1024
    gumbel_lmd = 1e+6
    cnst = math.sqrt(1. / n_basis)
    n_mixtures = sigma_var.size(0)
    n_samples = n_basis * n_mixtures
    batch_size, seq_len, nz = X_p_enc.size()


    def sample_gmm(W, batch_size):

        U = torch.FloatTensor(batch_size*n_samples, n_mixtures).uniform_()
        sigma_samples = F.softmax(U * gumbel_lmd).matmul(sigma_var)
        W_gmm = W.mul(1. / sigma_samples.unsqueeze(1))
        W_gmm = W_gmm.view(batch_size, n_samples, nz)
        return W_gmm


    W = Variable(torch.FloatTensor(batch_size*n_samples, nz).normal_(0, 1))
    W_gmm = sample_gmm(W, batch_size)                                  
    W_gmm = torch.transpose(W_gmm, 1, 2).contiguous()                  
    XW_p = torch.bmm(X_p_enc, W_gmm)                                    
    XW_f = torch.bmm(X_f_enc, W_gmm)                                    
    z_XW_p = cnst * torch.cat((torch.cos(XW_p), torch.sin(XW_p)), 2)
    z_XW_f = cnst * torch.cat((torch.cos(XW_f), torch.sin(XW_f)), 2)
    batch_mmd2_rff = torch.sum((z_XW_p.mean(1) - z_XW_f.mean(1))**2, 1)
    return batch_mmd2_rff






