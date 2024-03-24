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
    