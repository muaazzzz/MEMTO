import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from kmeans_pytorch import kmeans
import time

def to_var(x, volatile=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    return Variable(x, volatile=volatile)


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def k_means_clustering(x,n_mem,d_model):
    start = time.time()

    x = x.view([-1,d_model])
    print('running K Means Clustering. It takes few minutes to find clusters')
    
    # Check for NaN and inf values
    if torch.isnan(x).any():
        print(f'Warning: Input contains NaN values ({torch.isnan(x).sum().item()} values). Removing them.')
        valid_mask = ~torch.isnan(x).any(dim=1)
        x = x[valid_mask]
        if x.shape[0] < n_mem:
            print(f'Error: Not enough valid data points ({x.shape[0]}) for {n_mem} clusters. Using random initialization instead.')
            return F.normalize(torch.rand((n_mem, d_model), dtype=torch.float), dim=1).to(x.device)
    
    if torch.isinf(x).any():
        print(f'Warning: Input contains inf values ({torch.isinf(x).sum().item()} values). Clipping them.')
        x = torch.clamp(x, min=-1e6, max=1e6)
    
    # Normalize the input data to improve K-means stability
    x_mean = x.mean(dim=0, keepdim=True)
    x_std = x.std(dim=0, keepdim=True).clamp(min=1e-8)
    x_normalized = (x - x_mean) / x_std
    
    # sckit-learn xxxx (cuda problem)
    device = x.device if x.is_cuda else torch.device('cpu')
    try:
        _, cluster_centers = kmeans(X=x_normalized, num_clusters=n_mem, distance='euclidean', device=device)
        # Denormalize cluster centers back to original space
        cluster_centers = cluster_centers * x_std + x_mean
    except Exception as e:
        print(f'K-means failed with error: {e}. Using random initialization instead.')
        cluster_centers = F.normalize(torch.rand((n_mem, d_model), dtype=torch.float), dim=1).to(x.device)
    
    print("time for conducting Kmeans Clustering :", time.time() - start)
    print('K means clustering is done!!!')

    return cluster_centers