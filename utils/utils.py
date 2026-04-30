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
    # sckit-learn xxxx (cuda problem)
    device = x.device if x.is_cuda else torch.device('cpu')
    _, cluster_centers = kmeans(X=x, num_clusters=n_mem, distance='euclidean', device=device)
    print("time for conducting Kmeans Clustering :", time.time() - start)
    print('K means clustering is done!!!')

    return cluster_centers