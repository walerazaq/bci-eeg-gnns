#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import torch
import numpy as np
import pandas as pd
import tempfile
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.utils.augmentation import mask_feature, shuffle_node
from torch_sparse import SparseTensor
from scipy.sparse import csr_matrix
from pymatreader import read_mat
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score
from copy import deepcopy
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.nn import SimpleConv, GCNConv, GATv2Conv, GINConv, ChebConv, EdgeConv, global_mean_pool, global_max_pool, global_add_pool
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU
from torch_geometric.nn.inits import glorot, zeros
from collections import OrderedDict
from torch_geometric.utils import to_networkx
from torch_geometric.loader import DataLoader
from torch.utils.data import ConcatDataset, random_split
import ray
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler

from prepare_dataset import *
from utils import *
from models import *
from tuning import *

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

dataset = GraphDataset(r'/mnt/scratch2/users/asanni/EEG/', 'AlphaTrials', seed=4)

# Next we split the dataset into training, test, and validation sets.
total_size = len(dataset)

# Calculate the sizes of each split as percentages
train_size = int(0.6 * total_size)
val_size = int(0.2 * total_size)
test_size = total_size - train_size - val_size

# Random split the dataset based on percentages
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Define testset data loader
test_loader = DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=True
)

'''
The relevant model needs to be copied into the tuning pipeline
'''

# Training Models
#GCN_ = GCN(8, config["f1"], 4, readout='meanmax')
#GIN_ = GIN(8, config["f1"], 4, readout='meanmax')
#GAT_ = GAT(8, config["f1"], 4, config["num_heads"], readout='meanmax', concat=True)
#ChebC_ = ChebC(8, config["f1"], 4, config["chebFilterSize"], readout='meanmax')
#ChebEdge_ = ChebEdge(8, config["f1"], 4, config["chebFilterSize"], readout='meanmax')

# Testing Models
#GCN_ = GCN(8, best_result.config["f1"], 4, readout='meanmax')
#GIN_ = GIN(8, best_result.config["f1"], 4, readout='meanmax')
#GAT_ = GAT(8, best_result.config["f1"], 4, best_result.config["num_heads"], readout='meanmax', concat=True)
#ChebC_ = ChebC(8, best_result.config["f1"], 4, best_result.config["chebFilterSize"], readout='meanmax')
#ChebEdge_ = ChebEdge(8, best_result.config["f1"], 4, best_result.config["chebFilterSize"], readout='meanmax')

# Initialise Ray 
ray.init(num_cpus=4)

# Run Training Process
best_result = mainTrain(num_samples=15, max_num_epochs=300, gpus_per_trial=1)

# Shutdown Ray
ray.shutdown()

