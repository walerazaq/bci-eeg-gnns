import os
import torch
import numpy as np
import pandas as pd
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

from prepare_dataset import *
from utils import *
from models import *
from training import *

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

dataset = GraphDataset(r'/mnt/scratch2/users/asanni/EEG/', 'AlphaTrials')

# Initialise and train GCN Model

GCN_ = GCN(8, 64, 4, readout='meanmax')
gcnModels, gcnMetrics = train(GCN_, dataset, 2, 0.0017644502604784013, 0.026698409901099084, device)
Metrics(gcnMetrics)

# Initialise and train GIN Model

GIN_ = GIN(8, 4, 4, readout='meanmax')
ginModels, ginMetrics = train(GIN_, dataset, 8, 0.0030462861290017364, 0.005164638202300129, device)
Metrics(ginMetrics)

# Initialise and train GAT Model

GAT_ = GAT(8, 64, 4, readout='meanmax', concat=True, num_heads=2)
gatModels, gatMetrics = train(GAT_, dataset, 16, 0.001488237863877619, 0.00023980048585295436, device)
Metrics(gatMetrics)

# Initialise and train Chebnet Model

chebFilterSize = 1

ChebC_ = ChebC(8, 32, 4, chebFilterSize, readout='meanmax')
chebModels, chebMetrics = train(ChebC_, dataset, 2, 0.0032493935447595367, 0.0008162341059375671, device)
Metrics(chebMetrics)

# Initialise and train Chebnet + EdgeConv Model

chebFilterSize = 2

ChebEdge_ = ChebEdge(8, 4, 4, chebFilterSize, readout='meanmax')
chbedgModels, chbedgMetrics = train(ChebEdge_, dataset, 16, 0.011754486577724524, 0.051220909371486045, device)
Metrics(chbedgMetrics)

