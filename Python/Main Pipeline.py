# Import Libraries
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

# Check for CUDA and assign device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load Data
class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, group, session=1):
        self.root_dir = root_dir
        self.group_dir = os.path.join(root_dir, group)
        self.session = session
        self.data = self.load_data()

    def load_data(self):
        data = []
        session_str = f"ses-0{self.session}"
        for filename in os.listdir(self.group_dir):
            if filename.endswith('.mat') and session_str in filename:
                adjacency_matrix_file = os.path.join(self.group_dir, filename)
                class_label = (int(filename.split('_')[-1].split('.')[0][-1])) - 1
                node_features = self.load_node_features(filename)
                adjacency_matrix = self.load_adjacency_matrix(adjacency_matrix_file)
                adj = SparseTensor.from_scipy(adjacency_matrix)
                num_nodes = adjacency_matrix.shape[0]
                x = np.vstack(list(node_features.values())).T
                x = torch.tensor(x, dtype=torch.float32)
                data.append(Data(x=x,
                                 adj = adj,
                                 y=torch.tensor([class_label], dtype=torch.long)))
        return data

    def load_adjacency_matrix(self, filepath):
        mat_contents = read_mat(filepath)
        adjacency_matrix = mat_contents['BCM']
        adj_sparse = csr_matrix(adjacency_matrix)

        return adj_sparse

    def load_node_features(self, filename):
        node_features = {}
        for folder in os.listdir(self.group_dir):
            folder_path = os.path.join(self.group_dir, folder)
            if os.path.isdir(folder_path):
                node_feature_file = os.path.join(folder_path, filename)
                if os.path.exists(node_feature_file):
                    mat_contents = read_mat(node_feature_file)
                    if folder == 'Activity':
                        node_features['Activity'] = mat_contents['ACTIVITY']
                    elif folder == 'Mobility':
                        node_features['Mobility'] = mat_contents['MOBILITY']
                    elif folder == 'Complexity':
                        node_features['Complexity'] = mat_contents['COMPLEXITY']
                    elif folder == 'Strength':
                        node_features['Strength'] = mat_contents['strengths']
                    elif folder == 'ClusteringCoef':
                        node_features['ClusteringCoef'] = np.abs(mat_contents['clusteringcoef'])
                    elif folder == 'Efficiency':
                        node_features['Efficiency'] = np.abs(mat_contents['efficiency'])
                    elif folder == 'PSD':
                        node_features['PSD'] = mat_contents['PSD']
                    elif folder == 'Betweenness':
                        node_features['Betweenness'] = mat_contents['betweenness']
        return node_features

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Load and store dataset
foldDataset = GraphDataset(r'/mnt/scratch2/users/asanni/EEG/', 'AlphaTrials', session=1)
testDataset = GraphDataset(r'/mnt/scratch2/users/asanni/EEG/', 'AlphaTrials', session=2)

# Test set data loader
test_loader = DataLoader(
    testDataset,
    batch_size=32,
    shuffle=True
    )

# Defining Graph Pooling readout function
def graph_readout(x, method, batch):
    if method == 'mean':
        return global_mean_pool(x,batch)

    elif method == 'meanmax':
        x_mean = global_mean_pool(x,batch)
        x_max = global_max_pool(x,batch)
        return torch.cat((x_mean, x_max), dim=1)

    elif method == 'sum':
        return global_add_pool(x,batch)

    else:
        raise ValueError('Undefined readout opertaion')

# Define GNN models architectures
class Abstract_GNN(torch.nn.Module):
    """
    An Abstract class for all GNN models
    Subclasses should implement the following:
    - forward()
    - predict()
    """
    def __init__(self, num_nodes, f3, f1, f2, readout):
        super(Abstract_GNN, self).__init__()
        self.readout = readout

    def _reset_parameters(self):
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
                else:
                    nn.init.uniform_(p)

    def forward(self, x, adj, batch):

        raise NotImplementedError


class GCN(Abstract_GNN):
    def __init__(self, num_nodes, f3, f1, f2, readout, **kwargs):
        super().__init__(num_nodes, f3, f1, f2, readout)
        self.readout = readout
        self.conv1 = GCNConv(num_nodes, f1)
        self.conv2 = GCNConv(f1, f2)
        self.conv3 = GCNConv(f2, f3)

        last_dim = 2 if readout=='meanmax' else 1
        self.mlp = nn.Linear(f3*last_dim, f3)
        self._reset_parameters()


    def forward(self, x, adj, batch):
        x = F.relu(self.conv1(x, adj))
        x = F.relu(self.conv2(x, adj))
        x = self.conv3(x, adj)
        x = graph_readout(x, self.readout, batch)
        x = self.mlp(x)
        return x



class GIN(Abstract_GNN):
    def __init__(self, num_nodes, f3, f1, f2, readout, **kwargs):
        super().__init__(num_nodes, f3, f1, f2, readout)
        self.conv1 = GINConv(
            Sequential(Linear(num_nodes, f1), BatchNorm1d(f1), ReLU(),
                       Linear(f1, f1), ReLU()))
        
        self.conv2 = GINConv(
            Sequential(Linear(f1, f2), BatchNorm1d(f2), ReLU(),
                       Linear(f2, f2), ReLU()))
        
        self.conv3 = GINConv(
            Sequential(Linear(f2, f3), BatchNorm1d(f3), ReLU(),
                       Linear(f3, f3), ReLU()))

        last_dim = 2 if readout=='meanmax' else 1

        self.last = Linear(f3*last_dim, f3)

        self._reset_parameters()

    def forward(self, x, adj , batch):
        x = F.relu(self.conv1(x, adj))
        x = F.relu(self.conv2(x, adj))
        x = self.conv3(x, adj)
        x = graph_readout(x, self.readout, batch)
        x = self.last(x)
        return x


class GAT(Abstract_GNN):
    def __init__(self, num_nodes, f3, f1, f2, num_heads, readout, concat, **kwargs):
        super().__init__(num_nodes, f3, f1, f2, readout)

        self.conv1 = GATv2Conv(num_nodes, f1, heads=num_heads, concat=concat)
        m = num_heads if concat else 1
        self.conv2 = GATv2Conv(f1*m, f2, heads=num_heads)
        self.conv3 = GATv2Conv(f2*m, f3, heads=1)
        
        last_dim = 2 if readout=='meanmax' else 1
        self.mlp = nn.Linear(f3*last_dim, f3)
        self._reset_parameters()


    def forward(self, x, adj , batch):
        x = F.relu(self.conv1(x, adj))
        x = F.relu(self.conv2(x, adj))
        x = self.conv3(x, adj)
        x = graph_readout(x, self.readout, batch)
        x = self.mlp(x)
        return x

class ChebC(Abstract_GNN):
    def __init__(self, num_nodes, f3, f1, f2, k, readout):
        super().__init__(num_nodes, f3, f1, f2, readout)
        self.conv1 = ChebConv(num_nodes, f1, k)
        self.conv2 = ChebConv(f1, f2, k)
        self.conv3 = ChebConv(f2, f3, k)
        self.readout = readout
        last_dim = 2 if readout == 'meanmax' else 1
        self.fc = nn.Linear(f3 * last_dim, f3)

        self._reset_parameters()

    def forward(self, x, adj, batch):
        row, col, edge_weight = adj.coo()
        edge_weight = edge_weight.type(torch.float32)
        edge_index = torch.stack([row, col], dim=0)
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = self.conv3(x, edge_index, edge_weight)
        x = graph_readout(x, self.readout, batch)
        x = self.fc(x)
        return x

class ChebEdge(Abstract_GNN):
    def __init__(self, num_nodes, f3, f1, f2, k, readout):
        super().__init__(num_nodes, f3, f1, f2, readout)
        self.conv1 = ChebConv(num_nodes, f1, k)
        self.conv2 = ChebConv(f2, f3, k)

        self.edgeconv1 = EdgeConv(nn.Sequential(nn.Linear(f1*2, f2*2), nn.ReLU(),
                                               nn.Linear(f2*2, f2)))

        last_dim = 2 if readout == 'meanmax' else 1
        self.fc = nn.Linear(f3 * last_dim, f3)

        self._reset_parameters()

    def forward(self, x, adj, batch):
        row, col, edge_weight = adj.coo()
        edge_weight = edge_weight.type(torch.float32)
        edge_index = torch.stack([row, col], dim=0)
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.relu(self.edgeconv1(x, adj))
        x = self.conv2(x, edge_index, edge_weight)
        x = graph_readout(x, self.readout, batch)
        x = self.fc(x)
        return x

# LOSO Training and Validation Functions
def train(model, dataset, device):

    model = model.to(device)

    loss_function = LabelSmoothingCrossEntropy()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-2)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-5)

    best_models = []
    best_metrics = []
    fold = 0

    for i in range(0, len(dataset), len(dataset)//10):
        fold += 1
        print(f"Fold {fold}/10")

        val_start = i
        val_end = i + len(dataset)//10
        train_dataset = dataset[:val_start] + dataset[val_end:]
        val_dataset = dataset[val_start:val_end]

        train_loader = DataLoader(
            train_dataset,
            batch_size=4,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=4,
            shuffle=True
        )

        best_metric = -1
        best_metric_epoch = -1
        best_val_loss = 1000
        best_model = None
        epochs = 1000

        print('-' * 30)
        print('Training ... ')
        early_stop = 30
        es_counter = 0

        for epoch in range(epochs):

            print("-" * 10)
            print(f"epoch {epoch + 1}/{epochs}")
            model.train()
            epoch_train_loss = 0

            for i, data in enumerate(tqdm(train_loader)):
                batch = data.batch.to(device)
                x = data.x.to(device)
                y = data.y.to(device)
                u = data.adj.to(device)
                optimizer.zero_grad()

                out = model(x, u, batch)

                step_loss = loss_function(out, y)
                step_loss.backward(retain_graph=True)
                optimizer.step()
                epoch_train_loss += step_loss.item()

            epoch_train_loss = epoch_train_loss / (i + 1)
            lr_scheduler.step()
            val_loss, val_acc = validate_model(model, val_loader, device)
            print(f"epoch {epoch + 1} train loss: {epoch_train_loss:.4f}")

            if val_loss < best_val_loss:
                best_metric = val_acc
                best_val_loss = val_loss
                best_metric_epoch = epoch + 1
                best_model = deepcopy(model)
                print("saved new best metric model")
                es_counter = 0
            else:
                es_counter += 1

            if es_counter > early_stop:
                print('No loss improvement.')
                break

            print(
                "current epoch: {} current val loss {:.4f} current accuracy: {:.4f}  best accuracy: {:.4f} at loss {:.4f} at epoch {}".format(
                    epoch + 1, val_loss, val_acc, best_metric, best_val_loss, best_metric_epoch))

        print(f"train completed, best_val_loss: {best_val_loss:.4f} at epoch: {best_metric_epoch}")

        best_models.append(best_model)
        best_metrics.append((best_metric, best_val_loss))

    return best_models, best_metrics

def validate_model(model, val_loader, device):
    model.eval()
    val_loss = 0
    loss_func = nn.CrossEntropyLoss()

    labels = []
    preds = []
    for i, data in enumerate(val_loader):
            batch = data.batch.to(device)
            x = data.x.to(device)
            label = data.y.to(device)
            u = data.adj.to(device)

            out = model(x,u,batch)

            step_loss = loss_func(out, label)
            val_loss += step_loss.detach().item()
            preds.append(out.argmax(dim=1).detach().cpu().numpy())
            labels.append(label.cpu().numpy())
    preds = np.concatenate(preds).ravel()
    labels =  np.concatenate(labels).ravel()
    acc = balanced_accuracy_score(labels, preds)
    loss = val_loss/(i+1)

    return loss, acc
    
# Test function
def test_model(model, test_loader, device):
    model.eval()
    labels = []
    preds = []
    for i, data in enumerate(test_loader):
            batch = data.batch.to(device)
            x = data.x.to(device)
            label = data.y.to(device)
            u = data.adj.to(device)

            out = model(x,u,batch)
            preds.append(out.argmax(dim=1).detach().cpu().numpy())
            labels.append(label.cpu().numpy())
    preds = np.concatenate(preds).ravel()
    labels =  np.concatenate(labels).ravel()

    accuracy = balanced_accuracy_score(labels, preds)

    return accuracy

# Initialise and train GCN Model
GCN_ = GCN(8, 2, 4, readout='sum')
models, metrics = train(GCN_, foldDataset, device)

