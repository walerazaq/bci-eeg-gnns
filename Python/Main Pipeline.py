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
from torch.utils.data import ConcatDataset, random_split
import ray
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler

# Check for CUDA and assign device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load Data
class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, group, seed=None):
        self.root_dir = root_dir
        self.group_dir = os.path.join(root_dir, group)
        self.seed = seed
        self.data = self.load_data()

    def load_data(self):
        data = []
        for filename in os.listdir(self.group_dir):
            if filename.endswith('.mat'):
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
                
        if self.seed is not None:
            random.seed(self.seed) 
            random.shuffle(data)
        return data

    def load_adjacency_matrix(self, filepath):
        mat_contents = read_mat(filepath)
        adjacency_matrix = mat_contents['BCM']
        adjacency_matrix = np.abs(adjacency_matrix)
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
        return node_features

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Load and store dataset
dataset = GraphDataset(r'/mnt/scratch2/users/asanni/', 'Alpha', seed=7)

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

# Training Models
#GCN_ = GCN(7, 4, config["f1"], config["f2"], readout='meanmax')
#GIN_ = GIN(7, 4, config["f1"], config["f2"], readout='meanmax')
#GAT_ = GAT(7, 4, config["f1"], config["f2"], config["num_heads"], readout='meanmax', concat=True)
#ChebC_ = ChebC(7, 4, config["f1"], config["f2"], config["chebFilterSize"], readout='meanmax')
#ChebEdge_ = ChebEdge(7, 4, config["f1"], config["f2"], config["chebFilterSize"], readout='meanmax')

# Training and RayTune Pipeline
def trainEEG(config, train_dataset, val_dataset, device):
    model = GCN(7, 4, config["f1"], config["f2"], readout='meanmax')
    model = model.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = config["lr"], weight_decay = config["wd"])
    
    if train.get_checkpoint():
        loaded_checkpoint = train.get_checkpoint()
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state = torch.load(
                os.path.join(loaded_checkpoint_dir, "checkpoint.pt")
            )
            model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)
            
    # Dataloaders    
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=1
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=1
    )
    
    epochs = 50

    for epoch in range(epochs):  # loop over the dataset multiple times
        epoch_train_loss = 0.0
        epoch_steps = 0
        
        for i, data in enumerate(tqdm(train_loader)):
            batch = data.batch.to(device)
            x = data.x.to(device)
            y = data.y.to(device)
            u = data.adj.to(device)
            optimizer.zero_grad()

            out = model(x,u,batch)

            step_loss = loss_function(out, y)
            step_loss.backward(retain_graph=True)
            optimizer.step()
            epoch_train_loss += step_loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(
                    "[%d, %5d] loss: %.3f"
                    % (epoch + 1, i + 1, epoch_train_loss / epoch_steps)
                )
                epoch_train_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        loss_func = nn.CrossEntropyLoss()
        
        labels = []
        preds = []
        for i, data in enumerate(val_loader):
            with torch.no_grad():
                batch = data.batch.to(device)
                x = data.x.to(device)
                label = data.y.to(device)
                u = data.adj.to(device)

                out = model(x,u,batch)
                
                step_loss = loss_func(out, label)
                val_loss += step_loss.detach().item()
                val_steps += 1
                preds.append(out.argmax(dim=1).detach().cpu().numpy())
                labels.append(label.cpu().numpy())
                
        preds = np.concatenate(preds).ravel()
        labels =  np.concatenate(labels).ravel()
        acc = balanced_accuracy_score(labels, preds)
        
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
            torch.save(
                (model.state_dict(), optimizer.state_dict()), path
            )
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            train.report(
                {"loss": (val_loss / val_steps), "accuracy": acc},
                checkpoint=checkpoint,
            )
        
    print("Finished Training")

# Testing Models
#GCN_ = GCN(7, 4, best_result.config["f1"], best_result.config["f2"], readout='meanmax')
#GIN_ = GIN(7, 4, best_result.config["f1"], best_result.config["f2"], readout='meanmax')
#GAT_ = GAT(7, 4, best_result.config["f1"], best_result.config["f2"], best_result.config["num_heads"], readout='meanmax', concat=True)
#ChebC_ = ChebC(7, 4, best_result.config["f1"], best_result.config["f2"], best_result.config["chebFilterSize"], readout='meanmax')
#ChebEdge_ = ChebEdge(7, 4, best_result.config["f1"], best_result.config["f2"], best_result.config["chebFilterSize"], readout='meanmax')

# Test Function
def test_model(best_result, test_loader, device):
         
    best_trained_model = GCN(7, 4, best_result.config["f1"], best_result.config["f2"], readout='meanmax')

    best_trained_model.to(device)

    checkpoint_path = os.path.join(best_result.checkpoint.to_directory(), "checkpoint.pt")

    model_state, optimizer_state = torch.load(checkpoint_path)
    best_trained_model.load_state_dict(model_state)
    
    best_trained_model.eval()
    labels = []
    preds = []
    for i, data in enumerate(test_loader):
            batch = data.batch.to(device)
            x = data.x.to(device)
            label = data.y.to(device)
            u = data.adj.to(device)

            out = best_trained_model(x,u,batch)
            preds.append(out.argmax(dim=1).detach().cpu().numpy())
            labels.append(label.cpu().numpy())
    preds = np.concatenate(preds).ravel()
    labels =  np.concatenate(labels).ravel()

    accuracy = balanced_accuracy_score(labels, preds)

    return accuracy

# RayTune
def mainTrain(num_samples=10, max_num_epochs=50, gpus_per_trial=1):
    config = {
        "f1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "f2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "lr": tune.loguniform(1e-5, 1e-1),
        "wd": tune.loguniform(1e-5, 1e-1),
        #"chebFilterSize": tune.choice([1, 2, 4, 8, 16]),
        #"num_heads": tune.choice([1, 2, 4, 8, 16]),
        "batch_size": tune.choice([2, 4, 8, 16]),
    }
    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=3,
        reduction_factor=2,
    )
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(trainEEG, train_dataset=train_dataset, val_dataset=val_dataset, device=device),
            resources={"cpu": 2, "gpu": gpus_per_trial}
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        param_space=config,
    )
    results = tuner.fit()

    best_result = results.get_best_result("loss", "min")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(
        best_result.metrics["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_result.metrics["accuracy"]))

    test_acc = test_model(best_result, test_loader, device)
    print("Best trial test set accuracy: {}".format(test_acc))
    
    return best_result

# Initialise Ray 
ray.init(num_cpus=4)

# Run Training Process
best_result = mainTrain(num_samples=20, max_num_epochs=50, gpus_per_trial=2)

# Shutdown Ray
ray.shutdown()

