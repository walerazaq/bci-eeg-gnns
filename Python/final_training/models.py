class Abstract_GNN(torch.nn.Module):
    """
    An Abstract class for all GNN models
    Subclasses should implement the following:
    - forward()
    - predict()
    """
    def __init__(self, num_nodes, f1, f2, readout):
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
    def __init__(self, num_nodes, f1, f2, readout, **kwargs):
        super().__init__(num_nodes, f1, f2, readout)
        self.readout = readout
        self.conv1 = GCNConv(num_nodes, f1)
        self.conv2 = GCNConv(f1, f2)

        last_dim = 2 if readout=='meanmax' else 1
        self.mlp = nn.Linear(f2*last_dim, f2)
        self._reset_parameters()


    def forward(self, x, adj, batch):
        x = F.relu(self.conv1(x, adj))
        x = self.conv2(x, adj)
        x = graph_readout(x, self.readout, batch)
        x = self.mlp(x)
        return x



class GIN(Abstract_GNN):
    def __init__(self, num_nodes, f1, f2, readout, **kwargs):
        super().__init__(num_nodes, f1, f2, readout)
        self.conv1 = GINConv(
            Sequential(Linear(num_nodes, f1), BatchNorm1d(f1), ReLU(),
                       Linear(f1, f1), ReLU()))

        self.conv2 = GINConv(
            Sequential(Linear(f1, f2), BatchNorm1d(f2), ReLU(),
                       Linear(f2, f2), ReLU()))

        last_dim = 2 if readout=='meanmax' else 1

        self.last = Linear(f2*last_dim, f2)

        self._reset_parameters()

    def forward(self, x, adj , batch):
        x = self.conv1(x, adj)
        x = self.conv2(x, adj)
        x = graph_readout(x, self.readout, batch)
        x = self.last(x)
        return x


class GAT(Abstract_GNN):
    def __init__(self, num_nodes, f1, f2, readout, concat, num_heads, **kwargs):
        super().__init__(num_nodes, f1, f2, readout)

        self.conv1 = GATv2Conv(num_nodes, f1, heads=num_heads, concat=concat)
        m = num_heads if concat else 1
        self.conv2 = GATv2Conv(f1*m, f2, heads=1)
        last_dim = 2 if readout=='meanmax' else 1
        self.mlp = nn.Linear(f2*last_dim, f2)
        self._reset_parameters()


    def forward(self, x, adj , batch):
        x = self.conv1(x, adj)
        x = F.relu(x)
        x = self.conv2(x, adj)
        x = graph_readout(x, self.readout, batch)
        x = self.mlp(x)
        return x

class ChebC(Abstract_GNN):
    def __init__(self, num_nodes, f1, f2, k, readout):
        super().__init__(num_nodes, f1, f2, readout)
        self.conv1 = ChebConv(num_nodes, f1, k)
        self.conv2 = ChebConv(f1, f1, k)
        self.readout = readout
        last_dim = 2 if readout == 'meanmax' else 1
        self.fc = nn.Linear(f1 * last_dim, f2)

        self._reset_parameters()

    def forward(self, x, adj, batch):
        row, col, edge_weight = adj.coo()
        edge_weight = edge_weight.type(torch.float32)
        edge_index = torch.stack([row, col], dim=0)
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = graph_readout(x, self.readout, batch)
        x = self.fc(x)
        return x

class ChebEdge(Abstract_GNN):
    def __init__(self, num_nodes, f1, f2, k, readout):
        super().__init__(num_nodes, f1, f2, readout)
        
        self.conv1 = ChebConv(num_nodes, f1, k)
        self.edgeconv1 = EdgeConv(nn.Sequential(nn.Linear(f1*2, f1*2), nn.ReLU(), nn.Linear(f1*2, f2)))

        last_dim = 2 if readout == 'meanmax' else 1
        self.fc = nn.Linear(f2 * last_dim, f2)

        self._reset_parameters()

    def forward(self, x, adj, batch):
        row, col, edge_weight = adj.coo()
        edge_weight = edge_weight.type(torch.float32)
        edge_index = torch.stack([row, col], dim=0)
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.edgeconv1(x, adj)
        x = graph_readout(x, self.readout, batch)
        x = self.fc(x)
        return x
