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
