import numpy as np
import pickle
import os
# import networkx as nx
import torch
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader

import dgl



def check_substring(s ,a):
    """
    Check if s has a substring of a
    """
    return any(x in s for x in a)

def build_dgl_graph(feature, adj, prob_matrix=None, label=None):
    """
    Build dgl.heterograph.DGLGraph from adjacency matrix and feature matrix
    adj: adjacency matrix [*c, N, N]
    prob_matrix: probability(edge weight) matrix [*c, N, N]
    feature: feature matrix [*c, N, D]
    label: label matrix [*c, N, D]
    """
    #check dims
    assert adj.shape == prob_matrix.shape
    assert adj.shape[:-1] == feature.shape[:-1], print(adj.shape, feature.shape)
    
    # get number of nodes and number of classes
    num_nodes = adj.shape[-1]
    num_classes = prob_matrix.shape[-1]
    
    # build heterograph
    # first transfer the adj matrix to edge list
    adj = torch.from_numpy(adj)
    edges = torch.where(adj > 0)
    g = dgl.graph(edges)
    # g = dgl.add_self_loop(g)
    
    # add features
    g.ndata['feat'] = torch.from_numpy(feature).float()
    # add labels
    if label is not None:
        g.ndata['label'] = torch.from_numpy(label).float()
    # add edge weights
    if prob_matrix is not None:
        prob_matrix = torch.from_numpy(prob_matrix).float()
        g.edata['prob'] = prob_matrix[edges]
    
    return g

def load_IC_data(dataset_name, used_ratio=1., source_path = '../graphdiffdata/'):
    # if dataset_name in ['karate', 'jazz', 'power_grid', 'netscience']: # synthetic datasets
    # find the path of the dataset using key word dataset_name
    root = source_path + '/synthesis/'
    data = None
    for file in os.listdir(root):
        if (dataset_name in file) and (('.pkl' in file)):
            dataset_path = root + file
            with open(dataset_path, 'rb') as f:
                data = pickle.load(f)
            break
        elif (dataset_name in file) and (('SG' in file)):
            dataset_path = root + file
            with open(dataset_path, 'rb') as f:
                data = pickle.load(f)
            data['inverse_pair'] = data['inverse_pairs'].view(-1,*data['inverse_pairs'].shape[-2:]).numpy()[:128]
            data['adj'] = data['adj'].toarray()
            break
    if data is None:
        raise ValueError('Dataset not found', dataset_name)
            
    
    data['prob_matrix'] = np.zeros_like(data['adj'])
    # ground_truth = data['inverse_pair'][..., 0:1]
    # feature = data['inverse_pair'][..., -1:]
    adjacancy = data['adj']
    sample_num = data['inverse_pair'].shape[0]
    node_num = adjacancy.shape[0]
    raw_data_dict = {
        'inverse_pair': data['inverse_pair'],
        'adj': data['adj'].astype(np.int64),
        'prob_matrix': data['prob_matrix'].astype(np.float32)
    }

    if dataset_name=='digg':
        adjacancy = source_path + 'digg/data/digg_sub_net_adj.npy'
        adjacancy = np.load(adjacancy)
        inverse_pair = source_path + 'digg/data/digg_inverse_pair.npy'
        inverse_pair = np.load(inverse_pair)
        # ground_truth = inverse_pair[..., 0:1]
        # feature = inverse_pair[..., -1:]
        sample_num = data['inverse_pair'].shape[0]
        node_num = adjacancy.shape[0]
        raw_data_dict = {
            'inverse_pair': inverse_pair.astype(np.float32),
            'adj': adjacancy.astype(np.int64),
            'prob_matrix': None
        }
    
    if raw_data_dict['inverse_pair'].ndim > 3:
        raw_data_dict['inverse_pair'] = raw_data_dict['inverse_pair'].reshape(-1, *raw_data_dict['inverse_pair'].shape[-2:])
    raw_data_dict['inverse_pair'] = raw_data_dict['inverse_pair'][:int(used_ratio*raw_data_dict['inverse_pair'].shape[0])]
    
    return raw_data_dict, sample_num, node_num

def load_data(dataset_name, dataset_path='../graphdiffdata', used_ratio=1., not_latent=False):
    dataIC_train = ICDataset(dataset_name, mode='train', dataset_path=dataset_path, used_ratio=used_ratio, not_latent=not_latent)
    dataIC_valid = ICDataset(dataset_name, mode='valid', dataset_path=dataset_path, used_ratio=used_ratio, not_latent=not_latent)
    dataIC_test = ICDataset(dataset_name, mode='test', dataset_path=dataset_path, used_ratio=used_ratio, not_latent=not_latent)
    # dataloader
    train_dataloader = GraphDataLoader(dataIC_train, batch_size=1, shuffle=True)
    valid_dataloader = GraphDataLoader(dataIC_valid, batch_size=1, shuffle=False)
    test_dataloader = GraphDataLoader(dataIC_test, batch_size=1, shuffle=False)
    eval_train_dataloader = GraphDataLoader(dataIC_train, batch_size=1, shuffle=False)
    g = dataIC_train[0]
    num_features = g.ndata['feat'].shape[-1]
    
    return train_dataloader, valid_dataloader, test_dataloader, eval_train_dataloader, num_features



class ICDataset(DGLDataset):
    """
    Load IC dataset
    """
    def __init__(self, dataset_name, mode, dataset_path='../graphdiffdata/', used_ratio=1. ,train_ratio=0.75, valid_ratio=0.125, not_latent=False):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.used_ratio = used_ratio
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        assert mode in ["train", "valid", "test"]
        self.mode = mode
        self.one_hot_label = False
        self.not_latent = not_latent
        # load pickle
        # with open(f'{self.dataset_path}/{self.dataset_name}.pkl', 'rb') as f:
        #     self.raw_data_dict = pickle.load(f)
        #     if self.raw_data_dict['inverse_pair'].ndim > 3:
        #         self.raw_data_dict['inverse_pair'] = self.raw_data_dict['inverse_pair'].reshape(-1, *self.raw_data_dict['inverse_pair'].shape[-2:])
        #     self.raw_data_dict['inverse_pair'] = self.raw_data_dict['inverse_pair'][:int(self.used_ratio*self.raw_data_dict['inverse_pair'].shape[0])]
        self.raw_data_dict, self.sample_num, self.node_num = load_IC_data(dataset_name, used_ratio=self.used_ratio, source_path=self.dataset_path)
        # print(self.raw_data_dict.keys())
        super().__init__(name=dataset_name, force_reload=False, verbose=False)
    def process(self):
        data_dict = self.raw_data_dict
        # get graph 
        self.graphs = []
        sample_num = data_dict['inverse_pair'].shape[0]
        label = data_dict['inverse_pair'][0][..., 0:1]
        feat = data_dict['inverse_pair'][0][..., -1:]

        if self.one_hot_label:
            # to one-hot
            label = np.eye(data_dict['inverse_pair'].shape[-1])[label.astype(int)].reshape(-1, data_dict['inverse_pair'].shape[-1])
            feat = np.eye(data_dict['inverse_pair'].shape[-1])[feat.astype(int)].reshape(-1, data_dict['inverse_pair'].shape[-1])
            # set 0 in label and feat to -1
            label[label==0.] = -1.
            feat[feat==0.] = -1.
            self.graph = build_dgl_graph(
                feat, 
                data_dict['adj'], 
                prob_matrix=data_dict['prob_matrix'], 
                label=label
            )
        else:
            if self.not_latent:
                self.graph = build_dgl_graph(
                    feat-0.5,
                    data_dict['adj'], 
                    prob_matrix=data_dict['prob_matrix'], 
                    label=label-0.5
                )
            else:
                self.graph = build_dgl_graph(
                    feat,
                    data_dict['adj'], 
                    prob_matrix=data_dict['prob_matrix'], 
                    label=label
                )
                
        lo, hi = 0, int(sample_num*self.train_ratio)
        if self.mode == "valid":
            lo, hi = int(sample_num*self.train_ratio), int(sample_num*(self.train_ratio+self.valid_ratio))
        elif self.mode == "test":
            lo, hi = int(sample_num*(self.train_ratio+self.valid_ratio)), sample_num
        for idx in range(lo, hi):
            label = data_dict['inverse_pair'][idx][..., 0:1]
            feat = data_dict['inverse_pair'][idx][..., -1:]

            # import pdb
            # pdb.set_trace()
            if self.one_hot_label:
                # to one-hot
                label = np.eye(data_dict['inverse_pair'].shape[-1])[label.astype(int)].reshape(-1, data_dict['inverse_pair'].shape[-1])
                feat = np.eye(data_dict['inverse_pair'].shape[-1])[feat.astype(int)].reshape(-1, data_dict['inverse_pair'].shape[-1])
                label[label==0.] = -1.
                feat[feat==0.] = -1.    
                graph = build_dgl_graph(
                    feat, 
                    data_dict['adj'], 
                    prob_matrix=data_dict['prob_matrix'], 
                    label=label
                )
            else:
                if self.not_latent:
                    graph = build_dgl_graph(
                        feat-0.5, 
                        data_dict['adj'], 
                        prob_matrix=data_dict['prob_matrix'], 
                        label=label-0.5
                    )
                else:
                    graph = build_dgl_graph(
                        feat, 
                        data_dict['adj'], 
                        prob_matrix=data_dict['prob_matrix'], 
                        label=label
                    )
                
            self.graphs.append(graph)
    # @property
    # def graph_list_path(self):
    #     return os.path.join(
    #         self.dataset_path, "{}_dgl_graph_list.bin".format(self.mode)
    #     )
        
    # @property
    # def g_path(self):
    #     return os.path.join(
    #         self.dataset_path, "{}_dgl_graph.bin".format(self.mode)
    #     )
        
    # @property
    # def info_path(self):
    #     return os.path.join(self.dataset_path, "{}_info.pkl".format(self.mode))
    
    # def save(self):
    #     save_graphs(self.graph_list_path, self.graphs)
    #     save_graphs(self.g_path, self.graph)
    #     save_info(
    #         self.info_path, {"labels": self._labels, "feats": self._feats}
    #     )
    @property
    def num_labels(self):
        if self.one_hot_label:
            return 2
        return 1
        
    def __len__(self):
        """Return number of samples in this dataset."""
        return len(self.graphs)
    
    def __getitem__(self, item):
        """Get the item^th sample.

        Parameters
        ---------
        item : int
            The sample index.

        Returns
        -------
        :class:`dgl.DGLGraph`
            graph structure, node features and node labels.

            - ``ndata['feat']``: node features
            - ``ndata['label']``: node labels
        """
        # if self._transform is None:
        #     return self.graphs[item]
        # else:
        #     return self._transform(self.graphs[item])
        return self.graphs[item]
        
