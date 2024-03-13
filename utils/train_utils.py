import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
import dgl
import networkx as nx
import numpy as np
import torch_geometric

def save_model(model, path):
    torch.save(model.state_dict(), path)
    
def draw_data_distribution(data, save_path):
    plt.figure(figsize=(10, 5))
    plt.hist(data, bins=100)
    # draw horizontal line of mean and std
    plt.axvline(data.mean(), color='r', linestyle='dashed', linewidth=1)
    plt.axvline(data.mean() + data.std(), color='g', linestyle='dashed', linewidth=1)
    plt.axvline(data.mean() - data.std(), color='g', linestyle='dashed', linewidth=1)
    plt.savefig(save_path)
    plt.close()
    
def calculate_mean_distance(node_set, node_set_gt, graph):
    # node_set_a: list of node indices
    # node_set_b: list of node indices
    # graph: networkx graph
    # return: mean distance between node_set_a and node_set_b
    if type(graph)==dgl.DGLGraph:
        graph = graph.to_networkx()
    
    min_dists=[]
    
    for node_gt in node_set_gt:
        min_dist = 1000000
        for node in node_set:
            dist = nx.shortest_path_length(graph, node, node_gt)
            if dist < min_dist:
                min_dist = dist
                min_node = node
                min_node_gt = node_gt
        if min_dist == 1000000:
            continue
            # print("Error: no path between node {} and node {}".format(min_node, min_node_gt))
        min_dists.append(min_dist)
    return np.mean(min_dists).item()