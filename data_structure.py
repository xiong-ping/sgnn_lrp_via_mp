from typing import List
import networkx as nx
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

class Graph(object):
    def __init__(self, nbnodes: int, edges: List[List[int]], label: int, node_tags: List=None, node_features: List=None):
        self.nbnodes = nbnodes
        self.label = label
        self.edges = edges
        self.node_tags = node_tags
        self.node_features = node_features
        self.adj = None

    def get_adj(self):
        if self.adj == None: 
            adj = torch.eye(self.nbnodes)
            for i, j in self.edges:
                adj[i,j] = 1
                adj[j,i] = 1
            self.adj = adj
        return self.adj
