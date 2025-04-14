import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class SimpleGATModule(nn.Module):
    def __init__(self, in_channels=6, hidden_channels=16, out_channels=64, heads=1):
        super(SimpleGATModule, self).__init__()
        # First GAT layer: maps from in_channels to hidden_channels.
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, concat=False)
        # Second GAT layer: maps from hidden_channels to out_channels.
        self.gat2 = GATConv(hidden_channels, out_channels, heads=heads, concat=False)
        
    def forward(self, x, edge_index):
        # x: node feature matrix, edge_index: connectivity info.
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = self.gat2(x, edge_index)
        # Apply a final ReLU if you wish (or leave it to the decoder)
        #x = F.relu(x)
        return x
