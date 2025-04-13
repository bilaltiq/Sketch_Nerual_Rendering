import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, EdgeConv, GATConv
from torch_geometric.data import HeteroData

import gnn.basic

class SemanticModule(nn.Module):
    def __init__(self, in_channels=6):
        super(SemanticModule, self).__init__()
        self.local_head = gnn.basic.GeneralHeteroConv(['connected_to_sum', 'ordered_next_sum'], in_channels, 16, conv_class=GATConv, conv_kwargs={'heads': 3, 'concat': False})

        self.layers = nn.ModuleList([
            gnn.basic.ResidualGeneralHeteroConvBlock(['connected_to_sum', 'ordered_next_sum'], 16, 32),
            gnn.basic.ResidualGeneralHeteroConvBlock(['represents_sum', 'represented_by_sum', 'neighboring_vertical_mean', 'neighboring_horizontal_mean', 'contains_sum', 'order_add', 'perpendicular_mean'], 32, 64),

        ])


    def forward(self, x_dict, edge_index_dict):

        x_dict = self.local_head(x_dict, edge_index_dict)

        for layer in self.layers:
            x_dict = layer(x_dict, edge_index_dict)
        
        x_dict = {key: x.relu() for key, x in x_dict.items()}

        return x_dict



class Stroke_Decoder(nn.Module):
    def __init__(self, hidden_channels=128):
        super(Stroke_Decoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(64, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_channels, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),

        )

    def forward(self, x_dict):
        return self.decoder(x_dict['stroke'])
