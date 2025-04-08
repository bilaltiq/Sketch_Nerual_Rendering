import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, EdgeConv, HeteroConv 

import gnn.basic

# Include a spreadsheet of different rows for experiments and accuracy tracking; see what of these properties improves accuracy
# Take new best experiment and build up on that

class SemanticGANModule(nn.Module):
    def __init__(self, in_channels=6, hidden_channels=16, out_channels=64, heads=8): # Start with a smaller number of heads and gradually increase
        super(SemanticGANModule, self).__init__()
        self.local_head = HeteroConv({
            ('stroke', 'connected_to', 'stroke'): GATConv(in_channels, 16, heads=heads, concat=False),
            # ('stroke', 'ordered_next', 'stroke'): GATConv(in_channels, 16, heads=heads, concat=False)
        }, aggr='mean')

        self.layers = nn.ModuleList([
            HeteroConv({
                ('stroke', 'connected_to', 'stroke'): GATConv(16, 64, heads=heads, concat=False),
                # ('stroke', 'ordered_next', 'stroke'): GATConv(16, 64, heads=heads, concat=False)
            }, aggr='mean'),

            # HeteroConv({
            #     # ('stroke', 'represented_by', 'stroke'): GATConv(hidden_channels * heads, out_channels, heads=heads),
            #     # ('stroke', 'neighboring_vertical', 'stroke'): GATConv(hidden_channels * heads, out_channels, heads=heads),
            #     # ('stroke', 'neighboring_horizontal', 'stroke'): GATConv(hidden_channels * heads, out_channels, heads=heads),
            #     # ('stroke', 'contains', 'stroke'): GATConv(hidden_channels * heads, out_channels, heads=heads),
            #     # ('stroke', 'order_add', 'stroke'): GATConv(hidden_channels * heads, out_channels, heads=heads),
            #     # ('stroke', 'perpendicular', 'stroke'): GATConv(hidden_channels * heads, out_channels, heads=heads)
            # }, aggr='mean')

        ])

    def forward(self, x_dict, edge_index_dict):
        # print("Input x_dict shapes:")
        # for key, value in x_dict.items():
        #     print(f"{key}: {value.shape}")


        x_dict = self.local_head(x_dict, edge_index_dict)


        # print("After local_head x_dict shapes:")
        # for key, value in x_dict.items():
        #     print(f"{key}: {value.shape}")

        

        for layer in self.layers:
            x_dict = layer(x_dict, edge_index_dict)
            # print("After layer x_dict shapes:")
            # for key, value in x_dict.items():
            #     print(f"{key}: {value.shape}")

        x_dict = {key: x.relu() for key, x in x_dict.items()}

        return x_dict


class Stroke_Decoder_GAN(nn.Module):
    def __init__(self, hidden_channels=128):
        super(Stroke_Decoder_GAN, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(64, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_channels, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
        )

    def forward(self, x_dict):
        # print("Input to decoder x_dict['stroke'] shape:", x_dict['stroke'].shape)
        return torch.sigmoid(self.decoder(x_dict['stroke']))