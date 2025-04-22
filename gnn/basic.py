import torch.nn as nn
from torch_geometric.nn import EdgeConv, HeteroConv, GCNConv, GATConv, GATv2Conv
from torch_geometric.nn.models import MLP
import numpy as np
from torch_geometric.nn import aggr
from torch_geometric.utils import scatter
import torch
from torch_geometric.nn import SAGEConv, GAE


from torch_geometric.nn.conv import MessagePassing

def act_layer(act_type, inplace=False, neg_slope=0.2, n_prelu=1):
    """
    """
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act_type)
    return layer


def norm_layer(norm_type, nc):
    """
    """
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm1d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm1d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return layer

class MLPLinear(nn.Sequential):
    def __init__(self, channels, act_type='relu', norm_type='batch', bias=True):
        m = []
        for i in range(1, len(channels)):
            m.append(nn.Linear(channels[i - 1], channels[i], bias))
            if norm_type and norm_type != 'None':
                m.append(norm_layer(norm_type, channels[i]))
            if act_type:
                m.append(act_layer(act_type))
        super(MLPLinear, self).__init__(*m)
        
class MultiSeq(nn.Sequential):
    def __init__(self, *args):
        super(MultiSeq, self).__init__(*args)

    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs
    


class GeneralHeteroConv(torch.nn.Module):
    def __init__(self, gcn_types, in_channels, out_channels, instance_net_type = None, conv_class=EdgeConv, conv_kwargs=None):
        super(GeneralHeteroConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.gcn_types = gcn_types
        self.instance_net_type = instance_net_type
        self.conv_class = conv_class
        self.conv_kwargs = conv_kwargs if conv_kwargs is not None else {}
        self.gconv = HeteroConv(self.create_HeteroConv_dict(), aggr='mean')
            
    def find_aggr_fun(self):
        aggr_fns = []
        for gcn_type in self.gcn_types:
            gcn_type_split = gcn_type.split('_')
            aggr_fns.append(gcn_type_split[-1])
        return aggr_fns
    
    def create_HeteroConv_dict(self):
        heteroConv_dict = {}
        edges_types = [('stroke', 'ordered_next', 'stroke'),
                       ('stroke', 'connected_to', 'stroke')
                       ]
        
        aggr_fns = self.find_aggr_fun()
        for i in range(len(edges_types)):
            if self.instance_net_type =='HeteroConv' and i == len(edges_types) - 1 :
                heteroConv_dict[edges_types[i]] = EdgeConv(
                                                    nn=MLPLinear(
                                                        channels=[self.in_channels*2, self.out_channels],
                                                        act_type='relu', 
                                                        norm_type=None
                                                    ),
                                                    aggr=aggr_fns[i]
                                                )
            
            elif self.conv_class == GATConv:
                conv_layer = GATConv(
                    self.in_channels,
                    self.out_channels,
                    **self.conv_kwargs
                )
                # print('debug')
                heteroConv_dict[edges_types[i]] = conv_layer

            else:
                heteroConv_dict[edges_types[i]] = EdgeConv(
                                                    nn=MLPLinear(
                                                        channels=[self.in_channels*2, self.out_channels],
                                                        act_type='relu', 
                                                        norm_type=None
                                                    ),
                                                    aggr=aggr_fns[i]
                                                )
            
        return heteroConv_dict
            
    
    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None, data=None):
        if edge_attr_dict is None:
            edge_attr_dict = {}

        if 'stroke' in x_dict and x_dict['stroke'].size(0) == 0:
            x_dict_no_brep = {key: value for key, value in x_dict.items() if key != 'stroke'}
            edge_index_dict_no_brep = {key: value for key, value in edge_index_dict.items() if 'stroke' not in key}

            # Perform convolution without brep nodes and edges
            res = self.gconv(x_dict_no_brep, edge_index_dict_no_brep, edge_attr_dict)
        else:
            res = self.gconv(x_dict, edge_index_dict, edge_attr_dict)
                
        return res
    

class ResidualGeneralHeteroConvBlock(nn.Module):
    def __init__(self, gcn_types, in_channels, out_channels, is_instance_net=False, conv_class=EdgeConv, conv_kwargs=None):
        super(ResidualGeneralHeteroConvBlock, self).__init__()

        self.conv_kwargs = conv_kwargs or {}

        self.mlp_edge_conv = GeneralHeteroConv(gcn_types, in_channels, out_channels, is_instance_net, conv_class=conv_class, conv_kwargs=self.conv_kwargs)
        self.conv_class = conv_class

        self.residual = (in_channels == out_channels)
        if not self.residual:
            self.projection = nn.Linear(in_channels, out_channels)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None, data=None):

        # Store residual stroke features
        residual_stroke = x_dict['stroke']

        # Apply graph convolution
        out = self.mlp_edge_conv(x_dict, edge_index_dict, edge_attr_dict, data)

        # Apply residual connection
        if self.residual:
            out['stroke'] += residual_stroke
        else:
            out['stroke'] += self.projection(residual_stroke)

        return out


class LinkPredictionNet(nn.Module):
    '''
    This class is for link prediction
    net_type:
        - MLP: has as input the output of the last semantic EdgeConv layer
        - Conv: has as input the output of semantic softmax layer
    '''
    
    def __init__(self, channels, net_loss, n_blocks = None, mlp_segment = None, 
                 net_type = 'MLP'):
        super().__init__()
        self.n_blocks = n_blocks
        self.mlp_segment = mlp_segment
        self.channels = channels
        self.net_type = net_type
        self.net_loss = net_loss
        if self.net_type == 'MLP' or 'HeteroConv' in self.net_type:
            if self.net_type == 'HeteroConv2Mult':
                print('first MPL in_channels: ', self.channels*(1 + self.n_blocks))
                self.net = torch.nn.ModuleList([
                MLPLinear([self.channels*(1 + self.n_blocks), self.mlp_segment[0]], norm_type='batch', act_type='relu'), 
                MLPLinear(self.mlp_segment, norm_type='batch', act_type='relu'), 
                MLPLinear([self.mlp_segment[1], 1], norm_type='batch', act_type=None) 
                    
                ]) 
            else:
                print('first MPL in_channels: ', self.channels*(1 + self.n_blocks)*2)
                self.net = torch.nn.ModuleList([
                MLPLinear([self.channels*(1 + self.n_blocks)*2, self.mlp_segment[0]], norm_type='batch', act_type='relu'), 
                MLPLinear(self.mlp_segment, norm_type='batch', act_type='relu'), 
                MLPLinear([self.mlp_segment[1], 1], norm_type='batch', act_type=None) 
                    
                ])
            
        elif self.net_type == 'Conv':
            self.net1 = SAGEConv(self.channels, self.channels)
            self.net2 = SAGEConv(self.channels, self.channels)
        if self.net_loss == 'BCE':
            self.sigmoid = torch.nn.Sigmoid()
        elif self.net_loss == 'BCEWithLogits':
            self.sigmoid = None
        # self.net = torch.nn.Sequential(
        #     torch.nn.Linear(2 * self.out_segment, self.out_segment),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(self.out_segment, 1),
        #     # torch.nn.ReLU(),
        #     # torch.nn.Sigmoid()
        #     )

    def forward(self, z_dict, edge_label_index):
        # print('z_dict.shape: ', z_dict.shape)
        # print('edge_label_index.shape: ', edge_label_index.shape)
        row, col = edge_label_index
        if self.net_type == 'MLP' or 'HeteroConv' in self.net_type:
            # print('z_dict[row].shape: ', z_dict[row].shape)
            # print('z_dict[col].shape: ', z_dict[col].shape)
            if self.net_type == 'HeteroConv2Mult':
                z = torch.mul(z_dict[row], z_dict[col])
            else:
                z = torch.cat([z_dict[row], z_dict[col]], dim=-1)
            # print('z.shape: ', z.shape)
            for i, mlp_block in enumerate(self.net):
                z = mlp_block(z)
                # print(i, ' block - z.shape: ', z.shape)
            z = z.view(-1)
        elif self.net_type == 'Conv':
            z_dict = self.net1(z_dict, edge_label_index).relu()
            z_dict = self.net2(z_dict, edge_label_index).relu()
            z = (z_dict[row] * z_dict[col]).sum(dim=-1)
        if self.sigmoid is not None:
            z = self.sigmoid(z)
        return z

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, 2 * out_channels)
        self.conv3 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return self.conv3(x, edge_index)