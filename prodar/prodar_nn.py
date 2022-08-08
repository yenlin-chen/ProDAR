#!/usr/bin/env python
# coding: utf-8

import torch
from torch import nn
import torch
from torch import nn
from torch.nn import functional as F

from torch_geometric.nn import (
    GCNConv,
    GATConv,
    SAGEConv,
    global_max_pool,
    LayerNorm
)

gnn_extractor = {
    'gcn': GCNConv,
    'gat': GATConv,
    'sage': SAGEConv,
    'graphsage': SAGEConv
}

class ProDAR_NN(nn.Module):
    def __init__(self,
                 dim_node_feat, dim_pers_feat, dim_out,
                 dim_node_hidden,
                 dim_pers_embedding, dim_graph_embedding,
                 dropout_rate,
                 gnn_type, n_graph_layers, **gnn_kwargs):

        if not 'aggr' in gnn_kwargs.keys():
            gnn_kwargs['aggr'] = 'mean'

        if gnn_type == 'GCN':
            gnn_kwargs['improved'] = False
            gnn_kwargs['add_self_loops'] = True
        elif gnn_type == 'GAT':
            if not 'heads' in gnn_kwargs.keys():
                gnn_kwargs['heads'] = 1

        self.n_graph_layers = n_graph_layers

        super().__init__()

        self.conv_layers = nn.ModuleList([])
        for i in range(n_graph_layers):
            dim_input = dim_node_feat if i == 0 else dim_node_hidden

            conv_extractor = gnn_extractor[gnn_type.lower()]
            conv = conv_extractor(dim_input, dim_node_hidden, **gnn_kwargs)

            self.conv_layers.append(conv)

        self.max_aggr_block = nn.Sequential(
            nn.Linear(dim_node_hidden, dim_node_hidden),
            nn.Dropout(p=dropout_rate),
            nn.LayerNorm(dim_node_hidden)
        ) if gnn_kwargs['aggr']=='max' else None

        self.graph_block = nn.Sequential(
            nn.Linear(n_graph_layers * dim_node_hidden, dim_graph_embedding),
            nn.LayerNorm(dim_graph_embedding),
            nn.Dropout(p=dropout_rate),
            nn.ReLU()
        )

        self.pi_block = nn.Sequential(
            nn.Linear(dim_pers_feat, dim_pers_embedding),
            nn.LayerNorm(dim_pers_embedding),
            nn.Dropout(p=dropout_rate),
            nn.ReLU()
        ) if dim_pers_embedding else None

        dim_total_embedding = dim_graph_embedding + dim_pers_embedding
        self.fc_block = nn.Sequential(
            nn.Linear(dim_total_embedding, dim_out),
            nn.LayerNorm(dim_out),
            nn.Dropout(p=dropout_rate)
        )

    def forward(self, data):

        x = data.x

        jk_connection = torch.empty((self.n_graph_layers, *x.shape))
        for idx, conv in enumerate(self.conv_layers):
            x = conv(x, data.edge_index)
            x = F.relu(x)
            if self.max_aggr_block:
                x = self.max_aggr_block(x)
            jk_connection[idx] = x

        x = torch.cat(jk_connection, dim=1)
        x = global_max_pool(x, data.batch)

        if self.pi_block: # graph & persistence embedding (concatenated)
            x = torch.cat((self.graph_block(x), self.pi_block(data.pi)), dim=1)
        else: # graph embedding only
            x = self.graph_block(x)

        x = self.fc_block(x)

        return x

if __name__ == '__main__':

    model = ProDAR_NN(
        dim_node_feat=16, dim_pers_feat=25, dim_out=100,
        dim_node_hidden=32,
        dim_pers_embedding=8, dim_graph_embedding=4,
        dropout_rate=0.3,
        gnn_type='GCN', n_graph_layers=3,
        aggr='add'
    )

    print(model)
