#!/usr/bin/env python
# coding: utf-8

import json
import sys
from os import path

import torch
from torch import nn
from torch.nn import functional as F
from torchinfo import summary

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

        self.all_args = locals()
        del self.all_args['self'], self.all_args['__class__']

        if not 'aggr' in gnn_kwargs.keys():
            gnn_kwargs['aggr'] = 'mean'
        elif gnn_type == 'GCN' and gnn_kwargs['aggr'] != 'add':
            raise ValueError('GCN layers must have ADD for aggregation')

        if gnn_type == 'GCN':
            gnn_kwargs['improved'] = False
            gnn_kwargs['add_self_loops'] = True
        elif gnn_type == 'GAT':
            if not 'heads' in gnn_kwargs.keys():
                gnn_kwargs['heads'] = 1

        self.dim_node_hidden = dim_node_hidden

        super().__init__()

        conv_extractor = gnn_extractor[gnn_type.lower()]
        self.conv_layers = nn.ModuleList([])
        for i in range(n_graph_layers):
            dim_input = dim_node_feat if i == 0 else dim_node_hidden

            conv = conv_extractor(dim_input, dim_node_hidden, **gnn_kwargs)

            self.conv_layers.append(conv)

        self.max_aggr_block = nn.Sequential(
            nn.Linear(dim_node_hidden, dim_node_hidden),
            nn.LayerNorm(dim_node_hidden),
            nn.Dropout(p=dropout_rate)
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

        # jk_shape = (x.shape[0], len(self.conv_layers) * self.dim_node_hidden)
        jk_connection = [] #torch.empty(jk_shape).to(self.device)

        for idx, conv in enumerate(self.conv_layers):
            x = conv(x, data.edge_index)
            x = F.relu(x)
            if self.max_aggr_block:
                x = self.max_aggr_block(x)
            jk_connection.append(x)
            # jk_connection[:,idx*self.dim_node_hidden:(idx+1)*self.dim_node_hidden] = x

        # print(jk_connection.device)
        # print(data.batch.device)
        jk_connection = torch.cat(jk_connection, dim=1)
        x = global_max_pool(jk_connection, data.batch)

        if self.pi_block: # graph & persistence embedding (concatenated)
            # x = torch.cat((self.graph_block(x), self.pi_block(data.pi)), dim=1)
            x = torch.cat((self.graph_block(x), self.pi_block(data.pi.float())), dim=1)
        else: # graph embedding only
            x = self.graph_block(x)

        x = self.fc_block(x)

        return x

    def save_args(self, save_dir):

        with open(path.join(save_dir, 'prodar_nn-args.json'), 'w+') as f_out:
            json.dump(self.all_args, f_out,
                      indent=4, separators=(',', ': '), sort_keys=True)

        with open(path.join(save_dir, 'prodar_nn-summary.txt'), 'w+') as sys.stdout:
            print(self, end='\n\n')
            summary(self)
        sys.stdout = sys.__stdout__


if __name__ == '__main__':

    model = ProDAR_NN(
        dim_node_feat=21, dim_pers_feat=625, dim_out=123,
        dim_node_hidden=256,
        dim_pers_embedding=512, dim_graph_embedding=512,
        dropout_rate=0.1,
        gnn_type='GCN', n_graph_layers=5,
        aggr='add',
        improved=False, add_self_loops=True
    )

    # print(model)
    # summary(model)

    model.save_args('/Users/sebastian/Downloads')
