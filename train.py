#!/usr/bin/env python
# coding: utf-8

from prodar.model import ProDAR_NN
from prodar.experiments import ProDAR_Experiment
from data import datasets

import torch
from torch import nn
# import torchinfo

if __name__ == '__main__':

    dataset = datasets.ContactCorrPers8A(set_name='original_7k',
                                         entry_type='chain')

    model = ProDAR_NN(
        dim_node_feat=21, dim_pers_feat=625, dim_out=dataset.n_GO_terms,
        dim_node_hidden=256,
        dim_pers_embedding=512, dim_graph_embedding=512,
        dropout_rate=0.1,
        gnn_type='GCN', n_graph_layers=5,
        aggr='add',
        improved=False, add_self_loops=True
    )



    exp = ProDAR_Experiment(model)

    exp.set_dataloaders(dataset, batch_size=32)

    exp.set_loss_fn(nn.BCEWithLogitsLoss, pos_weight=dataset.pos_weight)
    exp.set_optimizer(torch.optim.Adam, lr=0.00005)

    exp.train_valid_loop(n_epoch=300)
