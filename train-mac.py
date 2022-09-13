#!/usr/bin/env python
# coding: utf-8

from prodar.model import ProDAR_NN
from prodar.experiments import ProDAR_Experiment
from prodar.datasets import ContactCorrPers8A
from prodar.visualization import Plotter

import torch
from torch import nn
# import torchinfo

if __name__ == '__main__':

    ####################################################################
    # set up experiemnt
    ####################################################################

    dataset = ContactCorrPers8A(set_name='original-restrained_5k',
                                go_thres=25,
                                entry_type='chain')

    # print(len(dataset))

    model = ProDAR_NN(
        dim_node_feat=21, dim_pers_feat=625, dim_out=dataset.n_GO_terms,
        dim_node_hidden=256,
        dim_pers_embedding=512, dim_graph_embedding=512,
        dropout_rate=0.1,
        gnn_type='GCN', n_graph_layers=5,
        aggr='add',
        improved=False, add_self_loops=True
    )

    exp = ProDAR_Experiment(
        model,
        name_suffix=f'{dataset.set_name}-{dataset.go_thres}-mac',
        rand_seed=69
    )
    plt = Plotter(save_dir=exp.save_dir)

    # experiment setup
    exp.set_dataset(dataset[:20])
    exp.set_loss_fn(nn.BCEWithLogitsLoss, pos_weight=dataset.pos_weight)
    exp.set_optimizer(torch.optim.Adam, lr=0.000025)

    ####################################################################
    # train
    ####################################################################

    exp.train_split(n_epochs=2, train_valid_ratio=0.9, batch_size=1)

    ####################################################################
    # plot stuff
    ####################################################################

    # plot pr_curve for model from the last iteration
    plt.plot_pr(*exp.get_pr_curve(exp.valid_dataloader),
                filename_suffix='last')

    from os import path

    # plot pr_curve for model with best validation accuracy
    exp.load_params(path.join(exp.save_dir, 'lowest_loss-model.pkl'))
    plt.plot_pr(*exp.get_pr_curve(exp.valid_dataloader),
                filename_suffix='lowest_loss')

    # plot pr_curve for model with best validation accuracy
    exp.load_params(path.join(exp.save_dir, 'best_f1-model.pkl'))
    plt.plot_pr(*exp.get_pr_curve(exp.valid_dataloader),
                filename_suffix='best_f1')
