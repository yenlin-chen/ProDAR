#!/usr/bin/env python
# coding: utf-8

import random
import numpy as np
from tqdm import tqdm
from datetime import datetime
from os import cpu_count, path, makedirs

import torch
from torch import nn
from torch.utils.data import random_split
import torchinfo

from torch_geometric.loader import DataLoader

self_dir = path.dirname(path.realpath(__file__))
df_history_root = path.join(self_dir, 'history')

df_rand_seed = 69
df_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class ProDAR_Experiment():

    def __init__(self, nn_model, rand_seed=df_rand_seed, device=df_device):

        # set up save directory
        self.exp_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.save_dir = path.join(df_history_root,
                                  f'experiment-{self.exp_time}')
        makedirs(self.save_dir, exist_ok=True)

        # save model arguments
        nn_model.save_args(self.save_dir)

        # start class setup
        self.model = nn.DataParallel(nn_model).to(device)

        self.device = device

        self.torch_gen = torch.Generator()
        self.torch_gen.manual_seed(rand_seed)

        print(self.model)
        torchinfo.summary(self.model)

    def _set_learning_rate(self, learning_rate):
        for group in optim.param_groups:
            group['lr'] = learning_rate

    def _set_train_dataloader(self, train_dataset, batch_size, shuffle,
                              num_workers, seed_worker):
        self.train_dataloader = DataLoader(
           train_dataset,
           batch_size=batch_size,
           shuffle=shuffle,
           num_workers=num_workers,
           worker_init_fn=seed_worker,
           generator=self.torch_gen
        )

        print(f'Training dataset: {len(self.train_dataloader.dataset)} '
              f'entries')

    def _set_valid_dataloader(self, valid_dataset, batch_size, num_workers):
        self.valid_dataloader = DataLoader(
           valid_dataset,
           batch_size=batch_size,
           num_workers=num_workers
        )

        print(f'Validation dataset: {len(self.valid_dataloader.dataset)} '
              f'entries')

    def set_dataloaders(self, dataset, valid_dataset=None,
                        train_valid_split=0.9,
                        batch_size=32, shuffle=False,
                        seed_worker=seed_worker,
                        num_workers=cpu_count()):

        if valid_dataset:
            print(f'Using \'{valid_dataset.set_name}\' for validation.')
            train_dataset = dataset
            self._set_train_dataloader(train_dataset, batch_size,
                                       shuffle, num_workers, seed_worker)
            self._set_valid_dataloader(valid_dataset, batch_size,
                                       num_workers)

            # save IDs in both datasets for reference
            np.savetxt(path.join(self.save_dir, 'train-id_list.txt'),
                       train_dataset.id_list, fmt='%s')
            np.savetxt(path.join(self.save_dir, 'valid-id_list.txt'),
                       valid_dataset.id_list, fmt='%s')

        else:
            print(f'Spliting \'{dataset.set_name}\' into training and '
                  f'validation dataset with ratio {train_valid_split}')

            n_train_data = int(len(dataset)*9//(10*batch_size)*batch_size)
            n_valid_data = len(dataset) - n_train_data
            train_dataset, valid_dataset = random_split(
                dataset,
                [n_train_data, n_valid_data],
                generator=self.torch_gen
            )

            self._set_train_dataloader(train_dataset, batch_size,
                                       shuffle, num_workers, seed_worker)
            self._set_valid_dataloader(valid_dataset, batch_size,
                                       num_workers)

            # save IDs in both datasets for reference
            np.savetxt(path.join(self.save_dir, 'train-id_list.txt'),
                       train_dataset.dataset.id_list, fmt='%s')
            np.savetxt(path.join(self.save_dir, 'valid-id_list.txt'),
                       valid_dataset.dataset.id_list, fmt='%s')

    def set_loss_fn(self, loss_fn, pos_weight=None, **loss_kwargs):
        if pos_weight is not None:
            pos_weight = pos_weight.to(self.device)
        self.loss_fn = loss_fn(pos_weight=pos_weight, **loss_kwargs)

    def set_optimizer(self, optimizer_fn, lr, **optim_kwargs):
        self.optimizer = optimizer_fn(self.model.parameters(), lr,
                                      **optim_kwargs)

    def _comp_tp_fp_tn_fn(self, output, data_y, thres=0.5):

        pred = torch.where(torch.sigmoid(output)>=thres, 1, 0)

        tp = torch.logical_and(pred==1, data_y==1).detach().sum().item()
        fp = torch.logical_and(pred==1, data_y==0).detach().sum().item()
        tn = torch.logical_and(pred==0, data_y==0).detach().sum().item()
        fn = torch.logical_and(pred==0, data_y==1).detach().sum().item()

        return np.array([tp, fp, tn, fn])

    def _train_one_epoch(self, epoch_number=0, thers=0.5):

        # set model to training mode
        self.model.train()

        total_loss = 0.
        tp_fp_tn_fn = np.zeros((4,))

        for i, data in enumerate(tqdm(self.train_dataloader,
                                      desc=(f'    {"Training":10s}'),
                                      ascii=True, dynamic_ncols=True)):

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            data.x, data.y = data.x.float(), data.y.float()
            data = data.to(self.device)

            # Make predictions for this
            output = self.model(data)

            # Compute the loss and its gradients
            loss = self.loss_fn(output, data.y)
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            total_loss += loss.item()

            # confusion matrix elements
            tp_fp_tn_fn += self._comp_tp_fp_tn_fn(output, data.y)

        avg_loss = total_loss / len(self.train_dataloader.dataset)

        return avg_loss, tp_fp_tn_fn

    @torch.no_grad()
    def _evaluate(self, dataloader, action_name='Evaluation'):

        self.model.train(False)

        total_loss = 0.
        tp_fp_tn_fn = np.zeros((4,))

        for i, data in enumerate(tqdm(dataloader,
                                      desc=f'    {action_name:10s}',
                                      ascii=True, dynamic_ncols=True)):

            data = data.to(self.device)
            data.x, data.y = data.x.float(), data.y.float()

            output = self.model(data)

            loss = self.loss_fn(output, data.y)

            total_loss += loss.item()

            # confusion matrix elements
            tp_fp_tn_fn += self._comp_tp_fp_tn_fn(output, data.y)

        avg_loss = total_loss / len(dataloader.dataset)

        return avg_loss, tp_fp_tn_fn

    def validate(self):
        return self._evaluate(action_name='Validation',
                                  dataloader=self.valid_dataloader)

    def test(self, test_dataset, batch_size=32, num_workers=cpu_count()):

        test_dataloader = DataLoader(
           test_dataset,
           batch_size=batch_size,
           num_workers=num_workers
        )

        np.savetxt(path.join(self.save_dir, 'test-id_list.txt'),
                   test_dataset.id_list, fmt='%s')

        print(f'Test dataset: {len(test_dataloader.dataset)} entries')

        return self._evaluate(action_name='Testing',
                              dataloader=test_dataloader)

    def comp_metrics(self, tp, fp, tn, fn):

        ppv = tp / (tp + fp)
        tpr = tp / (tp + fn)
        tnr = tn / (tn + fp)

        return ppv, tpr, tnr # precision, recall, specificity

    def train_valid_loop(self, n_epoch=300):

        best_vloss = 1e8

        # for idx in tqdm(range(n_epoch), desc='Training Model',
        #                 ascii=True, dynamic_ncols=True, position=0):
        for idx in range(n_epoch):

            epoch_number = idx + 1

            print(f'\nEPOCH {epoch_number} of {n_epoch}')

            # Make sure gradient tracking is on, and do a pass over the data
            train_loss, tp_fp_tn_fn = self._train_one_epoch()
            _, train_tpr, train_tnr = self.comp_metrics(*tp_fp_tn_fn)

            valid_loss, tp_fp_tn_fn = self.validate()
            _, valid_tpr, valid_tnr = self.comp_metrics(*tp_fp_tn_fn)

            train_acc = ( train_tpr + train_tnr )/2
            valid_acc = ( valid_tpr + valid_tnr )/2

            print(f'    <LOSS> train: {train_loss:.10f}, '
                  f'valid: {valid_loss:.10f}')
            print(f'    <ACC>  train: {train_acc:.10f}, '
                  f'valid: {valid_acc:.10f}')

            # Track best performance, and save the model's state
            if valid_loss < best_vloss:
                best_vloss = valid_loss
                self.save_params(prefix='best')

            epoch_number += 1

        self.save_params(prefix='last')

    def kfold(n_folds=5):
        pass

    def hyperparameter_grid_search(self):
        pass

    def save_params(self, prefix=None):

        # save entire model pickled
        torch.save(self.model,
                   path.join(self.save_dir, f'{prefix}-model.pkl'))

        # model parameters
        torch.save(self.model.state_dict(),
                   path.join(self.save_dir, f'{prefix}-model-state_dict.pt'))
        # optimizer parameters
        torch.save(self.optimizer.state_dict(),
                   path.join(self.save_dir, f'{prefix}-optim-state_dict.pt'))

    def load_params(self, params_file):

        self.model = torch.load(params_file).to(self.device)

    @torch.no_grad()
    def get_pr_curve(self, dataloader, n_intervals=100):

        # make pass through model and record output of last layer
        self.model.train(False)

        output_all = []
        data_y_all = []

        for i, data in enumerate(tqdm(dataloader,
                                      desc=f'    {"PR eval":10s}',
                                      ascii=True, dynamic_ncols=True)):

            data = data.to(self.device)
            data.x, data.y = data.x.float(), data.y.float()

            output = self.model(data)

            output_all.append(output)
            data_y_all.append(data.y)

        # compute confusion matrix elements for all thresholds (0~1)
        tp_fp_tn_fn_all = np.empty((n_intervals,4))

        for thres_idx, thres in enumerate(np.linspace(0,1,n_intervals)):

            for idx in range(len(output_all)):
                output = output_all[idx]
                data_y = data_y_all[idx]

                tp_fp_tn_fn += self._comp_tp_fp_tn_fn(output, data_y,
                                                      thres=thres)

            tp_fp_tn_fn_all[thres_idx] = tp_fp_tn_fn

        # compute metrics
        precision, recall, _ = self.comp_metrics(*tp_fp_tn_fn_all)

        aupr = np.trapz(np.flip(precision), x=np.flip(recall))
        f1 = 2*recall*precision / (recall+precision)
        f1_max_idx = np.argmax(f1)
        f1_max = f1[f1_max_idx]

        return precision, recall

if __name__ == '__main__':


    # from .../data import datasets
    from prodar import model

    dataset = datasets.ContactCorrPers8A(set_name='test', entry_type='chain')

    model = prodar_nn.ProDAR_NN(
        dim_node_feat=21, dim_pers_feat=625, dim_out=123,
        dim_node_hidden=256,
        dim_pers_embedding=512, dim_graph_embedding=512,
        dropout_rate=0.1,
        gnn_type='GCN', n_graph_layers=5,
        aggr='add',
        improved=False, add_self_loops=True
    )

    loss_fn = nn.BCEWithLogitsLoss()
    optim_fn = torch.optim.Adam

    ProDAR_Experiment(model, loss_fn)
