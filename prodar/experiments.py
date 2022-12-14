#!/usr/bin/env python
# coding: utf-8

from .visualization import Plotter

import random
import numpy as np
from tqdm import tqdm
from datetime import datetime
import time
from os import cpu_count, path, makedirs

import torch
from torch import nn
from torch.utils.data import random_split
from torch.utils.data.dataset import Subset
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

    def log(self, msg):
        with open(self.exp_logfile, 'a+') as f_out:
            f_out.write(f'[{datetime.now().strftime("%Y/%m/%d %H:%M:%S")}] '
                        f'{msg}\n')
            f_out.flush()

    def __init__(self, nn_model, name_suffix='',
                 rand_seed=df_rand_seed, device=df_device):

        # set up save directory
        self.exp_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.save_dir = path.join(df_history_root,
                                  f'experiment-{self.exp_time}-{name_suffix}')
        makedirs(self.save_dir, exist_ok=True)

        self.exp_logfile = path.join(self.save_dir, 'experiment.log')

        # save model arguments
        nn_model.save_args(self.save_dir)

        # start class setup
        self.model = nn.DataParallel(nn_model).to(device)

        self.device = device

        self.torch_gen = torch.Generator()
        self.torch_gen.manual_seed(rand_seed)

        print(self.model)
        torchinfo.summary(self.model)

        self.f1_max_hist = np.empty((0,3)) # f1_max, prec, recall
        self.loss_acc_hist = np.empty((0,4))

        self.plotter = Plotter(self.save_dir)

        self.log(f'rand_seed: {rand_seed}')

    def _set_learning_rate(self, learning_rate):
        for group in optim.param_groups:
            group['lr'] = learning_rate

        self.log(f'lr: {learning_rate}')

    def set_loss_fn(self, loss_fn, pos_weight=None, **loss_kwargs):
        if pos_weight is not None:
            pos_weight = pos_weight.to(self.device)
        self.loss_fn = loss_fn(pos_weight=pos_weight, **loss_kwargs)

        self.log(f'loss_fn: {loss_fn.__name__}')

    def set_optimizer(self, optimizer_fn, lr, **optim_kwargs):
        self.optimizer = optimizer_fn(self.model.parameters(), lr,
                                      **optim_kwargs)

        self.log(f'optimizer_fn: {optimizer_fn.__name__}, lr: {lr}')

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

        self.train_mfgo_dict = self.get_mfgo_dict(self.train_dataloader)

        msg = (f'Training dataset: {len(self.train_dataloader.dataset)} '
               f'entries')
        print(msg)
        self.log(msg)

    def _set_valid_dataloader(self, valid_dataset, batch_size, num_workers):
        self.valid_dataloader = DataLoader(
           valid_dataset,
           batch_size=batch_size,
           num_workers=num_workers
        )

        msg = (f'Validation dataset: {len(self.valid_dataloader.dataset)} '
               f'entries')
        print(msg)
        self.log(msg)

    def _set_dataloaders(self, train_dataset, valid_dataset,
                         batch_size, shuffle=False,
                         seed_worker=seed_worker,
                         num_workers=cpu_count()):

        self._set_train_dataloader(train_dataset, batch_size,
                                   shuffle, num_workers, seed_worker)
        self._set_valid_dataloader(valid_dataset, batch_size,
                                   num_workers)

        self.log(f'Training and Validation dataloader set '
                 f'(batch_size: {batch_size}, shuffle: {shuffle}, '
                 f'num_workers: {num_workers})')

    def set_dataset(self, dataset):

        self.n_GO_terms = dataset.n_GO_terms
        self.dataset = dataset

    def set_valid_dataset(self, valid_dataset):
        if not valid_dataset.n_GO_terms == self.n_GO_terms:
            raise ValueError('Mismatching number of classes (MF-GO terms) in '
                             'datasets')

        self.valid_dataset = valid_dataset

    def _comp_tp_fp_tn_fn(self, output, data_y, thres=0.5):

        pred = torch.where(torch.sigmoid(output)>=thres, 1, 0)

        tp = torch.logical_and(pred==1, data_y==1).detach().sum().item()
        fp = torch.logical_and(pred==1, data_y==0).detach().sum().item()
        tn = torch.logical_and(pred==0, data_y==0).detach().sum().item()
        fn = torch.logical_and(pred==0, data_y==1).detach().sum().item()

        return np.array([tp, fp, tn, fn])

    def comp_metrics(self, tp, fp, tn, fn):

        with np.errstate(divide='ignore', invalid='ignore'):

            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            specificity = tn / (tn + fp)

            f1 = 2*recall*precision / (recall+precision)

        return precision, recall, specificity, f1

    def _train_one_epoch(self):

        # set model to training mode
        self.model.train()

        total_loss = 0.
        tp_fp_tn_fn = np.zeros((4,))

        for i, data in enumerate(tqdm(self.train_dataloader,
                                      desc=(f'    {"Training":10s}'),
                                      ascii=True, dynamic_ncols=True)):

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            data.x = data.x.float()
            data.y = torch.zeros((len(data.ID), self.n_GO_terms))
            for idx in range(len(data.ID)):
                ID = data.ID[idx]
                data.y[idx, self.train_mfgo_dict[ID]] = 1
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
    def _evaluate(self, dataloader, return_pr, action_name='Evaluation',
                  suppress_warning=False):

        if not hasattr(self, 'loss_fn') and not suppress_warning:
            print(' >> Warning: loss function not set')
        self.log('loss_fn not set')

        self.model.train(False)

        total_loss = 0.

        output_all = []
        data_y_all = []

        mfgo_dict = self.get_mfgo_dict(dataloader)

        for i, data in enumerate(tqdm(dataloader,
                                      desc=f'    {action_name:10s}',
                                      ascii=True, dynamic_ncols=True)):

            data.x = data.x.float()
            data.y = torch.zeros((len(data.ID), self.n_GO_terms))
            for idx in range(len(data.ID)):
                ID = data.ID[idx]
                data.y[idx, mfgo_dict[ID]] = 1
            data = data.to(self.device)

            output = self.model(data)

            if hasattr(self, 'loss_fn'):
                total_loss += self.loss_fn(output, data.y).item()

            output_all.append(output)
            data_y_all.append(data.y)

        avg_loss = total_loss / len(dataloader.dataset)

        # compute confusion matrix elements for all thresholds (0~1)
        if return_pr:
            n_intervals = 501
            tp_fp_tn_fn_all = np.empty((n_intervals,4))

            # iterate over threshold values
            for thres_idx, thres in enumerate(np.linspace(0,1,n_intervals)):

                tp_fp_tn_fn = np.zeros((4,))

                for idx in range(len(output_all)):
                    output = output_all[idx]
                    data_y = data_y_all[idx]

                    tp_fp_tn_fn += self._comp_tp_fp_tn_fn(output, data_y,
                                                          thres=thres)

                tp_fp_tn_fn_all[thres_idx] = tp_fp_tn_fn

            def_tp_fp_tn_fn = tp_fp_tn_fn_all[(n_intervals-1)//2]

            return avg_loss, def_tp_fp_tn_fn, tp_fp_tn_fn_all

        # compute confusion matrix elements for default threshold (0.5)
        else:
            tp_fp_tn_fn = np.zeros((4,))

            for idx in range(len(output_all)):
                output = output_all[idx]
                data_y = data_y_all[idx]

                tp_fp_tn_fn += self._comp_tp_fp_tn_fn(output, data_y)

            return avg_loss, tp_fp_tn_fn

    def validate(self, return_pr=False):
        return self._evaluate(dataloader=self.valid_dataloader,
                              return_pr=return_pr,
                              action_name='Validation')

    def test(self, test_dataset, batch_size=32, num_workers=cpu_count()):

        test_dataloader = DataLoader(
           test_dataset,
           batch_size=batch_size,
           num_workers=num_workers
        )

        np.savetxt(path.join(self.save_dir, 'test-id_list.txt'),
                   test_dataset.id_list, fmt='%s')

        print(f'Test dataset: {len(test_dataloader.dataset)} entries')

        return self._evaluate(dataloader=test_dataloader,
                              return_pr=False,
                              action_name='Testing')

    def get_pr_curve(self, dataloader, n_intervals=500):

        _, _, tp_fp_tn_fn_all = self._evaluate(dataloader,
                                               return_pr=True,
                                               action_name='PR eval')

        precision, recall, _, _ = self.comp_metrics(*tp_fp_tn_fn_all.T)

        return precision, recall

    def _train_valid_loop(self, n_epochs, train_hist_file,
                          plot_name=''):

        self.log(f'Training for {n_epochs} epochs')
        training_start_time = time.time()

        # write headers to history file
        if not path.exists(train_hist_file):
            with open(train_hist_file, 'w+') as f_out:
                f_out.write('epoch training_loss valid_loss '
                            'train_acc valid_acc f1_max\n')
                f_out.flush()

        lowest_vloss = 1e8
        best_f1 = 0
        # train_loss, valid_loss, train_acc, valid_acc
        f1_max_hist = np.empty((n_epochs,3))
        loss_acc_hist = np.empty((n_epochs,4))

        for idx in range(n_epochs):

            epoch_start_time = time.time()

            epoch_number = idx + 1

            print(f'\nEPOCH {epoch_number} of {n_epochs}')

            # Make sure gradient tracking is on, and do a pass over the data
            train_loss, tp_fp_tn_fn = self._train_one_epoch()
            _, train_recall, train_spec, _ = self.comp_metrics(*tp_fp_tn_fn)

            valid_loss, tp_fp_tn_fn, tp_fp_tn_fn_all = self.validate(return_pr=True)
            _, valid_recall, valid_spec, _ = self.comp_metrics(*tp_fp_tn_fn)

            # compute all metrics
            precision, recall, _, f1 = self.comp_metrics(*tp_fp_tn_fn_all.T)
            f1_max_idx = np.nanargmax(f1)
            f1_max = f1[f1_max_idx]
            f1_max_hist[idx] = [f1_max, precision[f1_max_idx], recall[f1_max_idx]]

            train_acc = ( train_recall + train_spec )/2
            valid_acc = ( valid_recall + valid_spec )/2

            print(f'    <LOSS> train: {train_loss:.10f}, '
                  f'valid: {valid_loss:.10f}')
            print(f'    <ACC>  train: {train_acc:.10f}, '
                  f'valid: {valid_acc:.10f}')
            loss_acc_hist[idx] = [train_loss, valid_loss, train_acc, valid_acc]

            # write training history to drive
            msg = (f'{epoch_number:.15f} {train_loss:.15f} {valid_loss:.15f} '
                   f'{train_acc:.15f} {valid_acc:.15f} {f1_max:.15f}')
            with open(train_hist_file, 'a+') as f_out:
                f_out.write(msg+'\n')
                f_out.flush()

            # Track best performance, and save the model's state
            if valid_loss < lowest_vloss:
                lowest_vloss = valid_loss
                self.save_params(prefix='lowest_loss')

            if f1_max > best_f1:
                best_f1 = f1_max
                self.save_params(prefix='best_f1')

            filename_suffix = (
                f'epoch_{epoch_number}' if plot_name is None else
                f'{plot_name}-epoch_{epoch_number}'
            )
            if epoch_number % 25 == 0:
                self.plotter.plot_pr(precision, recall,
                                     filename_suffix=filename_suffix)

            self.log(f'Epoch {epoch_number} complete with f1_max={f1_max:.15f}'
                     f' (Wall time: {int(time.time()-epoch_start_time)})')

        self.save_params(prefix='last')

        self.f1_max_hist = np.vstack((self.f1_max_hist, f1_max_hist))
        self.plotter.plot_f1_max_hist(self.f1_max_hist)

        self.loss_acc_hist = np.vstack((self.loss_acc_hist, loss_acc_hist))
        self.plotter.plot_loss_acc_hist(self.loss_acc_hist)

        self.log(f'Finish training for {n_epochs} epochs '
                 f'(Wall time: {int(time.time()-training_start_time)})')

        return loss_acc_hist, f1_max_hist

    def train_valid(self, n_epochs, batch_size=64):
        pass

    def train_split(self, n_epochs, train_valid_ratio=0.9, batch_size=64):

        # split dataset into training and validation sets
        if float(train_valid_ratio) >= 1.0 and float(train_valid_ratio) > 0.0:
            msg = (f'Cannot split training-validation set with '
                   f'ratio {train_valid_ratio}')
            self.log(msg)
            raise ValueError(msg)

        msg = (f'Spliting \'{self.dataset.set_name}\' into training and '
               f'validation dataset with ratio {train_valid_ratio}')
        print(msg)
        self.log(msg)

        n_total = len(self.dataset)
        if n_total > batch_size:
            n_train_data = int(
                n_total*train_valid_ratio - (n_total*train_valid_ratio)%batch_size
            )
        else:
            n_train_data = int(n_total*train_valid_ratio)
        n_valid_data = n_total - n_train_data

        #### TODO ####
        # ensure that every MF-GO appears in the training set at least
        # once (currently it is possible that some MF-GO is only present
        # in the validation set, or vice versa)
        ##############

        train_dataset, valid_dataset = random_split(
            self.dataset,
            [n_train_data, n_valid_data],
            generator=self.torch_gen
        )

        self._set_dataloaders(train_dataset,
                              valid_dataset=valid_dataset,
                              batch_size=batch_size,
                              shuffle=False)

        # save IDs in both datasets for reference
        dataset_id_list = np.array(self.dataset.id_list)
        np.savetxt(path.join(self.save_dir, 'train-id_list.txt'),
                   dataset_id_list[train_dataset.indices], fmt='%s')
        np.savetxt(path.join(self.save_dir, 'valid-id_list.txt'),
                   dataset_id_list[valid_dataset.indices], fmt='%s')

        train_hist_file = path.join(self.save_dir, 'training_hist.txt')
        loss_acc_hist, f1_max_hist = self._train_valid_loop(
            n_epochs=n_epochs,
            train_hist_file=train_hist_file
        )

    def train_kfold(self, n_epochs=300, n_folds=5, batch_size=64):

        self.log(f'Training with {n_folds} folds with {n_epochs} epochs each')

        indices = np.arange(len(self.dataset))
        n_valid = len(self.dataset)//5
        np.random.shuffle(indices)

        # save id list for all folds
        for fold_idx in range(n_folds):
            fold_num = fold_idx + 1
            if fold_num < n_folds:
                valid_idx = indices[fold_idx*n_valid:(fold_idx+1)*n_valid]
            else:
                valid_idx = indices[fold_idx*n_valid:]

            id_list_filename = (
                f'holdout_{fold_num}_of_{n_folds}folds-id_list.txt'
            )
            np.savetxt(path.join(self.save_dir, id_list_filename),
                       np.array(self.dataset.id_list)[valid_idx], fmt='%s')

        self.log(f'id_list.txt for all {n_folds} folds saved')

        for fold_idx in range(n_folds):
            fold_num = fold_idx + 1
            print(f'\n###########\nFold {fold_num} of {n_folds}\n###########')

            # split data into training and validation
            if fold_num < n_folds:
                valid_idx = indices[fold_idx*n_valid:(fold_idx+1)*n_valid]
            else:
                valid_idx = indices[fold_idx*n_valid:]
            train_idx = np.setdiff1d(indices, valid_idx)
            print(type(train_idx))
            print(train_idx.shape)
            print(type(valid_idx))
            print(valid_idx.shape)
            self._set_dataloaders(self.dataset[list(train_idx)],
                                  valid_dataset=self.dataset[list(valid_idx)],
                                  batch_size=batch_size,
                                  shuffle=False)

            # re-initialize parameters
            for layer in self.model.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

            # train for n_epochs epochs
            train_hist_file = path.join(self.save_dir,
                                        f'training_hist-fold_{fold_num}.txt')
            loss_acc_hist, f1_max_hist = self._train_valid_loop(
                n_epochs=n_epochs,
                train_hist_file=train_hist_file,
                plot_name=f'fold_{fold_num}'
            )

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

        self.log(f'Model saved: {prefix}')

    def load_params(self, params_file):

        print(f'Loading model parameters from file {params_file}...')

        self.model = torch.load(params_file,
                                map_location=self.device)#.to(self.device)

        self.log(f'Model loaded: {params_file}')

    def get_mfgo_dict(self, dataloader):

        # print(Subset)
        # print(type(dataloader))
        if isinstance(dataloader.dataset, Subset):
            return dataloader.dataset.dataset.mfgo_dict
        else:
            # print(type(datloader.dataset))
            return dataloader.dataset.mfgo_dict


if __name__ == '__main__':

    # from .../data import datasets
    from prodar import model

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
        name_suffix=f'{dataset.set_name}-{dataset.go_thres}-dell',
        rand_seed=69
    )
    plt = Plotter(save_dir=exp.save_dir)

    # experiment setup
    exp._set_dataloaders(dataset[:20], batch_size=64)
    exp.set_loss_fn(nn.BCEWithLogitsLoss, pos_weight=dataset.pos_weight)
    exp.set_optimizer(torch.optim.Adam, lr=0.000025)

    exp.train_valid_kfold(n_folds=5)
