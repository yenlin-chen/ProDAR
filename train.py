#!/usr/bin/env python
# coding: utf-8

from data import datasets
from prodar import prodar_nn
from os import cpu_count

from math import floor
import torch

df_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class ProDAR_Experiment():

    def __init__(self, nn_model, loss_fn, device=df_device):

        self.model = nn_model
        self.loss_fn = loss_fn

        self.device = device

    def set_optimizer(self, optimizer_func, lr, **optim_kwargs):
        self.optimizer = optimizer_func(self.model.parameters(), lr,
                                        **optim_kwargs)

    def _set_learning_rate(self, learning_rate):
        for g in optim.param_groups:
            g['lr'] = learning_rate

    def save_model(self, name):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = f'model_{timestamp}_{name}'
        torch.save(self.model.state_dict(), model_path)

    def _set_train_dataloader(self, train_dataset, batch_size, shuffle,
                              num_workers, seed_worker):
        self.n_train_data = len(train_dataset)
        self.train_dataloader = DataLoader(
           train_dataset,
           batch_size=batch_size,
           shuffle=True,
           num_workers=num_workers,
           worker_init_fn=seed_worker,
           generator=torch.Generator(self.device)
        )

        print(f'Training dataset: {self.n_train_data} entries')

    def _set_valid_dataloader(self, valid_dataset, batch_size, num_workers):
        self.n_valid_data = len(valid_dataset)
        self.valid_dataloader = DataLoader(
           valid_dataset,
           batch_size=batch_size,
           num_workers=num_workers
        )

        print(f'Validation dataset: {self.n_valid_data} entries')

    def set_dataloaders(self, dataset, valid_dataset=None,
                        train_valid_split=0.9,
                        batch_size=32, shuffle=True,
                        seed_worker=seed_worker,
                        num_workers=cpu_count()):

        if valid_dataset:
            print(f'Using {valid_dataset.set_name} for validation.')
            self._set_train_dataloader(dataset, batch_size,
                                       shuffle, num_workers, seed_worker)
            self._set_valid_dataloader(valid_dataset, batch_size,
                                       num_workers)
        else:
            print(f'Spliting {dataset.set_name} into training and '
                  f'validation dataset with ratio {train_valid_split}')
            n_train_data = floor(n_data*0.9)
            self._set_train_dataloader(dataset[:n_train_data], batch_size,
                                       shuffle, num_workers, seed_worker)
            self._set_valid_dataloader(dataset[n_train_data:], batch_size,
                                       num_workers)

    def _train_one_epoch(self):

        # set model to training mode
        self.model.train()
        running_loss = 0.
        last_loss = 0.

        for i, data in enumerate(tqdm(self.training_dataloader,
                                       desc='Training dataset',
                                       ascii=True, dynamic_ncols=True,
                                       position=1)):

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # Make predictions for this batch
            output = self.model(data.to(self.device))

            # Compute the loss and its gradients
            loss = self.loss_fn(output, data.y)
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            total_loss += loss.item()
            if i % 10 == 9:
                last_loss = running_loss / 10 # loss per batch
                print(f'  batch {i + 1} loss: {last_loss}')
                running_loss = 0.

        avg_loss = total_loss / self.n_train_data

        return avg_loss

    def validate(self):

        self.model.train(False)

        running_vloss = 0.
        for i, vdata in enumerate(tqdm(self.validation_loader,
                                       desc='Validation datset',
                                       ascii=True, dynamic_ncols=True,
                                       position=1)):

            y_vpred = self.model(data.to(self.device))

            vloss = self.loss_fn(y_vpred, data.y)
            running_vloss += vloss.item()
        avg_vloss = running_vloss / self.n_valid_data
        print(f'LOSS train {avg_loss} valid {avg_vloss}')

        return avg_vloss

    def train_valid_loop(self, n_epoch=300):

        best_vloss = 1e8

        for idx in tqdm(range(n_epochs), desc='Training Model',
                        ascii=True, dynamic_ncols=True, position=0):
            epoch_number = idx + 1

            print('EPOCH {}:'.format(epoch_number + 1))

            # Make sure gradient tracking is on, and do a pass over the data
            avg_loss = self.train_one_epoch(epoch_number)

            avg_vloss = self.validate()

            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                self.save_model()

            epoch_number += 1


    def kfold(n_folds=5):
        pass


    def hyperparameter_grid_search(self):
        pass

if __name__ == '__main__':

    dataset = datasets.ContactCorrPers8A(set_name='test', entry_type='chain')

    model = prodar_nn.ProDAR_NN(
        dim_node_feat=16, dim_pers_feat=25, dim_out=100,
        dim_node_hidden=32,
        dim_pers_embedding=8, dim_graph_embedding=4,
        dropout_rate=0.3,
        gnn_type='GCN', n_graph_layers=3,
        aggr='add'
    )

    optimizer = torch.optim.Adam

    ProDAR_Experiment(dataset, model, optimizer)
