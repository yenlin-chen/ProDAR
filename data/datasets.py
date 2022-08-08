#!/usr/bin/env python
# coding: utf-8

if __name__ == '__main__':
    from preprocessing import (
        preprocessor as pp,
        utils
    )
else:
    from .preprocessing import (
        preprocessor as pp,
        utils
    )
from os import path
import json
import networkx as nx
import numpy as np
import torch
import torch_geometric as pyg
# from torch_geometric.data import Data, Dataset
# from torch_geometric.utils import from_networkx
from tqdm import tqdm


self_dir = path.dirname(path.realpath(__file__))

df_processed_root = path.join(self_dir, 'pyg_processed')

# load indices of residues
with open(path.join(self_dir, 'residues.json'), 'r') as fin:
    res_dict = json.load(fin)

def remove_PI(data):
    '''
    Removes the persistence image attribute from pyg object.
    '''
    del data.pi
    return data

class ProDAR_Dataset(pyg.data.Dataset):

    def __init__(self, set_name, entry_type, cont, corr, pers,
                 cutoff, gamma, thres, n_modes, simplex):

        '''

        '''

        print('Initializing dataset...')

        if not any((cont, corr)):
            raise ValueError('At least one pipeline must be turned on.')
        else:
            print('  Contact pipeline in activated.') if cont else None
            print('  Correlation pipeline in activated.') if corr else None
            print('  Persistence homology pipeline in activated.'
                  '') if pers else None

        self.set_name = set_name
        self.entry_type = entry_type

        ################################################################
        # folder names containing data of the specified setup
        ################################################################
        self.nma_setup = pp.nma_setup_folder_name.format(cutoff,
                                                         gamma,
                                                         thres,
                                                         n_modes)
        self.pi_setup = pp.pi_setup_folder_name.format(simplex)
        self.contCorr_setup = f'cont_{cont}-corr_{corr}'
        self.folder_name = (f'{self.nma_setup}-{self.pi_setup}-'
                                f'{self.contCorr_setup}')

        self.raw_graph_dir = path.join(pp.df_graph_root, self.nma_setup)
        self.raw_pi_dir = path.join(pp.df_pi_root, self.pi_setup)

        self.stats_root = path.join(pp.df_stats_root, set_name)
        target_dir = path.join(self.stats_root, 'target')

        self.stats_dir = path.join(self.stats_root, self.folder_name)

        ################################################################
        # get list of IDs for this dataset
        ################################################################
        file1 = path.join(self.stats_dir, pp.df_mfgo_filename)
        file2 = path.join(target_dir, pp.df_mfgo_filename)
        if path.exists(file1):
            self.mfgo_file = file1
        elif path.exists(file2):
            self.mfgo_file = file2
        else:
            FileNotFoundError(f'Cannot find MF-GO file '
                              f'{pp.df_mfgo_filename} for selection '
                              f'{set_name}')

        # with open(self.mfgo_file, 'r') as f_in:
        #     self.mfgo_dict = json.load(f_in)
        # self.id_list = [ID for ID in self.mfgo_dict]

        # self.entry_type = utils.check_id_type(self.id_list)


        ################################################################
        # save all dataset parameters
        ################################################################
        self.cont = cont
        self.corr = corr
        self.pers = pers

        self.cutoff = cutoff
        self.gamma = gamma
        self.thres = thres
        self.n_modes = n_modes

        self.simplex = simplex

        ################################################################
        # misc
        ################################################################
        # self.n_GO_terms = max([e for v in self.mfgo_dict.values()
        #                          for e in v]) + 1

        ################################################################
        # Call constuctor of parent class
        ################################################################
        transform = remove_PI if not pers else None

        super().__init__(self.raw_graph_dir, transform, None, None)
        print('Dataset Initialization Complete\n')

    @property
    def mfgo_dict(self):
        with open(self.mfgo_file, 'r') as f_in:
            mfgo_dict = json.load(f_in)
        return mfgo_dict

    @property
    def id_list(self):
        return [ID for ID in self.mfgo_dict]

    @property
    def raw_dir(self): # -> str:
        return self.raw_graph_dir

    @property
    def processed_dir(self): # -> str:
        return path.join(df_processed_root, self.folder_name)

    @property
    def raw_file_names(self):
        return [f'{ID}.json' for ID in self.id_list]

    @property
    def processed_file_names(self):
        return [f'{ID}.pt' for ID in self.id_list]

    def download(self):
        process = pp.Preprocessor(set_name=self.set_name,
                                  entry_type=self.entry_type)
        process.preprocess(simplex=self.simplex,
                           cutoff=self.cutoff,
                           gamma=self.gamma,
                           corr_thres=self.thres,
                           n_modes=self.n_modes,
                           retry_failed=False,
                           rebuild_pi=False,
                           rebuild_graph=False,
                           update_mfgo=True,
                           verbose=True)
        self.mfgo_file = path.join(self.stats_dir, pp.df_mfgo_filename)

    def process(self):
        count = 0
        for idx, ID in enumerate(tqdm(self.id_list, desc='  Processing data',
                                      ascii=True, dynamic_ncols=True)):
            with open(path.join(self.raw_graph_dir, ID+'.json'), 'r') as fin:
                js_graph = json.load(fin)

            nx_graph = nx.readwrite.json_graph.node_link_graph(js_graph)
            data = pyg.utils.from_networkx(nx_graph)

            # delete correlation edge to turn off the corr pipeline
            if not self.corr:
                indices = torch.argwhere(data.weight==1).squeeze()
                data.edge_index = torch.index_select(data.edge_index,
                                                     1, indices)
            # delete contact edge to turn off the contact pipeline
            if not self.cont:
                indices = torch.argwhere(data.weight==-1).squeeze()
                data.edge_index = torch.index_select(data.edge_index,
                                                     1, indices)

            ############################################################
            # residue type
            ############################################################

            n_unique_residues = np.unique(list(res_dict.values()))
            x = np.zeros((len(data.resname), len(n_unique_residues)))#,
                         # dtype=np.int_)

            for j, residue in enumerate(data.resname):
                if residue not in res_dict:
                    residue = 'XAA'
                x[j, res_dict[residue]] = 1

            data.x = torch.from_numpy(x)

            pi = np.load(path.join(self.raw_pi_dir, ID+'.npy'))
            data.pi = torch.from_numpy(pi)

            ############################################################
            # MFGO
            ############################################################

            self.n_GO_terms = ( max([e for v in self.mfgo_dict.values()
                                     for e in v]) + 1 )
            y = np.zeros((1, self.n_GO_terms))#, dtype=np.in)
            y[0, self.mfgo_dict[ID]] += 1

            data.y = torch.from_numpy(y)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            # remove unnecessary objects

            del data.resname, data.weight

            processed_filename = ID+'.pt'
            torch.save(data, path.join(self.processed_dir, processed_filename))

            # count+=1
            # if count == 5:
            #     break

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        return torch.load(path.join(self.processed_dir,
                                    self.processed_file_names[idx]))


class Contact8A(ProDAR_Dataset):

    def __init__(self, set_name, entry_type):

        cont, corr, pers = True, False, False
        cutoff = 8

        super().__init__(set_name=set_name, entry_type=entry_type,
                         cont=cont, corr=corr, pers=pers,
                         cutoff=cutoff, gamma=pp.df_gamma, thres=pp.df_corr_thres,
                         n_modes=pp.df_n_modes, simplex=pp.df_simplex)

    @property
    def raw_dir(self): # -> str:
        return super().raw_dir

    @property
    def processed_dir(self): # -> str:
        return super().processed_dir

    @property
    def raw_file_names(self):
        return super().raw_file_names

    @property
    def processed_file_names(self):
        return super().processed_file_names

    def download(self):
        return super().download()

    def process(self):
        return super().process()

    def len(self):
        return super().len()

    def get(self, idx):
        return super().get(idx)

class ContactCorrPers8A(ProDAR_Dataset):

    def __init__(self, set_name, entry_type):

        cont, corr, pers = True, True, True
        cutoff = 8

        super().__init__(set_name=set_name, entry_type=entry_type,
                         cont=cont, corr=corr, pers=pers,
                         cutoff=cutoff, gamma=pp.df_gamma, thres=pp.df_corr_thres,
                         n_modes=pp.df_n_modes, simplex=pp.df_simplex)

    @property
    def raw_dir(self): # -> str:
        return super().raw_dir

    @property
    def processed_dir(self): # -> str:
        return super().processed_dir

    @property
    def raw_file_names(self):
        return super().raw_file_names

    @property
    def processed_file_names(self):
        return super().processed_file_names

    def download(self):
        return super().download()

    def process(self):
        return super().process()

    def len(self):
        return super().len()

    def get(self, idx):
        return super().get(idx)

class ContactCorrPers12A(ProDAR_Dataset):

    def __init__(self, set_name):

        cont, corr, pers = True, True, True
        cutoff = 12

        super().__init__(set_name=set_name, cont=cont, corr=corr, pers=pers,
                         cutoff=cutoff, gamma=pp.df_gamma, thres=pp.df_corr_thres,
                         n_modes=pp.df_n_modes, simplex=pp.df_simplex)

    @property
    def raw_dir(self): # -> str:
        return super().raw_dir

    @property
    def processed_dir(self): # -> str:
        return super().processed_dir

    @property
    def raw_file_names(self):
        return super().raw_file_names

    @property
    def processed_file_names(self):
        return super().processed_file_names

    def download(self):
        return super().download()

    def process(self):
        return super().process()

    def len(self):
        return super().len()

    def get(self, idx):
        return super().get(idx)

if __name__ == '__main__':

    # dataset = ProDAR_Dataset(set_name='newtest',
    #                          cont=True, corr=True, pers=False,
    #                          cutoff=pp.df_cutoff, gamma=pp.df_gamma,
    #                          thres=pp.df_corr_thres, n_modes=pp.df_n_modes,
    #                          simplex=pp.df_simplex)

    # dataset = Contact8A(set_name='newtest')

    dataset = ContactCorrPers8A(set_name='test',
                                entry_type='chain')


    # dataset = ContactCorrPers8A(set_name='deepfri-all',
    #                             entry_type='chain')
