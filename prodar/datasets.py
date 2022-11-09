#!/usr/bin/env python
# coding: utf-8

from .data.preprocessing import (
    preprocessor as pp,
    utils
)
from .data import res_dict

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
data_dir = path.join(self_dir, 'data')
df_processed_root = path.join(data_dir, 'pyg_processed')

def remove_PI(data):
    '''
    Removes the persistence image attribute from pyg object.
    '''
    del data.pi
    return data

class ProDAR_Dataset(pyg.data.Dataset):

    def __init__(self, set_name, go_thres, entry_type, cont, corr, pers,
                 cutoff, gamma, corr_thres, n_modes, simplex):

        '''

        '''

        print('Initializing dataset...')

        if not any((cont, corr)):
            raise ValueError('At least one pipeline must be turned on.')
        else:
            print('  Contact pipeline is active.') if cont else None
            print('  Correlation pipeline is active.') if corr else None
            print('  Persistence homology pipeline is active.'
                  '') if pers else None

        self.set_name = set_name
        self.go_thres = go_thres
        self.entry_type = entry_type

        ################################################################
        # folder names containing data of the specified setup
        ################################################################
        self.nma_setup = pp.nma_setup_folder_name.format(cutoff,
                                                         gamma,
                                                         corr_thres,
                                                         n_modes)
        self.pi_setup = pp.pi_setup_folder_name.format(simplex)
        self.go_thres_setup = pp.go_thres_folder_name.format(self.go_thres)

        self.contCorr_setup = f'cont_{cont}-corr_{corr}'
        self.folder_name = (f'{self.nma_setup}-{self.pi_setup}-'
                                f'{self.contCorr_setup}')

        self.raw_graph_dir = path.join(pp.df_graph_root, self.nma_setup)
        self.raw_pi_dir = path.join(pp.df_pi_root, self.pi_setup)

        self.stats_root = path.join(pp.df_stats_root, set_name)

        self.stats_dir = path.join(self.stats_root,
                                   f'{self.nma_setup}-{self.pi_setup}',
                                   self.go_thres_setup)

        ################################################################
        # save all dataset parameters
        ################################################################
        self.cont = cont
        self.corr = corr
        self.pers = pers

        self.cutoff = cutoff
        self.gamma = gamma
        self.corr_thres = corr_thres
        self.n_modes = n_modes

        self.simplex = simplex

        ################################################################
        # get list of IDs for this dataset
        ################################################################
        file = path.join(self.stats_dir, pp.df_mfgo_filename)

        # file exists if the preprocessor was executed on this dataset
        if path.exists(file):
            self.mfgo_file = file
            self.n_GO_terms = np.unique([e for v in self.mfgo_dict.values()
                                           for e in v]).size
        # run preprocessor if it was not executed before
        else:
            self.download()

        ################################################################
        # Call constuctor of parent class
        ################################################################
        transform = remove_PI if not pers else None

        super().__init__(self.raw_graph_dir, transform, None, None)
        print('Dataset Initialization Complete\n')

    @property
    def mfgo_dict(self):
        # print(self.mfgo_file)
        with open(self.mfgo_file, 'r') as f_in:
            mfgo_dict = json.load(f_in)
        return mfgo_dict

    @property
    def id_list(self):
        return [ID for ID in self.mfgo_dict]

    @property
    def pos_weight(self):
        mfgo_list = [e for v in self.mfgo_dict.values() for e in v]
        unique, count = np.unique(mfgo_list,
                                  return_counts=True)
        # pos_weights = num of negative samples / true samples
        pos_weight = ( len(self.mfgo_dict)-count ) / count
        return torch.from_numpy(pos_weight)

    @property
    def raw_dir(self):
        return self.raw_graph_dir

    @property
    def processed_dir(self):
        return path.join(df_processed_root, self.folder_name)

    @property
    def raw_file_names(self):
        return [f'{ID}.json' for ID in self.id_list]

    @property
    def processed_file_names(self):
        return [f'{ID}.pt' for ID in self.id_list]

    def download(self):
        '''
        Run preprocessor processes to generate raw data.
        Uses existing label file if exists.
        '''

        label_dir = path.join(self.stats_root, 'labels')

        process = pp.Preprocessor(set_name=self.set_name,
                                  go_thres=self.go_thres,
                                  entry_type=self.entry_type)

        if not path.exists(path.join(label_dir, pp.df_mfgo_filename)):
            process.gen_labels(retry_download=False,
                               redownload=False,
                               verbose=True)

        process.preprocess(simplex=self.simplex,
                           cutoff=self.cutoff,
                           gamma=self.gamma,
                           corr_thres=self.corr_thres,
                           n_modes=self.n_modes,
                           retry_download=False,
                           rebuild_pi=False,
                           rebuild_graph=False,
                           update_mfgo=True,
                           verbose=True)
        self.mfgo_file = path.join(self.stats_dir, pp.df_mfgo_filename)
        self.n_GO_terms = np.unique([e for v in self.mfgo_dict.values()
                                       for e in v]).size

    def process(self):
        for idx, ID in enumerate(tqdm(self.id_list,
                                      desc='  Processing data (PyG)',
                                      ascii=True, dynamic_ncols=True)):
            with open(path.join(self.raw_graph_dir, ID+'.json'), 'r') as fin:
                js_graph = json.load(fin)

            nx_graph = nx.readwrite.json_graph.node_link_graph(js_graph)
            data = pyg.utils.from_networkx(nx_graph)

            # delete correlation edge to turn off the corr pipeline
            if not self.corr:
                indices = torch.argwhere(~(data.weight==2)).squeeze()
                data.edge_index = torch.index_select(data.edge_index,
                                                     1, indices)
            # delete contact edge to turn off the contact pipeline
            if not self.cont:
                indices = torch.argwhere(~(data.weight==1)).squeeze()
                data.edge_index = torch.index_select(data.edge_index,
                                                     1, indices)

            ############################################################
            # residue type
            ############################################################

            n_unique_residues = np.unique(list(res_dict.values()))
            x = np.zeros((len(data.resname), len(n_unique_residues)),
                         dtype=np.int_)

            for j, residue in enumerate(data.resname):
                if residue not in res_dict:
                    residue = 'XAA'
                x[j, res_dict[residue]] = 1

            data.x = torch.from_numpy(x)

            pi = np.load(path.join(self.raw_pi_dir, ID+'.npy'))
            data.pi = torch.from_numpy(pi).float()

            ############################################################
            # labels
            ############################################################

            # y = np.zeros((1, self.n_GO_terms), dtype=np.int_)
            # y[0, self.mfgo_dict[ID]] = 1

            # data.y = torch.from_numpy(y)
            data.ID = ID

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            # remove unnecessary objects

            del data.resname, data.weight

            processed_filename = ID+'.pt'
            torch.save(data, path.join(self.processed_dir, processed_filename))

    def len(self):
        return len(self.mfgo_dict)

    def get(self, idx):
        return torch.load(path.join(self.processed_dir,
                                    self.processed_file_names[idx]))

class Contact8A(ProDAR_Dataset):

    def __init__(self, set_name, go_thres, entry_type):

        cont, corr, pers = True, False, False
        cutoff = 8

        super().__init__(set_name=set_name, go_thres=go_thres,
                         entry_type=entry_type,
                         cont=cont, corr=corr, pers=pers,
                         cutoff=cutoff, gamma=pp.df_gamma,
                         corr_thres=pp.df_corr_thres,
                         n_modes=pp.df_n_modes, simplex=pp.df_simplex)

    @property
    def mfgo_dict(self):
        return super().mfgo_dict

    @property
    def id_list(self):
        return super().id_list

    @property
    def pos_weight(self):
        return super().pos_weight

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

    def __init__(self, set_name, go_thres, entry_type):

        cont, corr, pers = True, True, True
        cutoff = 8

        super().__init__(set_name=set_name, go_thres=go_thres,
                         entry_type=entry_type,
                         cont=cont, corr=corr, pers=pers,
                         cutoff=cutoff, gamma=pp.df_gamma,
                         corr_thres=pp.df_corr_thres,
                         n_modes=pp.df_n_modes, simplex=pp.df_simplex)

    @property
    def mfgo_dict(self):
        return super().mfgo_dict

    @property
    def id_list(self):
        return super().id_list

    @property
    def pos_weight(self):
        return super().pos_weight

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

    def __init__(self, set_name, go_thres, entry_type):

        cont, corr, pers = True, True, True
        cutoff = 12

        super().__init__(set_name=set_name, go_thres=go_thres,
                         entry_type=entry_type,
                         cont=cont, corr=corr, pers=pers,
                         cutoff=cutoff, gamma=pp.df_gamma,
                         corr_thres=pp.df_corr_thres,
                         n_modes=pp.df_n_modes, simplex=pp.df_simplex)

    @property
    def mfgo_dict(self):
        return super().mfgo_dict

    @property
    def id_list(self):
        return super().id_list

    @property
    def pos_weight(self):
        return super().pos_weight

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

class AlternativeDataset(pyg.data.Dataset):

    def __init__(self, set_name, entry_type, cont, corr, pers,
                 cutoff, gamma, corr_thres, n_modes, simplex):

        '''

        '''

        if path.exists(path.join(self_dir, 'alt_data_rerun')):
            from .alt_data_rerun.preprocessing import (
                preprocessor as altpp
            )
        else:
            raise NotImplementedError('No alternative datasets found.')

        print('Initializing dataset...')

        if not any((cont, corr)):
            raise ValueError('At least one pipeline must be turned on.')
        else:
            print('  Contact pipeline is active.') if cont else None
            print('  Correlation pipeline is active.') if corr else None
            print('  Persistence homology pipeline is active.'
                  '') if pers else None

        self.set_name = set_name
        # self.go_thres = go_thres
        self.entry_type = entry_type

        ################################################################
        # folder names containing data of the specified setup
        ################################################################
        self.nma_setup = altpp.nma_setup_folder_name.format(cutoff,
                                                         gamma,
                                                         corr_thres,
                                                         n_modes)
        self.pi_setup = altpp.pi_setup_folder_name.format(simplex)
        self.contCorr_setup = f'cont_{cont}-corr_{corr}'
        self.folder_name = (f'{self.nma_setup}-{self.pi_setup}-'
                                f'{self.contCorr_setup}')

        self.raw_graph_dir = path.join(altpp.df_graph_root, self.nma_setup)
        self.raw_pi_dir = path.join(altpp.df_pi_root, self.pi_setup)

        self.stats_root = path.join(altpp.df_stats_root, set_name)

        self.stats_dir = path.join(self.stats_root,
                                   f'{self.nma_setup}-{self.pi_setup}')

        ################################################################
        # save all dataset parameters
        ################################################################
        self.cont = cont
        self.corr = corr
        self.pers = pers

        self.cutoff = cutoff
        self.gamma = gamma
        self.corr_thres = corr_thres
        self.n_modes = n_modes

        self.simplex = simplex

        ################################################################
        # get list of IDs for this dataset
        ################################################################
        file = path.join(self.stats_dir, altpp.df_mfgo_filename)

        # file exists if the preprocessor was executed on this dataset
        if path.exists(file):
            self.mfgo_file = file
            self.n_GO_terms = np.unique([e for v in self.mfgo_dict.values()
                                           for e in v]).size
        # run preprocessor if it was not executed before
        else:
            self.download()

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
    def pos_weight(self):
        mfgo_list = [e for v in self.mfgo_dict.values() for e in v]
        unique, count, indices = np.unique(mfgo_list,
                                           return_inverse=True,
                                           return_counts=True)
        pos_weight = ( len(self.mfgo_dict)-count[indices] ) / count[indices]
        print(count[indices])
        return torch.from_numpy(pos_weight)

    @property
    def raw_dir(self):
        return self.raw_graph_dir

    @property
    def processed_dir(self):
        return path.join(df_processed_root, self.folder_name)

    @property
    def raw_file_names(self):
        return [f'{ID}.json' for ID in self.id_list]

    @property
    def processed_file_names(self):
        return [f'{ID}.pt' for ID in self.id_list]

    def download(self):
        pass
        '''
        Run preprocessor processes to generate raw data.
        Uses existing label file if exists.
        '''

        # label_dir = path.join(self.stats_root, 'labels')

        # process = altpp.Preprocessor(set_name=self.set_name,
        #                           entry_type=self.entry_type)

        # if not path.exists(path.join(label_dir, altpp.df_mfgo_filename)):
        #     process.gen_labels(threshold=0,
        #                        retry_download=False,
        #                        redownload=False,
        #                        verbose=True)

        # process.preprocess(simplex=self.simplex,
        #                    cutoff=self.cutoff,
        #                    gamma=self.gamma,
        #                    corr_thres=self.corr_thres,
        #                    n_modes=self.n_modes,
        #                    retry_download=False,
        #                    rebuild_pi=False,
        #                    rebuild_graph=False,
        #                    update_mfgo=True,
        #                    verbose=True)
        # self.mfgo_file = path.join(self.stats_dir, altpp.df_mfgo_filename)
        # self.n_GO_terms = np.unique([e for v in self.mfgo_dict.values()
        #                                for e in v]).size

    def process(self):
        for idx, ID in enumerate(tqdm(self.id_list,
                                      desc='  Processing data (PyG)',
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
            x = np.zeros((len(data.resname), len(n_unique_residues)),
                         dtype=np.int_)

            for j, residue in enumerate(data.resname):
                if residue not in res_dict:
                    residue = 'XAA'
                x[j, res_dict[residue]] = 1

            data.x = torch.from_numpy(x)

            pi = np.load(path.join(self.raw_pi_dir, ID+'.npy'))
            data.pi = torch.from_numpy(pi).float()

            ############################################################
            # labels
            ############################################################

            # y = np.zeros((1, self.n_GO_terms), dtype=np.int_)
            # y[0, self.mfgo_dict[ID]] = 1

            # data.y = torch.from_numpy(y)
            data.ID = ID

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            # remove unnecessary objects

            del data.resname, data.weight

            processed_filename = ID+'.pt'
            torch.save(data, path.join(self.processed_dir, processed_filename))

    def len(self):
        return len(self.mfgo_dict)

    def get(self, idx):
        return torch.load(path.join(self.processed_dir,
                                    self.processed_file_names[idx]))
