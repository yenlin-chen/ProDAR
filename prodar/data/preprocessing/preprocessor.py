#!/usr/bin/env python
# coding: utf-8

from . import utils

import json
import requests
from os import (
    path,
    makedirs,
    chdir,
    getcwd,
    cpu_count,
    remove as remove_file
)
from tqdm import tqdm
from datetime import datetime
import numpy as np
import prody
import signal
from math import sqrt
import gudhi as gd
import gudhi.representations
import networkx as nx

self_dir = path.dirname(path.realpath(__file__))

########################################################################
# GLOBAL VARIABLES (USED BY SCIPTS THAT IMPORT THIS MODULE)
########################################################################
df_preprocessed_root = path.join(self_dir, 'preprocessed')
df_graph_root = path.join(df_preprocessed_root, 'json_graphs')
df_pi_root = path.join(df_preprocessed_root, 'persistence_images')
df_rcsb_template = path.join(self_dir, 'rcsb-payload-template')

df_stats_root = path.join(self_dir, 'stats')

df_payload_filename = 'payload.json'
df_payload_template = 'payload-template.json'
# df_entity_filename = 'entity-from_rcsb.txt'
df_id_list_filename = 'id_list-auth_asym_id.txt'
df_mfgo_cnt_filename = 'mfgo-count.txt'
df_mfgo_filename = 'id-mfgo.json'
df_noMFGO_filename = 'id-without_MFGO.txt'
# df_mfgo_cnt_filename = 'mfgo-chain_cnt.csv'
df_chain_filename = 'chain-all.txt'
df_failed_chain_filename = 'chain-failed.txt'
df_pdb_filename = 'pdb-all.txt'
df_failed_pdb_filename = 'pdb-failed.txt'

nma_setup_folder_name = 'cutoff_{}A-gamma_{}-thres_{}-nModes_{}'
pi_setup_folder_name = 'simplex_{}'
go_thres_folder_name = 'GOthres_{}'

# makedirs(df_graph_root, exist_ok=True)
# makedirs(df_pi_root, exist_ok=True)

# default parameters
df_atomselect = 'calpha'

df_simplex = 'alpha'
df_pi_range = [0, 50, 0, 50*sqrt(2)/2]
df_pi_size = [25, 25]

df_cutoff = 8
df_gamma = 1
df_corr_thres = 0.5
df_n_modes = 20

class Preprocessor():

    def __init__(self, set_name, entry_type, go_thres=0,
                 atomselect=df_atomselect, verbose=True):

        '''
        Directory manager for data preprocesing. Also provides the
        functions required to build ProDAR datasets.

        Caches and logs are maintained to save significant time on
        subsequent runs.
        '''

        print('\nInstantiating preprocessor...', end='')

        # name of protein selection, e.g. original_7k
        self.set_name = set_name
        self.entry_type = entry_type # chain or pdb
        self.atomselect = atomselect
        self.go_thres = go_thres

        ################################################################
        # save and cache location
        ################################################################
        self.stats_root = path.join(df_stats_root, set_name)
        self.label_dir = path.join(self.stats_root, 'labels',
                                   go_thres_folder_name.format(go_thres))
        self.target_dir = path.join(self.stats_root, 'target')
        self.rcsb_dir = path.join(self.stats_root, 'rcsb-search-api')

        self.mfgo_cache_dir = path.join(self_dir, 'cache', 'mfgo')
        self.cif_cache_dir = path.join(self_dir, 'cache', 'cif')
        self.pdb_cache_dir = path.join(self_dir, 'cache', 'pdb')

        ################################################################
        # filename of log read by  all datasets
        ################################################################
        self.log_dir = path.join(self_dir, 'log')
        self.mfgo_log = path.join(self.log_dir,
                                  'pdb-failed_to_fetch_mfgo.log')
        self.struct_log = path.join(self.log_dir,
                                    f'{entry_type}-download_failed.log')

        ################################################################
        # filename for dataset-specific log
        ################################################################
        self.label_logname = 'id-failed_to_label.log'
        self.process_logname = 'id-failed_to_process.log'
        self.entity2chain_logname = 'id-failed_to_convert_entity2chain.log'

        self.verbose = verbose

        ################################################################
        # create directories
        ################################################################
        makedirs(self.stats_root, exist_ok=True)
        makedirs(self.label_dir, exist_ok=True)
        # makedirs(self.target_dir, exist_ok=True)
        makedirs(self.mfgo_cache_dir, exist_ok=True)
        makedirs(self.cif_cache_dir, exist_ok=True)
        makedirs(self.pdb_cache_dir, exist_ok=True)
        makedirs(self.log_dir, exist_ok=True)

        ################################################################
        # set up ProDy
        ################################################################
        # print()
        # prody.confProDy(verbosity='info' if verbose else 'none')
        prody.confProDy(verbosity='none')
        prody.pathPDBFolder(folder=self.pdb_cache_dir, divided=False)

        print('Done', end='\n\n')

        self.go_url = 'https://www.ebi.ac.uk/pdbe/api/mappings/go/'

    def _get_mfgo_for_pdb(self, ID, redownload=False,
                          request_timeout=10, verbose=None):

        '''
        Retrieves the MF-GO annotation for the given PDB entry.
        Returns the annotations as a dictionary, along with a string
        explaining the reason of success or failure of the process.
        '''

        if utils.check_id_type(ID) != 'pdb':
            raise ValueError('The list of IDs must be PDB IDs, not '
                             f'{utils.check_id_type(ID)}')

        verbose = self.verbose if verbose is None else verbose
        utils.vprint(verbose, f'Retrieving MF-GO for \'{ID}\'...',
                     end='', flush=True)

        filename = utils.id_to_filename(ID)
        mfgo_cache = path.join(self.mfgo_cache_dir, f'{filename}.json')

        redownload_msg = False
        if path.exists(mfgo_cache):
            if redownload:
                remove_file(mfgo_cache)
                # change the output for rebuilt entries
                redownload_msg = True
            else:
                msg = 'Annotations found on disk'
                with open(mfgo_cache, 'r') as f_in:
                    mfgo = json.load(f_in)
                utils.vprint(verbose, msg)
                return mfgo, msg

        try:
            data = requests.get(self.go_url+ID, timeout=request_timeout)
        except requests.Timeout:
            msg = 'GET request timeout'
            utils.vprint(verbose, msg)
            return None, msg

        # failure on the server side
        if data.status_code != 200:
            msg = f'GET request failed with code {data.status_code}'
            utils.vprint(verbose, msg)
            return None, msg

        decoded = data.json()
        go_dict = decoded[ID.lower()]['GO']

        mfgo = {}
        for code in go_dict:
            if go_dict[code]['category'] == 'Molecular_function':
                mfgo[code] = {'category': 'Molecular_function',
                              'mappings': go_dict[code]['mappings']}

        with open(mfgo_cache, 'w+') as f_out:
            json.dump(mfgo, f_out,
                      indent=4, separators=(',', ': '), sort_keys=True)

        msg = 'Re-downloaded' if redownload_msg else 'Downloaded'
        utils.vprint(verbose, msg)
        return mfgo, msg

    """
    def _rcsb_query(self, payload):
        '''
        Make query to RCSB using the RCSB Search API to obtain a list of
        PDB-entity IDs with a specified sequence similarity cutoff.
        '''

        # GET request
        url = f"https://search.rcsb.org/rcsbsearch/v2/query?json={payload}"
        data = requests.get(url)

        # decode returned data if request is successful
        if data.status_code != 200:
            err_file = path.join(self.rcsb_dir, 'error.txt')
            with open(err_file, 'w+') as f:
                f.write(f'{data.text}')
                f.flush()
            return None, data.text
        decoded = data.json()

        print(f" -> {decoded['total_count']} proteins received from RCSB")

        # convert decoded data into lists
        polymers = [entry['identifier'] for entry in decoded['result_set']]
        pdbs = np.unique([ID[:4] for ID in polymers])

        # save list to corresponding directory
        np.savetxt(path.join(self.rcsb_dir, 'pdb_entity.txt'),
                   polymers, fmt='%s')
        np.savetxt(path.join(self.rcsb_dir, 'pdb.txt'),
                   pdbs, fmt='%s')

        return polymers, 'Query successful'

    def _entityID_to_chainID(self, pdbID, entityID, retry_download,
                             redownload, verbose):

        # dataest-specific log
        dataset_log = path.join(self.rcsb_dir, self.entity2chain_logname)
        open(dataset_log, 'w+').close() # clear file content

        # backup and clear logfile
        if retry_download and path.exists(self.mfgo_log):
            utils.backup_file(self.mfgo_log)
            # the list of IDs to skip will be empty
            utils.rm_log_entries(self.mfgo_log, id_list)

        # get list of PDBs/chains that should be skipped
        log_content, logged_ids = utils.read_logs(self.mfgo_log)
        logged_ids = [ID for ID in logged_ids]

        ################################################################
        # retrieve MF-GO annotations
        ################################################################
        mfgo_list = [] # holder for list of all MF-GOs
        for ID in tqdm(id_list, unit=' entities',
                       desc='Converting IDs',
                       ascii=True, dynamic_ncols=True):

            ############################################################
            # skip if the PDB/chain failed to download in a previous run
            ############################################################
            if ID in logged_ids:
                # copy entry to dataset-specific log
                idx = logged_ids.index(ID)
                utils.append_to_file(log_content[idx], dataset_log)
                unsuccessful_ids.append(ID)
                tqdm.write(f'  Skipping \'{ID}\'')
                continue

            # if the PDB entry was not skipped
            tqdm.write(f'  Processing \'{ID}\'...')

            ############################################################
            # try to download MF-GO
            ############################################################
            tqdm.write('    Fetching MF-GO...')
            mfgo, msg = self._get_mfgo_for_pdb(ID[:4], redownload=False,
                                               verbose=False)
            tqdm.write(f'        {msg}')
            if mfgo is None:
                utils.append_to_file(f'{ID} -> MF-GO: {msg}', dataset_log)
                utils.append_to_file(f'{ID} -> MF-GO: {msg}', self.mfgo_log)
                unsuccessful_ids.append(ID)
                continue
            else:
                successful_ids.append(ID)





    def gen_set_from_rcsb(self, similarity_cutoff=None, retry_download=False,
                          redownload=False, verbose=None):
        '''
        Make query to RCSB using the RCSB Search API to obtain a list of
        PDB-entity IDs with a specified sequence similarity cutoff.
        The list of entity IDs is then converted to a list of chain IDs.
        Entities without any MF-GO annotations are discarded, hence the
        chains from that entity will not be present in the returned list
        of chains.
        '''

        print('Generating set from query to RCSB...')
        verbose = self.verbose if verbose is None else verbose

        payload_file = path.join(self.rcsb_dir, df_payload_filename)
        chain_file = path.join(self.rcsb_dir, 'pdb_chain.txt')
        id_list_file = path.join(self.target_dir, df_id_list_filename)

        if similarity_cutoff is not None:
            # read and modify template with similarity cutoff
            template_file = path.join(df_rcsb_template, df_payload_template)
            with open(template_file, 'r') as f_in:
                payload = f_in.read().replace('SIMILARITY_PLACEHOLDER',
                                              similarity_cutoff)
            # check if the specified payload file matches the existing
            if path.exists(payload_file):
                with open(payload_file, 'r') as f_in:
                    existing_payload = f_in.read()
                if existing_payload != payload:
                    raise RuntimeError(f'{self.rcsb_dir} must not contain '
                                       f'{df_payload_filename} if '
                                       f'similarity_cutoff is specified')
            else: # save a copy of payload for reference
                with open(payload_file, 'w+') as f_out:
                    f_out.write(payload+'\n')

        # read readied payload file
        elif similarity_cutoff is None:
            with open(payload_file, 'r') as f:
                payload = f.read()

        polymers, msg = self._rcsb_query(payload)

        # find all chains in pdbID-entityID and save to file
        for polymer in enumerate(tqdm(polymers, ascii=True)):
            pdbID, entityID = polymer.split('_')
            chainIDs = self._entityID_to_chainID(pdbID, int(entityID),
                                                 retry_download,
                                                 redownload,
                                                 verbose)
            # save a copy where only one chain from each PDB is selected
            utils.append_to_file(id_list_file, f'{pdbID}-{chainIDs[0]}')
            # save a list of all chains in the entity
            for chainID in chainIDs:
                utils.append_to_file(chain_file, f'{pdbID}-{chainID}')

        with open(id_list_file, 'r') as f_in:
            id_list = f_in.read()

        return id_list
    """

    def gen_labels(self, id_list=None, retry_download=False,
                   redownload=False, verbose=None):

        '''
        1. Retrieves MF-GO annotations for all ID to retrieve list of GO
           categories
        2. Generates the labels for all IDs base on the list of GO
           entries (the dimension of the label is equal to the length of
           the list)
        '''

        id_list_file = path.join(self.target_dir, df_id_list_filename)
        if not id_list:
            id_list = np.loadtxt(id_list_file, dtype=np.str_)
            print('gen_labels() called without specifying list of IDs')
            print(f'Processing {id_list_file}')
        else:
            if path.exists(id_list_file):
                raise RuntimeError(f'{self.label_dir} must not contain '
                                   f'{df_id_list_filename} if id_list is '
                                   f'specified')
        print('Generating labels...')
        verbose = self.verbose if verbose is None else verbose

        # holder for PBDs/chains that are successfully preprocessed
        successful_ids = []
        unsuccessful_ids = []

        if utils.check_id_type(id_list) != 'chain':
            raise ValueError('The list of IDs must be chains, not '
                             f'{utils.check_id_type(id_list)}')

        # dataset-specific log
        dataset_log = path.join(self.label_dir, self.label_logname)
        open(dataset_log, 'w+').close() # clear file content

        # backup and clear logfile
        if retry_download and path.exists(self.mfgo_log):
            utils.backup_file(self.mfgo_log)
            # the list of IDs to skip will be empty
            utils.rm_log_entries(self.mfgo_log, id_list)

        # get list of PDBs/chains that should be skipped
        log_content, logged_ids = utils.read_logs(self.mfgo_log)
        logged_ids = [ID for ID in logged_ids]
        unit_str = 'chains' if self.entry_type=='chain' else 'PDB entries'
        utils.vprint(verbose, f' -> {len(logged_ids)} {unit_str} found in log')

        ################################################################
        # retrieve MF-GO annotations
        ################################################################
        mfgo_list = [] # holder for list of all MF-GOs
        for ID in tqdm(id_list, unit=f' {unit_str}',
                       desc='Retrieving MF-GO',
                       ascii=True, dynamic_ncols=True):

            ############################################################
            # skip if the PDB/chain failed to download in a previous run
            ############################################################
            if ID in logged_ids:
                # copy entry to dataset-specific log
                idx = logged_ids.index(ID)
                utils.append_to_file(log_content[idx], dataset_log)
                unsuccessful_ids.append(ID)
                tqdm.write(f'  Skipping \'{ID}\'')
                continue

            # if the PDB entry was not skipped
            tqdm.write(f'  Processing \'{ID}\'...')

            ############################################################
            # try to download MF-GO
            ############################################################
            tqdm.write('    Fetching MF-GO...')
            mfgo, msg = self._get_mfgo_for_pdb(ID[:4], redownload=False,
                                               verbose=False)
            tqdm.write(f'        {msg}')
            if mfgo is None:
                utils.append_to_file(f'{ID} -> MF-GO: {msg}', dataset_log)
                utils.append_to_file(f'{ID} -> MF-GO: {msg}', self.mfgo_log)
                unsuccessful_ids.append(ID)
                continue
            else:
                successful_ids.append(ID)

            ############################################################
            # get a list of all mfgo codes
            ############################################################
            for code in mfgo:
                for m in mfgo[code]['mappings']:
                    if m['chain_id'] == ID[5:]:
                        mfgo_list.append(code)
                        break # at most one entry of the chain per MF-GO

        # get list of unique items
        mfgo_unique, mfgo_cnt = np.unique(mfgo_list, return_counts=True)

        # discard annotations with too few entries
        if self.go_thres:
            mask = mfgo_cnt >= self.go_thres
            if not np.any(mask):
                raise RuntimeError(f'No MF-GOs with over '
                                   f'{self.go_thres} entries')
            # mfgo_thres = mfgo_unique[~mask]
            mfgo_unique = mfgo_unique[mask]
            mfgo_cnt = mfgo_cnt[mask]

        # save info to drive
        np.savetxt(path.join(self.label_dir, df_mfgo_cnt_filename),
                   np.column_stack((mfgo_unique, mfgo_cnt)), fmt='%s %s')

        ################################################################
        # generate labels for dataset
        ################################################################
        id_mfgo = {}
        for ID in tqdm(successful_ids, unit=f' {unit_str}',
                       desc='Generating labels',
                       ascii=True, dynamic_ncols=True):

            # skip ID if MF-GO was not downloaded
            if ID in unsuccessful_ids:
                continue
            else:
                mfgo, _ = self._get_mfgo_for_pdb(ID[:4], redownload=False,
                                                 verbose=False)

            labels = []
            for code in mfgo:
                for m in mfgo[code]['mappings']:
                    if m['chain_id'] == ID[5:]:
                        # get index of the code
                        loc = np.argwhere(mfgo_unique==code)
                        # if there is no match (if num < self.go_thres)
                        if loc.size == 0:
                            continue
                        # append index to list if there is a match
                        else:
                            labels.append(int(loc[0,0]))

            id_mfgo[utils.id_to_filename(ID)] = labels

        # write labels to drive
        mfgo_file = path.join(self.label_dir, df_mfgo_filename)
        with open(mfgo_file, 'w+') as f_out:
            json.dump(id_mfgo, f_out,
                      indent=4, separators=(',', ': '), sort_keys=True)

        return id_mfgo

    def _get_struct(self, ID, verbose=None):

        '''
        Retrieves protein structure through ProDy. Returns the structure
        as a ProDy 'atoms' type, the correct pdbID-chainID (uppercase,
        lowercase, etc.), and a string explaining the reason of success
        or failure of the process.
        '''

        verbose = self.verbose if verbose is None else verbose
        utils.vprint(verbose, f'Retrieving protein structure for \'{ID}\'...',
                     end='', flush=True)

        # switch working directory to prevent ProDy from polluting
        # current working dir with .cif files
        cwd = getcwd()
        chdir(self.cif_cache_dir)

        # if not try_CIF:
        #     # try to fetch PDB
        #     cache = prody.fetchPDB(ID[:4])
        #     # try to fetch mmCIF if PDB cannot be downloaded
        #     if cache is None:
        #         # try:
        #         #     prody.parseMMCIF(ID[:4], subset='calpha')
        #         # except:
        #         chdir(cwd)
        #         msg = 'Cannot download structure'
        #         utils.vprint(verbose, msg)
        #         return None, msg

        # return data for all chains in PDB entry if no chains were
        # specified
        if len(ID) == 4:
            atoms = prody.parsePDB(ID, subset=self.atomselect)
            chdir(cwd) # switch working directory back ASAP

            # if parse was successful
            if atoms is not None:
                msg = 'Structure downloaded/parsed'
                utils.vprint(verbose, msg)
                return atoms, msg
            else: # parse failed
                msg = 'Cannot download PDB structure'
                utils.vprint(verbose, msg)
                return None, msg

        # if chain ID was specified
        else:
            pdbID, chainID = ID[:4], ID[5:]

            try:
                atoms = prody.parsePDB(pdbID,
                                       subset=self.atomselect,
                                       chain=chainID)
            except UnicodeDecodeError as errMsg:
                chdir(cwd)
                utils.vprint(verbose, errMsg)
                return None, errMsg

            # try to parse chain if structure is retrieved
            if atoms is not None:
                chdir(cwd)

                if atoms[chainID] is not None:
                    msg = 'Structure downloaded/parsed'
                    utils.vprint(verbose, 'Done')
                    return atoms[chainID].copy(), msg

                # elif atoms[chainID[:2]] is not None:
                #     msg = 'Structure downloaded/parsed with modified chain ID'
                #     utils.vprint(verbose, 'Done')
                #     return atoms[chainID[:2]].copy(), msg

                else:
                    msg = f'ProDy cannot resolve chain ID {chainID}'
                    utils.vprint(verbose, msg)
                    return None, msg

            else: # ProDy cannot download structure
                chdir(cwd)
                msg = 'ProDy cannot download structure'
                utils.vprint(verbose, msg)
                return None, msg

    def _get_PI(self, ID, simplex,
                img_range=df_pi_range, img_size=df_pi_size,
                coords=None, rebuild_existing=False, verbose=None):

        '''
        Returns the persistence image for the specified pdb entry or
        chain.

        The function checks if the persistence image for the specified
        ID is saved on disk, and returns the data if it is found.
        Otherwise, the structure for the ID will be retrieved by
        _get_struct() if the coords is not give, and the newly computed
        product is saved on disk.
        '''

        verbose = self.verbose if verbose is None else verbose
        utils.vprint(verbose, f'Retrieving persistence image for \'{ID}\'...',
                     end='', flush=True)

        filename = utils.id_to_filename(ID)
        save_dir = path.join(df_pi_root, pi_setup_folder_name.format(simplex))
        makedirs(save_dir, exist_ok=True)
        pi_file = path.join(save_dir, f'{filename}.npy')

        rebuild_msg = False
        if path.exists(pi_file):
            if rebuild_existing:
                remove_file(pi_file)
                # change the output for rebuilt entries
                rebuild_msg = True
            else:
                msg = 'Data found on disk'
                pers_img = np.load(pi_file)
                utils.vprint(verbose, msg)
                return pers_img, msg

        # try computing if file not found on disk
        if not coords: # use coords if given to save computation
            atoms, msg = self._get_struct(ID, verbose=False)
            if atoms is None:
                utils.vprint(verbose, msg)
                return None, msg
            else:
                coords = atoms.getCoords().tolist()

        # simplicial  complex
        if simplex == 'alpha':
            scx = gd.AlphaComplex(points=coords).create_simplex_tree()
        elif simplex == 'rips':
            distMtx = sp.spatial.distance_matrix(coords, coords, p=2)
            scx = gd.RipsComplex(distance_matrix=distMtx).create_simplex_tree()

        # persistence image
        pi = gd.representations.PersistenceImage(
            bandwidth=1,
            weight=lambda x: max(0, x[1]*x[1]),
            im_range=img_range,
            resolution=img_size
        )

        scx.persistence()

        pInterval_d1 = scx.persistence_intervals_in_dimension(1)
        pInterval_d2 = scx.persistence_intervals_in_dimension(2)

        if pInterval_d1.size!=0 and pInterval_d2.size!=0:
            pers_img = pi.fit_transform([np.vstack((pInterval_d1,
                                                    pInterval_d2))])
        elif pInterval_d1.size!=0 and pInterval_d2.size==0:
            pers_img = pi.fit_transform([pInterval_d1])
        elif pInterval_d1.size==0 and pInterval_d2.size!=0:
            pers_img = pi.fit_transform([pInterval_d2])
        else:
            # discard PDB entry if size is 0 in both dimensions
            msg = 'Persistence interval in both dimensions are 0'
            utils.vprint(verbose, msg)
            return None, msg

        # if computation is successful
        msg = 'Rebuilt' if rebuild_msg else 'Computed'
        np.save(pi_file, pers_img)
        utils.vprint(verbose, msg)
        return pers_img, msg

    def comp_freqs(self, ID, atoms=None,
                   cutoff=df_cutoff, gamma=df_gamma,
                   n_modes=df_n_modes, nCPUs=cpu_count(), verbose=None):

        '''
        Computes and returns first few modes with ProDy for the
        specified PDB entry or chain, depending on whether a chain ID is
        specified in the argument 'ID'.
        '''

        verbose = self.verbose if verbose is None else verbose
        utils.vprint(verbose, f'Computing modes for \'{ID}\'...',
                     end='', flush=True)

        if not atoms: # use coords if given to save computation
            atoms, msg = self._get_struct(ID, verbose=False)
            if atoms is None:
                utils.vprint(verbose, msg)
                return None, None, msg

        anm = prody.ANM(name=ID)
        anm.buildHessian(atoms, cutoff=cutoff, gamma=gamma,
                         norm=True, n_cpu=nCPUs)

        # modal analysis
        try:
            anm.calcModes(n_modes=n_modes, zeros=False, turbo=True)
        except Exception as err:
            # discard PDB entry if normal mode analysis fails
            msg = 'Unable to compute modes.'
            utils.vprint(verbose, msg)
            return None, None, msg

        freqs = [sqrt(mode.getEigval()) for mode in anm]

        return anm, freqs, 'Computed'

    def _get_graph(self, ID,
                   cutoff, gamma, corr_thres, n_modes,
                   nCPUs, atoms=None, rebuild_existing=False, verbose=None):
        '''
        Returns the contact edges and the coorelation edges from the
        results of NMA.

        Checks if the graph for the specified ID is already saved on
        disk, and returns the data if found. If not found, structure for
        the PDB entry or chain will be downloaded using _get_struct(),
        unless 'atoms' is specified. The arguments will be directed to
        comp_freqs() for modal analysis with ProDy.
        '''

        verbose = self.verbose if verbose is None else verbose
        utils.vprint(verbose, f'Retrieving graphs for \'{ID}\'...',
                     end='', flush=True)

        # define save directory for graphs
        save_dir = path.join(df_graph_root,
                             nma_setup_folder_name.format(cutoff, gamma,
                                                          corr_thres, n_modes))
        makedirs(save_dir, exist_ok=True)
        filename = utils.id_to_filename(ID)
        graph_file = path.join(save_dir, f'{filename}.json')

        # returns data found on disk if rebuild is not required
        rebuild_msg = False
        if path.exists(graph_file):
            if rebuild_existing:
                remove_file(graph_file)
                # change output message if entry is rebuilt
                rebuild_msg = True
            else:
                msg = 'Data found on disk'
                with open(graph_file, 'r') as f_in:
                    graph_dict = json.load(f_in)
                utils.vprint(verbose, msg)
                return graph_dict, msg

        # if file was not found
        utils.vprint(verbose)
        if not atoms: # use coords if given to save computation
            atoms, msg = self._get_struct(ID, verbose=False)
            if atoms is None:
                utils.vprint(verbose, msg)
                return None, msg

        utils.vprint(verbose, '  Computing modes...', end='', flush=True)
        anm, freqs, msg = self.comp_freqs(ID, atoms=atoms, cutoff=cutoff,
                                          gamma=gamma,
                                          n_modes=n_modes, nCPUs=nCPUs,
                                          verbose=False)
        if not anm:
            msg = 'Unable to compute modes.'
            utils.vprint(verbose, msg)
            return None, msg

        # compute contact map
        utils.vprint(verbose, 'Kirchhoff...', end='', flush=True)
        cont = - anm.getKirchhoff().astype(np.int_) # not contact map
        np.fill_diagonal(cont, 1) # contact map is completed here

        # compute correlation map
        utils.vprint(verbose, 'Cross Correlation...')
        corr = prody.calcCrossCorr(anm)
        mask = np.abs(corr) > corr_thres
        corr = np.where(mask, 1, 0) # correlation map is completed here

        # compute adjacency matrix
        diff = np.where(np.logical_and(corr==1, cont==0), -1, 0)
        comb = cont + diff

        # create Networkx graph object
        utils.vprint(verbose, '    Building graph ', end='', flush=True)
        utils.vprint(verbose, 'Edges...', end='', flush=True)
        graph = nx.from_numpy_array(np.abs(comb))

        graph.graph['pdbID'] = ID[:4]
        if len(ID) > 4:
            graph.graph['chainID'] = ID[5:]

        # define node attributes
        utils.vprint(verbose, 'Node Attributes...', end='', flush=True)
        resnames = atoms.getResnames()
        attrs = {i: {'resname': r} for i, r in enumerate(resnames)}

        nx.set_node_attributes(graph, attrs)

        # define edge attributes
        utils.vprint(verbose, 'Edge Attributes...', end='', flush=True)
        for nodeI, nodeJ in graph.edges:
            if comb[nodeI][nodeJ] == 1: # contact edge
                graph.edges[(nodeI, nodeJ)]['weight'] = 1
            elif comb[nodeI][nodeJ] == -1: # correlation edge
                graph.edges[(nodeI, nodeJ)]['weight'] = -1

        # map serial ID to residue ID
        mapping = dict(zip(graph, atoms.getResnums().tolist()))
        graph = nx.relabel.relabel_nodes(graph, mapping)

        graph_dict = nx.readwrite.json_graph.node_link_data(graph)

        with open(graph_file, 'w+') as f_out:
            json.dump(graph_dict, f_out,
                      indent=4, separators=(',', ': '), sort_keys=True)

        # freqs = [sqrt(mode.getEigval()) for mode in anm]
        # cont_nEdges, corr_nEdges = [int(np.sum(cont)), int(-np.sum(diff))]

        msg = 'Rebuilt' if rebuild_msg else 'Computed'
        utils.vprint(verbose, msg)
        return graph_dict, msg

    def _update_MFGO_indices(self, successful_ids, save_dir, verbose=True):

        '''
        Updates the list of ID-MFGO saved in preprocessing/stats/target,
        and saves a new copy to preprocessing/stats. The new copy can be
        used for GNN training.
        '''

        if utils.check_id_type(successful_ids) != self.entry_type:
            raise ValueError('The list of IDs did not match specified type '
                             f'{self.entry_type}')

        utils.vprint(verbose, 'Updating MFGO indices...', end='')
        with open(path.join(self.label_dir, df_mfgo_filename),
                  'r') as f_in:
            id_mfgo = json.load(f_in)

        # fish out all entries in successful_ids
        try:
            new_id_mfgo = { utils.id_to_filename(ID): id_mfgo[ID]
                            for ID in successful_ids }
        except KeyError as err:
            print(f'{err} was not found in '
                  f'{path.join(self.label_dir, df_mfgo_filename)}')
            utils.vprint(verbose, 'MFGO indices will not be updated')
            return None

        mfgo_list = [e for v in new_id_mfgo.values()
                       for e in v]
        new_mfgo_unique, new_mfgo_cnt = np.unique(mfgo_list,
                                                  return_counts=True)

        # discard annotations with too few entries
        if self.go_thres:
            mask = new_mfgo_cnt >= self.go_thres
            if not np.any(mask):
                raise RuntimeError(f'No MF-GOs with over '
                                   f'{self.go_thres} entries')
            new_mfgo_unique = new_mfgo_unique[mask]
            new_mfgo_cnt = new_mfgo_cnt[mask]

        # squeeze the numbering towards 0 (start from 0 continuously)
        for ID in new_id_mfgo:
            labels = []
            for code in new_id_mfgo[ID]:
                loc = np.argwhere(new_mfgo_unique==code)
                if loc.size == 0:
                    continue
                else:
                    labels.append(int(loc[0,0]))
                new_id_mfgo[ID] = labels

        # update mfgo-count file
        cnt_file = path.join(self.label_dir, df_mfgo_cnt_filename)
        if path.exists(cnt_file):
            mfgo_unique = np.loadtxt(cnt_file, dtype=str)[:,0]

            new_cnt = np.column_stack((mfgo_unique[new_mfgo_unique],
                                       new_mfgo_cnt))
        else:
            new_cnt = np.column_stack((np.arange(new_mfgo_unique.size),
                                       new_mfgo_cnt))

        # save count to drive
        np.savetxt(path.join(save_dir, df_mfgo_cnt_filename),
                   new_cnt, fmt='%s %s')

        # check if any labels are present in every data entry
        warning_file = path.join(save_dir, 'warning-mfgo_update.txt')
        if any(new_mfgo_cnt==len(new_id_mfgo)):
            msg = (f'Warning: Labels '
                   f'{new_mfgo_unique[new_mfgo_cnt==len(new_id_mfgo)]} '
                   f'exists in all data entries\n')
            with open(warning_file, 'w+') as f_out:
                f_out.write(msg)
                f_out.flush()
        elif path.exists(warning_file):
            remove_file(warning_file)

        # save id-mfgo
        with open(path.join(save_dir, df_mfgo_filename),
                  'w+') as f_out:
            json.dump(new_id_mfgo, f_out,
                      indent=4, separators=(',', ': '), sort_keys=True)

        utils.vprint(verbose, '')
        ids_without_mfgos = [ID for ID in new_id_mfgo if not new_id_mfgo[ID]]
        np.savetxt(path.join(save_dir, df_noMFGO_filename),
                   ids_without_mfgos, fmt='%s')

        utils.vprint(verbose, 'Done')
        return id_mfgo

    def preprocess(self, id_list=None,
                   # persistence images
                   simplex=df_simplex,
                   # normal mode analysis
                   cutoff=df_cutoff, gamma=df_gamma, corr_thres=df_corr_thres,
                   n_modes=df_n_modes, nCPUs=cpu_count(),
                   retry_download=False, rebuild_pi=False, rebuild_graph=False,
                   update_mfgo=True, verbose=None):

        '''
        Generates all the data needed for training ProDAR.
        '''

        mfgo_file = path.join(self.label_dir, df_mfgo_filename)
        if not id_list:
            with open(mfgo_file, 'r') as f_in:
                id_mfgo = json.load(f_in)
            id_list = [ID for ID in id_mfgo]
            print('preprocess() called without specifying list of IDs')
            print(f'Processing {mfgo_file}')
        else:
            if path.exists(mfgo_file):
                raise RuntimeError(f'{self.label_dir} must not contain '
                                   f'{df_mfgo_filename} if id_list is '
                                   f'specified')

            id_list_file = path.join(self.target_dir, df_id_list_filename)
            if path.exists(id_list_file):
                raise RuntimeError(f'{self.target_dir} must not contain '
                                   f'{df_id_list_filename} if id_list is '
                                   f'specified')

            np.savetxt(id_list_file, id_list, fmt='%s')
            print('Processing data...')
        verbose = self.verbose if verbose is None else verbose

        # holder for PBDs/chains that are successfully preprocessed
        successful_ids = []
        unsuccessful_ids = []

        if utils.check_id_type(id_list) != self.entry_type:
            raise ValueError('The list of IDs did not match specified type '
                             f'{self.entry_type}')

        # save directory
        nma_setup = nma_setup_folder_name.format(cutoff, gamma,
                                                 corr_thres, n_modes)
        pi_setup = pi_setup_folder_name.format(simplex)
        go_thres_setup = go_thres_folder_name.format(self.go_thres)
        save_dir = path.join(self.stats_root,
                             f'{nma_setup}-{pi_setup}',
                             go_thres_setup)
        makedirs(save_dir, exist_ok=True)

        # dataset-specific log
        dataset_log = path.join(save_dir, self.process_logname)
        open(dataset_log, 'w+').close() # clear file content

        # # dataset-sepcific warning
        # warning_file = path.join(save_dir, 'warning-chain_id.txt')
        # open(warning_file, 'w+').close() # clear file content

        # backup and clear logfile
        if retry_download and path.exists(self.struct_log):
            utils.backup_file(self.struct_log)
            # the list of IDs to skip will be empty
            utils.rm_log_entries(self.struct_log, id_list)

        # get list of PDBs/chains that should be skipped
        log_content, logged_ids = utils.read_logs(self.struct_log)
        unit_str = 'chains' if self.entry_type=='chain' else 'PDB entries'
        utils.vprint(verbose, f' -> {len(logged_ids)} {unit_str} found in log')

        for ID in tqdm(id_list, unit=f' {unit_str}',
                       desc='Processing data',
                       ascii=True, dynamic_ncols=True):

            ############################################################
            # skip everything if all data for ID is found on disk
            ############################################################
            if not rebuild_pi and not rebuild_graph:
                graph_file = path.join(df_graph_root, nma_setup,
                                       f'{utils.id_to_filename(ID)}.json')
                pi_file = path.join(df_pi_root, pi_setup,
                                    f'{utils.id_to_filename(ID)}.npy')
                if path.exists(graph_file) and path.exists(pi_file):
                    successful_ids.append(ID)
                    tqdm.write(f'  All data for \'{ID}\' found on disk.')
                    continue

            ############################################################
            # if the PDB/chain failed to download in a previous run
            ############################################################
            if ID in logged_ids:
                # copy entry to dataset-specific log
                idx = logged_ids.index(ID)
                utils.append_to_file(log_content[idx], dataset_log)
                unsuccessful_ids.append(ID)
                tqdm.write(f'  Skipping processing of \'{ID}\'')
                continue

            # if the PDB entry was not skipped
            tqdm.write(f'  Processing \'{ID}\'...')

            ############################################################
            # try to download/parse structure
            ############################################################
            tqdm.write('    Download/Parsing Structure...') #, end='')
            atoms, msg = self._get_struct(ID, verbose=False)
            tqdm.write(f'      {msg}')
            if atoms is None:
                # write entry to dataset-specific log
                utils.append_to_file(f'{ID} -> ProDy: {msg}', dataset_log)
                # write new entry to log for all datasets
                utils.append_to_file(f'{ID} -> ProDy: {msg}', self.struct_log)
                unsuccessful_ids.append(ID)
                continue
            # # check if chain ID is modified for ProDy to work
            # elif msg[-8:] == 'chain ID':
            #     utils.append_to_file(f'{ID} -> ProDy: {msg}', warning_file)
            coords = atoms.getCoords().tolist()

            ############################################################
            # try to generate persistence image
            ############################################################
            tqdm.write('    Persistence Img...') #, end='')
            pers_img, msg = self._get_PI(ID, coords=coords,
                                         simplex=simplex,
                                         rebuild_existing=rebuild_pi,
                                         verbose=False)
            tqdm.write(f'      {msg}')
            if pers_img is None:
                # write entry to dataset-specific log
                utils.append_to_file(f'{ID} -> PI: {msg}', dataset_log)
                unsuccessful_ids.append(ID)
                continue

            ############################################################
            # try to generate graphs (normal mode analysis)
            ############################################################
            tqdm.write('    Graph (NMA)...') #, end='')
            graph_dict, msg = self._get_graph(ID, atoms=atoms,
                                              cutoff=cutoff, gamma=gamma,
                                              corr_thres=corr_thres,
                                              n_modes=n_modes, nCPUs=nCPUs,
                                              rebuild_existing=rebuild_graph,
                                              verbose=False)
            tqdm.write(f'      {msg}')
            if graph_dict is None:
                # write entry to dataset-specific log
                utils.append_to_file(f'{ID} -> NMA: {msg}', dataset_log)
                unsuccessful_ids.append(ID)
                continue

            ############################################################
            # all computations succeeded
            ############################################################
            successful_ids.append(ID)

        print('Saving...') #, end='')
        if self.entry_type == 'chain':
            # save successful ids
            np.savetxt(path.join(save_dir, df_chain_filename),
                       successful_ids, fmt='%s')
            successful_pdb = [e[:4] for e in successful_ids]
            np.savetxt(path.join(save_dir, df_pdb_filename),
                       successful_pdb, fmt='%s')

            # save unsuccessful ids
            np.savetxt(path.join(save_dir,
                                 df_failed_chain_filename),
                       unsuccessful_ids, fmt='%s')
            unsuccessful_pdb = [e[:4] for e in unsuccessful_ids]
            np.savetxt(path.join(save_dir, df_failed_pdb_filename),
                       unsuccessful_pdb, fmt='%s')
        else:
            np.savetxt(path.join(save_dir, df_pdb_filename),
                       successful_ids, fmt='%s')
            np.savetxt(path.join(save_dir, df_failed_pdb_filename),
                       unsuccessful_ids, fmt='%s')

        print('  Done')

        # output summary
        print(f' -> {len(successful_ids)} out of {len(id_list)} '
              f'{unit_str} successfully processed '
              f'({len(unsuccessful_ids)} {unit_str} failed)')

        # update label files
        if update_mfgo:
            print('Updating MFGO label files...') #, end='')
            id_mfgo = self._update_MFGO_indices(successful_ids, save_dir,
                                                verbose=False)
            if id_mfgo:
                print('  Done')
            else:
                print('  Update aborted')

        print('>>> Preprocessing Complete')

    def cleanup(self):

        while True:
            inp = input(f'All preprocessed data not in {self.set_name} '
                        f'will be removed. Proceed? (y/n)')
            if inp.lower() in ['y', 'n']:
                break
