#!/usr/bin/env python
# coding: utf-8

from prodar.data.preprocessing import preprocessor as pp

if __name__ == '__main__':

    # process = Preprocessor(set_name='test', entry_type='chain')

    # process.preprocess(simplex=pp.df_simplex,
    #                    cutoff=8, gamma=pp.df_gamma,
    #                    corr_thres=pp.df_corr_thres, n_modes=pp.df_n_modes,
    #                    retry_download=False,
    #                    rebuild_pi=False, rebuild_graph=False,
    #                    update_mfgo=True, verbose=None)

    # dataset_list = ['original_7k', 'deepfri-all', 'deepfri-test',
    #                 'deepfri-train', 'deepfri-valid', 'deepfri_and_original']
    dataset_list = ['original-rcsb_10k']

    for set_name in dataset_list:

        process = pp.Preprocessor(set_name=set_name,
                                  entry_type='chain',
                                  go_thres=25)

        id_mfgo = process.gen_labels(id_list=None,
                                     retry_download=False, redownload=False,
                                     verbose=None)
        process.preprocess(simplex=pp.df_simplex,
                           cutoff=8, gamma=pp.df_gamma,
                           corr_thres=pp.df_corr_thres, n_modes=pp.df_n_modes,
                           retry_download=False,
                           rebuild_pi=False, rebuild_graph=False,
                           update_mfgo=True, verbose=None)
