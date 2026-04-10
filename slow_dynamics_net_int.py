# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 10:43:09 2026

@author: yuiry shymkiv; ys2605@columbia.edu
"""

# ---- importing functions ----
import sys
import numpy as np

# importing slow dynamics pipeline
pipeline_dir = 'C:/Users/ys2605/Desktop/stuff/slow_dynamics_analysis'    # edit this  
sys.path.append(pipeline_dir + '/functions')
from f_sd_utils import f_load_caim_data_mat, f_load_rnn_test, f_get_values, f_get_network_intrinsic_timescales, f_save_fig, f_plot_fig_tau_networks_comb

#%%
# ---- Save figures as along the way ---- 
save_figs = False  # can turn on or off to save figures or not
fig_dir = 'C:/Users/ys2605/Desktop/stuff/papers/AC_paper_protocol/figures/python_int'    # edit this
#%%
# ---- loading mismatch CaIm datasets ----
# for external datasets theses steps need to be modified
data_dir = 'F:/AC_data/caiman_data_missmatch/'   # edit this 
 
# loading raw firing rates, trial types, and stimuli times from oddball dataset
data_ob = f_load_caim_data_mat(
    data_dir,                            # data directory
    ext_list=['mat'],                    # data extension
    tags=['ammn', '_processed_data'],    # data tags
    num_files=20,                        # limit number of files to load
    deconvolution='oasis',               # deconvolution oasis or smoothdfdt
    smooth_std_duration=0.1)             # in sec

frame_rate = 1000/np.mean(f_get_values(data_ob, 'volume_period')) # in Hz; edit this for external datasets

#%%
# ---- loading RNN neuronal activity durung control inputs ----
# three types of RNNs trainings: odball recognition, control freq recognition, and untrained
# all three were tested with control inputs and neuronal activity was extracted
data_dir_rnn = 'F:/RNN_stuff/RNN_data/test_data/'
fname_rnn = 'RNN_test_data_2024_5_24_9h_42m'
data_rnn = f_load_rnn_test(
    data_dir_rnn, 
    fname_rnn + '_cont_data.npy',
    fname_rnn + '_params.npy',
    max_net_load = 5,
    flatten_runs = False,
    cut_zero_trials = True,
    num_initial_trials_skip = 20,
    limit_network_types=['ob trained', 'freq trained', 'untrained'])     # ob trained, freq trained, untrained

frame_rate_rnn = 1000/np.mean(f_get_values(data_rnn, 'volume_period')) # in Hz; edit this for external datasets
training_type = np.array(f_get_values(data_rnn, 'training'))

#%%
# ---- Compute intrinsic timescales for CaIm data ----
tau_ob_net_all = []
tau_ob_cell_all = []
for n_dset in range(len(data_ob)):
    
    print('CaIm dset %d' %(n_dset))
    
    tau_net, tau_cell = f_get_network_intrinsic_timescales(data_ob[n_dset]['firing_rates'], frame_rate)
    
    tau_ob_net_all.append(tau_net)
    tau_ob_cell_all.append(tau_cell)

#%%
# ---- Compute intrinsic timescales for RNN data ----
tau_rnn_net_all = []
tau_rnn_cell_all = []
for n_rnn in range(len(data_rnn)):
    
    print('rnn %d' %(n_rnn))
    
    tau_net, tau_cell = f_get_network_intrinsic_timescales(data_rnn[n_rnn]['firing_rates'], frame_rate_rnn)

    tau_rnn_net_all.append(tau_net)
    tau_rnn_cell_all.append(tau_cell)

#%%
# ---- Plotting intrinsic timescales for CaIm and RNN data ----
fig = f_plot_fig_tau_networks_comb(
    tau_ob_net_all,
    tau_rnn_net_all,
    training_type,
    tau_ob_cell_all,
    tau_rnn_cell_all,
    do_log=True)

if save_figs:
    f_save_fig(fig, path=fig_dir, name_tag='')

