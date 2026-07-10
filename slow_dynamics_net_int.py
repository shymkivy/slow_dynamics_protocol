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
import sd_utils as sd

#%%
# ---- Save figures as along the way ---- 
save_figs = False  # can turn on or off to save figures or not
fig_dir = 'C:/Users/ys2605/Desktop/stuff/papers/AC_paper_protocol/figures/python_int'    # edit this
seed = 0           # integer for reproducible results, or None for random
#%%
# ---- loading mismatch CaIm datasets ----
# for external datasets these steps need to be modified
data_dir = 'F:/AC_data/caiman_data_missmatch/'   # edit this 
 
# loading raw firing rates, trial types, and stimuli times from oddball dataset
# returns a list of dicts (one per dataset); main field 'firing_rates' is (neurons, time)
data_ob = sd.load_caim_data_mat(
    data_dir,                            # data directory
    ext_list=['mat'],                    # data extension
    tags=['ammn', '_processed_data'],    # data tags
    num_files=20,                        # limit number of files to load
    deconvolution='oasis',               # deconvolution oasis or smoothdfdt
    smooth_std_duration=0.1)             # in sec

frame_rate = 1000/np.mean(sd.get_values(data_ob, 'volume_period')) # in Hz; edit this for external datasets

#%%
# ---- loading RNN neuronal activity during control inputs ----
# three types of RNNs trainings: oddball recognition, control freq recognition, and untrained
# all three were tested with control inputs and neuronal activity was extracted
# returns a list of dicts (one per network); firing_rates is (neurons, time) if flatten_runs else (runs, neurons, time)
data_dir_rnn = 'F:/RNN_stuff/RNN_data/test_data/'
fname_rnn = 'RNN_test_data_2024_5_24_9h_42m2'
data_rnn = sd.load_rnn_test(
    data_dir_rnn, 
    fname_rnn + '_cont_data.npy',
    fname_rnn + '_params.npy',
    max_net_load = 5,               # max networks per type (lower = less memory)
    flatten_runs = False,           # keep runs separate as 3D (runs, neurons, time), for per-run INT
    cut_zero_trials = True,         # drop prepended zero-padding trials
    num_initial_trials_skip = 20,   # skip first N trials
    limit_network_types=['ob trained', 'freq trained', 'untrained'],     # types to load; [] = all
    seed=seed)                      # reproducibility seed (None = random)

frame_rate_rnn = 1000/np.mean(sd.get_values(data_rnn, 'volume_period')) # in Hz; edit this for external datasets
training_type = np.array(sd.get_values(data_rnn, 'training'))

#%%
# ---- Compute intrinsic timescales for CaIm data ----
tau_ob_net_all = []
tau_ob_cell_all = []
for n_dset in range(len(data_ob)):
    
    print('CaIm dset %d' %(n_dset))
    
    tau_net, tau_cell = sd.get_network_intrinsic_timescales(data_ob[n_dset]['firing_rates'], frame_rate)
    
    tau_ob_net_all.append(tau_net)
    tau_ob_cell_all.append(tau_cell)

#%%
# ---- Compute intrinsic timescales for RNN data ----
tau_rnn_net_all = []
tau_rnn_cell_all = []
for n_rnn in range(len(data_rnn)):
    
    print('rnn %d' %(n_rnn))
    
    tau_net, tau_cell = sd.get_network_intrinsic_timescales(data_rnn[n_rnn]['firing_rates'], frame_rate_rnn)

    tau_rnn_net_all.append(tau_net)
    tau_rnn_cell_all.append(tau_cell)

#%%
# ---- Plotting intrinsic timescales for CaIm and RNN data ----
# do_log=True -> log y-scale ; do_log=False -> linear y-scale
fig, ax, groups = sd.plot_fig_tau_networks_comb(
    tau_ob_net_all,
    tau_rnn_net_all,
    training_type,
    tau_ob_cell_all,
    tau_rnn_cell_all,
    do_log=True)

# statistical comparison: prints result + draws bracket on the panel (scale auto-detected)
# test: 'ttest' (Welch), 'mannwhitney', 'wilcoxon'/'ttest_rel' (paired); alternative: 'greater'/'less'/'two-sided'
sd.stat_compare(ax, groups, 'CaIm net', 'CaIm neuron',
                test='mannwhitney', alternative='two-sided')
sd.stat_compare(ax, groups, 'CaIm neuron', 'Ob neuron',
                test='mannwhitney', alternative='two-sided')
sd.stat_compare(ax, groups, 'CaIm neuron', 'Freq neuron',
                test='mannwhitney', alternative='two-sided')
sd.stat_compare(ax, groups, 'CaIm net', 'Ob net',
                test='mannwhitney', alternative='two-sided')
sd.stat_compare(ax, groups, 'CaIm net', 'Freq net',
                test='mannwhitney', alternative='two-sided')

if save_figs:
    sd.save_fig(fig, path=fig_dir, name_tag='')

