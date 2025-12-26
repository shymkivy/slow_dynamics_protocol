# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 12:00:33 2025

@author: ys2605
"""

# importing functions

pipeline_dir = 'C:/Users/ys2605/Desktop/stuff/slow_dynamics_analysis'

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

sys.path.append(pipeline_dir + '/functions')
from f_sd_utils import f_get_fnames_from_dir, f_load_firing_rates, f_smooth_dfdt, f_get_frames, f_get_stim_trig_resp
from f_sd_decoder import f_run_binwise_dec, f_plot_binwise_dec, f_shuffle_trials

from f_sd_utils import f_save_fig

#%%
# loading datasets
data_dir = 'F:/AC_data/caiman_data_missmatch/'
flist = f_get_fnames_from_dir(data_dir, ext_list = ['mat'], tags=['ammn', '_processed_data'])  # 'results_cnmf_sort'

# setting parameters
frame_rate = 1000/35.14     # in Hz
# [pre, post] time around stimulus onset to analyze
        # in sec

#%%
firing_rates_all = []
trial_types_all = []
stim_times_all = []
mmn_ori_all = []
for n_fl in range(len(flist)):
    # loading raw calcium data, trial types, and stimuli times
    fpath = data_dir + flist[n_fl]
    data_out = f_load_firing_rates(fpath=fpath)
    if data_out['files_loaded']:
        # deconvolving calcium traces into firing rate proxy
        firing_rates = f_smooth_dfdt(data_out['ca_traces'], sigma_frames=frame_rate*0.1)     # 100ms smoothing std
        
        firing_rates_all.append(firing_rates)
        trial_types_all.append(data_out['trial_types'])
        stim_times_all.append(data_out['stim_times'])
        mmn_ori_all.append(data_out['MMN_ori'])

#%% getting stim trig resp for decoder
# computing the number of frames to in the trial window
trial_frames, plot_t = f_get_frames(trial_win = [-1, 3], frame_rate=frame_rate)

stim_trig_resp_all = []
for n_fl in range(len(firing_rates_all)):  
    # computing stimulus triggered average (neurons, frames, trials)
    stim_trig_resp = np.transpose(f_get_stim_trig_resp(firing_rates_all[n_fl], stim_times_all[n_fl], trial_frames = trial_frames), axes=(1, 2, 0))
    stim_trig_resp_all.append(stim_trig_resp)
    
#%%
train_test_method = 'diag'          # full, diag, train_at_stim, test_at_stim
plot_legend = ('Data', 'Shuff')

perform_diag_all = []
for n_fl in range(len(stim_trig_resp_all)):
    stim_trig_resp = stim_trig_resp_all[n_fl]
    trial_types = trial_types_all[n_fl]
    trial_types_use = trial_types<=10
    
    X_all = [stim_trig_resp[:,trial_types_use,:], stim_trig_resp[:,trial_types_use,:]]
    Y_all = [trial_types[trial_types_use], f_shuffle_trials(trial_types[trial_types_use])]
    
    perform_diag = f_run_binwise_dec(X_all, Y_all, train_test_method=train_test_method, pca_var_frac = 1)
    perform_diag_all.append(perform_diag)

#%%
figs_diag = f_plot_binwise_dec(perform_diag_all, train_test_method=train_test_method, plot_t=plot_t, plot_legend=plot_legend, plot_start=-1, plot_end=3, title_tag='Freq single trial decoder')

# f_save_fig(figs_diag['diag'], path=fig_dir, name_tag='')

#%%
train_test_method = 'full'          # full, diag, train_at_stim, test_at_stim
n_fl = 0

trial_types_use = trial_types_all[n_fl]<=10

X_all = [stim_trig_resp_all[n_fl][:,trial_types_use,:], stim_trig_resp_all[n_fl][:,trial_types_use,:]]
Y_all = [trial_types_all[n_fl][trial_types_use], f_shuffle_trials(trial_types_all[n_fl][trial_types_use])]
plot_legend = ('Data', 'Shuff')

perform_full = f_run_binwise_dec(X_all, Y_all, train_test_method=train_test_method, pca_var_frac = 1)

figs_full = f_plot_binwise_dec(perform_full, train_test_method=train_test_method, plot_t=plot_t, plot_legend=plot_legend, plot_start=-1, plot_end=2, fixed_time=0.25, title_tag='Freq single trial decoder')

# f_save_fig(figs_full['full'][0], path=fig_dir, name_tag='')
# f_save_fig(figs_full['full'][1], path=fig_dir, name_tag='')

# f_save_fig(figs_full['diag'], path=fig_dir, name_tag='')
# f_save_fig(figs_full['fixed_train'], path=fig_dir, name_tag='')
# f_save_fig(figs_full['fixed_test'], path=fig_dir, name_tag='')

#%% correlation stuff

trial_frames, plot_t = f_get_frames(trial_win = [0.05, .95], frame_rate = frame_rate)
stim_trig_resp_corr_all = []
for n_fl in range(len(firing_rates_all)):  
    # computing stimulus triggered average (neurons, frames, trials)
    stim_trig_resp = np.transpose(f_get_stim_trig_resp(firing_rates_all[n_fl], stim_times_all[n_fl], trial_frames = trial_frames), axes=(1, 2, 0))
    stim_trig_resp_corr_all.append(stim_trig_resp)
    
#%% corr analysis

trials_analyze = np.arange(1,11)
corr_vals = np.zeros((len(stim_trig_resp_corr_all), len(trials_analyze)))

for n_fl in range(len(stim_trig_resp_corr_all)):
    stim_trig_resp = stim_trig_resp_corr_all[n_fl]
    trial_types = trial_types_all[n_fl]
    
    for n_tn in range(len(trials_analyze)):
        tn1 = trials_analyze[n_tn]
        tr_idx = trial_types == tn1
        
        stim_trig_resp2 = stim_trig_resp[:,tr_idx,:]
        stim_trig_resp3 = np.mean(stim_trig_resp2, axis=0)
        
        
        distances = squareform(pdist(stim_trig_resp3))
        SI = 1 - distances
        SI2 = np.tril(SI, k=-1)
        corr_vals[n_fl, n_tn] = np.mean(SI2[SI2.astype(bool)])
        
        if 0:
            plt.figure()
            plt.imshow(SI2)
    


#%% loading echo data

# loading datasets
data_dir_echo = 'F:/AC_data/caiman_data_echo/'
flist_echo = f_get_fnames_from_dir(data_dir_echo, ext_list = ['mat'], tags=['cont', '_processed_data'])

# setting parameters
frame_rate = 1000/107.2125   # in Hz
# [pre, post] time around stimulus onset to analyze
        # in sec

firing_rates_all = []
trial_types_all = []
stim_times_all = []
isi_all = []
for n_fl in range(len(flist_echo)):
    # loading raw calcium data, trial types, and stimuli times
    fpath = data_dir_echo + flist_echo[n_fl]
    data_out = f_load_firing_rates(fpath=fpath)
    
    if data_out['files_loaded']:
        # deconvolving calcium traces into firing rate proxy
        firing_rates = f_smooth_dfdt(data_out['ca_traces'], sigma_frames=frame_rate*0.1)     # 100ms smoothing std
        
        firing_rates_all.append(firing_rates)
        trial_types_all.append(data_out['trial_types'])
        stim_times_all.append(data_out['stim_times'])
        isi_all.append(data_out['isi'])

#%% correlation stuff echo data

trial_frames, plot_t = f_get_frames(trial_win = [0.05, .95], frame_rate = frame_rate)
stim_trig_resp_corr_all = []
for n_fl in range(len(firing_rates_all)):  
    # computing stimulus triggered average (neurons, frames, trials)
    stim_trig_resp = np.transpose(f_get_stim_trig_resp(firing_rates_all[n_fl], stim_times_all[n_fl], trial_frames = trial_frames), axes=(1, 2, 0))
    stim_trig_resp_corr_all.append(stim_trig_resp)

#%% corr analysis echo

trials_analyze = np.arange(1,11)
corr_vals = np.zeros((len(stim_trig_resp_corr_all), len(trials_analyze)))

for n_fl in range(len(stim_trig_resp_corr_all)):
    stim_trig_resp = stim_trig_resp_corr_all[n_fl]
    trial_types = trial_types_all[n_fl]
    
    for n_tn in range(len(trials_analyze)):
        tn1 = trials_analyze[n_tn]
        tr_idx = trial_types == tn1
        
        stim_trig_resp2 = stim_trig_resp[:,tr_idx,:]
        stim_trig_resp3 = np.mean(stim_trig_resp2, axis=0)
        
        distances = squareform(pdist(stim_trig_resp3))
        SI = 1 - distances
        SI2 = np.tril(SI, k=-1)
        corr_vals[n_fl, n_tn] = np.mean(SI2[SI2.astype(bool)])
        
        if 0:
            plt.figure()
            plt.imshow(SI2)

#%%
idx_uq = np.unique(isi_all)

plt.figure()
for n_isi in range(len(idx_uq)):
    idx1 = idx_uq[n_isi] == np.array(isi_all).flatten()
    plt.plot(np.ones(np.sum(idx1))*idx_uq[n_isi], corr_vals[idx1,:], '.', 'c')


#%%
# plt.close('all')

fig_dir = 'C:/Users/ys2605/Desktop/stuff/papers/AC_paper_protocol/figures/python/'

plt.figure()
plt.imshow(data_out['ca_traces'], aspect='auto', cmap='gray')

fig1, ax1 = plt.subplots()
im1 = ax1.imshow(firing_rates[:,data_out['vid_cuts_trace']], aspect='auto', cmap='gray', clim=[0, 0.2])
ax1.set_title('Firing rates raster; dataset %d' % (n_fl)) 
ax1.set_ylabel('Neurons')
ax1.set_xlabel('Frames')
cbar = fig1.colorbar(im1)
cbar.set_label('Firing rate')

#f_save_fig(fig1, path=fig_dir, name_tag='')

plt.figure()
plt.plot(firing_rates[1,:])
plt.plot(data_out['ca_traces'][1,:]/np.max(data_out['ca_traces'][1,:]))


x = np.mean(stim_trig_resp_all[n_fl][:,trial_types_all[n_fl] == 4,:], axis=1)
plt.figure()
plt.imshow(x.T)
plt.ylabel('neurons')
plt.xlabel('frames')

plt.figure()
plt.plot(np.mean(x, axis=1))

#%%
#plt.set_cmap('viridis')     # cividis, viridis, 

