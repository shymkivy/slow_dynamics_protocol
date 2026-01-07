# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 12:00:33 2025

@author: ys2605
"""

# ---- importing functions ----
pipeline_dir = 'C:/Users/ys2605/Desktop/stuff/slow_dynamics_analysis'    # edit this  

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

sys.path.append(pipeline_dir + '/functions')
from f_sd_utils import f_get_fnames_from_dir, f_load_firing_rates, f_get_frames, f_get_stim_trig_resp
from f_sd_decoder import f_run_binwise_dec, f_plot_binwise_dec, f_shuffle_trials
from f_sd_utils import f_save_fig

#%% ---- loading mismatch datasets ----
data_dir = 'F:/AC_data/caiman_data_missmatch/'    # edit this  
# search for files to load using tags in the filename
flist = f_get_fnames_from_dir(data_dir, ext_list = ['mat'], tags=['ammn', '_processed_data'])  # 'results_cnmf_sort'

firing_rates_all = []
trial_types_all = []
stim_times_all = []
mmn_ori_all = []
volume_period_all = []

# here you can limit how many datasets to analyze
for n_fl in range(20):   # len(flist) or some number
    # loading raw firing rates, trial types, and stimuli times
    # here you can indicate to use oasis deconvolution or smoothdfdt
    data_out = f_load_firing_rates(fpath=data_dir + flist[n_fl], deconvolution='oasis', smooth_std_duration=0)     # oasis, smoothdfdt; normally smooth with 0.1sec 
    
    # extract data and parameters we will use from each dataset
    if data_out['files_loaded']:
        firing_rates_all.append(data_out['firing_rates'])
        trial_types_all.append(data_out['trial_types'])
        stim_times_all.append(data_out['stim_times'])
        mmn_ori_all.append(data_out['MMN_ori'])
        volume_period_all.append(data_out['volume_period'])

#%% ---- extracting trials using stimulus times ----
# trial window to extract is indicated in seconds, also frame rate needs to be provided (assuming all datasets have similar frame rate)
trial_frames, plot_t = f_get_frames(trial_win = [-1, 3], frame_rate=1000/np.mean(volume_period_all))

stim_trig_resp_all = []
for n_fl in range(len(firing_rates_all)):  
    # computing stimulus triggered average (neurons, frames, trials)
    stim_trig_resp = f_get_stim_trig_resp(firing_rates_all[n_fl], stim_times_all[n_fl], trial_frames = trial_frames)
    stim_trig_resp_all.append(stim_trig_resp)
    
#%% ---- training binwise decoder diagonally ----
train_test_method = 'diag'     # training options: full, diag, train_at_stim, test_at_stim
plot_legend = ('Data', 'Shuff')

dec_data_all = []
for n_fl in range(len(stim_trig_resp_all)):
    print('Training dataset %d/%d' % (n_fl+1, len(stim_trig_resp_all)))
    stim_trig_resp = stim_trig_resp_all[n_fl]
    trial_types = trial_types_all[n_fl]
    trial_types_use = trial_types<=10
    
    X_all = [stim_trig_resp[:,:,trial_types_use], stim_trig_resp[:,:,trial_types_use]]
    # to creatae shuffled version we shuffle the trial types
    Y_all = [trial_types[trial_types_use], f_shuffle_trials(trial_types[trial_types_use])]
    
    dec_data = f_run_binwise_dec(X_all, Y_all, train_test_method=train_test_method, pca_var_frac = 1, num_cv=5, normalize = False, add_noise_sigma=1e-5, get_train_coeffs=True)
    dec_data_all.append(dec_data)

#%% ---- plotting decoder results ----
figs_diag = f_plot_binwise_dec(dec_data_all, plot_t=plot_t, plot_legend=plot_legend, plot_start=-1, plot_end=3, title_tag='Freq single trial decoder')

# ---- save figure ----
fig_dir = 'C:/Users/ys2605/Desktop/stuff/papers/AC_paper_protocol/figures/python'  # edit this  
save_figs = False  # can turn on or off to save figure or not
if save_figs:
    f_save_fig(figs_diag['diag'], path=fig_dir, name_tag='')

#%% ---- analyze the full decoding space ----
train_test_method = 'full'          # full, diag, train_at_stim, test_at_stim
n_fl = 0        # dataset to analyze

trial_types_use = trial_types_all[n_fl]<=10

X_all = [stim_trig_resp_all[n_fl][:,:,trial_types_use], stim_trig_resp_all[n_fl][:,:,trial_types_use]]
Y_all = [trial_types_all[n_fl][trial_types_use], f_shuffle_trials(trial_types_all[n_fl][trial_types_use])]
plot_legend = ('Data', 'Shuff')

dec_data_full = f_run_binwise_dec(X_all, Y_all, train_test_method=train_test_method, pca_var_frac = 1, num_cv=5, normalize = False, add_noise_sigma=1e-5, log=True)

figs_full = f_plot_binwise_dec(dec_data_full, plot_t=plot_t, plot_legend=plot_legend, plot_start=-1, plot_end=2, fixed_time=0.25, title_tag='Freq single trial decoder')

# f_save_fig(figs_full['full'][0], path=fig_dir, name_tag='')
# f_save_fig(figs_full['full'][1], path=fig_dir, name_tag='')

# f_save_fig(figs_full['diag'], path=fig_dir, name_tag='')
# f_save_fig(figs_full['fixed_train'], path=fig_dir, name_tag='')
# f_save_fig(figs_full['fixed_test'], path=fig_dir, name_tag='')

#%% correlation stuff

trial_frames, plot_t = f_get_frames(trial_win = [0.03, .95], frame_rate=1000/np.mean(volume_period_all))
stim_trig_resp_corr_all = []
for n_fl in range(len(firing_rates_all)):  
    # computing stimulus triggered average (neurons, frames, trials)
    stim_trig_resp = f_get_stim_trig_resp(firing_rates_all[n_fl], stim_times_all[n_fl], trial_frames = trial_frames)
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
        
        stim_trig_resp2 = stim_trig_resp[:,:,tr_idx]
        stim_trig_resp3 = np.mean(stim_trig_resp2, axis=1)
        
        
        distances = squareform(pdist(stim_trig_resp3.T))
        SI = 1 - distances
        SI2 = np.tril(SI, k=-1)
        corr_vals[n_fl, n_tn] = np.mean(SI2[SI2.astype(bool)])
        
        if 0:
            plt.figure()
            plt.imshow(SI2)
    





#%%
# plt.close('all')

fig_dir = 'C:/Users/ys2605/Desktop/stuff/papers/AC_paper_protocol/figures/python/'

plt.figure()
plt.imshow(data_out['ca_traces'], aspect='auto', cmap='gray')

fig1, ax1 = plt.subplots()
im1 = ax1.imshow(data_out['firing_rates'][:,data_out['vid_cuts_trace']], aspect='auto', cmap='gray', clim=[0, 0.2])
ax1.set_title('Firing rates raster; dataset %d' % (n_fl)) 
ax1.set_ylabel('Neurons')
ax1.set_xlabel('Frames')
cbar = fig1.colorbar(im1)
cbar.set_label('Firing rate')

#f_save_fig(fig1, path=fig_dir, name_tag='')

plt.figure()
plt.plot(data_out['firing_rates'][1,:])
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

