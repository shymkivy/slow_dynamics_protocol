# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 18:11:13 2026

@author: ys2605
"""

# ---- importing functions ----
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

# importing slow dynamics pipeline
pipeline_dir = 'C:/Users/ys2605/Desktop/stuff/slow_dynamics_analysis'    # edit this  
sys.path.append(pipeline_dir + '/functions')
from f_sd_utils import f_get_fnames_from_dir, f_load_caim_data, f_get_values, f_get_frames, f_get_stim_trig_resp, f_compute_tuning, f_save_fig

#%% ---- loading echo data ----
data_dir = 'F:/AC_data/caiman_data_echo/'
# search for files to load using tags in the filename
flist = f_get_fnames_from_dir(data_dir, ext_list = ['mat'], tags=['cont', '_processed_data'])

# loading raw firing rates, trial types, and stimuli times
# here you can indicate to use oasis deconvolution or smoothdfdt
data_out = f_load_caim_data(data_dir, flist, deconvolution='oasis', smooth_std_duration=0)

#%% ---- calculate cell tuning and extract responsive cells ----
# indicate a list of trial types to analyze. In this case the trial types are indexed from 1-10
trials_analyze = np.arange(1,11)

trial_frames, plot_t_tuning = f_get_frames(trial_win = [-1, 2], frame_rate = 1000/np.mean(f_get_values(data_out, 'volume_period')))

resp_cells_all = []
for n_fl in range(len(data_out)):  
    # computing stimulus triggered average (neurons, frames, trials)
    stim_trig_resp = f_get_stim_trig_resp(data_out[n_fl]['firing_rates'], data_out[n_fl]['stim_times'], trial_frames=trial_frames)

    print('computing stats dset %d/%d' % (n_fl+1, len(data_out)))
    resp_cells = f_compute_tuning(stim_trig_resp, data_out[n_fl]['trial_types'], trials_analyze, plot_t_tuning, num_samp=2000, z_thresh = 3, sig_resp_win = [0, 1.2])
    resp_cells_all.append(resp_cells)

#%% ---- corrrlation analysis echo ----
trials_analyze = np.arange(1,11)

trial_frames, plot_t = f_get_frames(trial_win = [-0.05, .95], frame_rate = 1000/np.mean(f_get_values(data_out, 'volume_period')))
corr_vals = np.full((len(data_out), len(trials_analyze)), np.nan)

for n_fl in range(len(data_out)):
    
    stim_trig_resp = f_get_stim_trig_resp(data_out[n_fl]['firing_rates'], data_out[n_fl]['stim_times'], trial_frames=trial_frames)
    trial_types = data_out[n_fl]['trial_types']
    resp_cells = resp_cells_all[n_fl]
    
    if 0:
        stim_trig_resp = stim_trig_resp - np.mean(stim_trig_resp)

    if 1:   # add some uncorrelated noise to add stability
        stim_trig_resp = stim_trig_resp + np.random.normal(0, 1e-5, size=stim_trig_resp.shape)
    
    resp_marg = np.sum(resp_cells, axis=1).astype(bool)
    for n_tn in range(len(trials_analyze)):
        
        if sum(resp_cells[:,n_tn]) > 5:
            tn1 = trials_analyze[n_tn]
            tr_idx = trial_types == tn1
            
            stim_trig_resp2 = stim_trig_resp[:,:,tr_idx]
            stim_trig_resp3 = stim_trig_resp2[resp_marg,:,:]
            stim_trig_resp4 = np.mean(stim_trig_resp3, axis=1)
            
            distances = squareform(pdist(stim_trig_resp4.T, metric='correlation'))     # cosine, correlation
            SI = 1 - distances
            SI2 = np.tril(SI, k=-1)
            corr_vals[n_fl, n_tn] = np.mean(SI2[SI2.astype(bool)])
            
            if 0:
                if tn1==4:
                    plt.figure()
                    plt.imshow(SI)
                    plt.title('isi = ' + str(data_out[n_fl]['isi']))

if 0:
    plt.figure()
    plt.imshow(corr_vals)
    
    plt.figure()
    plt.plot(plot_t, np.mean(stim_trig_resp[92,:,trial_types==5], axis=0))

#%%
isi_all = f_get_values(data_out, 'isi')
idx_uq = np.unique(isi_all)
col1 = plt.colormaps['jet'](np.linspace(0, 1, 10))

plt.figure()
corr_tn_all = np.zeros((len(trials_analyze), len(idx_uq)))
for n_tn in range(len(trials_analyze)):
    corr_tn = np.full(len(idx_uq), np.nan)
    for n_isi in range(len(idx_uq)):
        idx1 = (idx_uq[n_isi] == np.array(isi_all)).flatten()
        if np.sum(~np.isnan(corr_vals[idx1,n_tn])):
            corr_tn[n_isi] = np.nanmean(corr_vals[idx1,n_tn])
            corr_tn_all[n_tn, n_isi] = np.nanmean(corr_vals[idx1,n_tn])
    if np.sum(~np.isnan(corr_tn)):
        plt.plot(idx_uq, corr_tn, '-o', color=col1[n_tn])
    
plt.plot(idx_uq, np.nanmean(corr_tn_all, axis=0), '-o', color='k')





if 0:
    plt.figure()
    plt.imshow(col1[:,None,:3])