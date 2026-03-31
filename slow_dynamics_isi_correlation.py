# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 18:11:13 2026

@author: ys2605
"""

# ---- importing functions ----
import sys
import numpy as np
import matplotlib.pyplot as plt

# importing slow dynamics pipeline
pipeline_dir = 'C:/Users/ys2605/Desktop/stuff/slow_dynamics_analysis'    # edit this  
sys.path.append(pipeline_dir + '/functions')
from f_sd_utils import f_load_caim_data_mat, f_get_values, f_get_frames, f_get_mouse_id, f_get_stim_trig_resp, f_compute_tuning, f_compute_correlation, f_plot_fig_isi_corr_trials, f_plot_fig_SI_mat, f_compute_correlation_mat, f_save_fig

#%%
# ---- Save figures as along the way ---- 
save_figs = True  # can turn on or off to save figures or not
fig_dir = 'C:/Users/ys2605/Desktop/stuff/papers/AC_paper_protocol/figures/python_corr'    # edit this

#%% ---- loading echo data ----
data_dir = 'F:/AC_data/caiman_data_echo/'    # edit this

# loading raw firing rates, trial types, and stimuli times from echo dataset
data_echo = f_load_caim_data_mat(
    data_dir,                            # data directory
    ext_list=['mat'],                    # data extension
    tags=['cont', '_processed_data'],    # data tags
    num_files=None,                      # limit number of files to load
    deconvolution='oasis',               # deconvolution oasis or smoothdfdt
    smooth_std_duration=0.1)             # in sec

frame_rate = 1000/np.mean(f_get_values(data_echo, 'volume_period')) # in Hz; edit this for external datasets
isi_list = np.array(f_get_values(data_echo, 'isi'))

#%% ---- calculate cell tuning and extract responsive cells ----

trial_frames_tuning, plot_t_tuning = f_get_frames(
    trial_win=[-1, 2],         # sec relative to stim onset (pre, post)
    frame_rate=frame_rate)

resp_cells_all = []
for n_fl in range(len(data_echo)):  
    # computing stimulus triggered average (neurons, frames, trials)
    stim_trig_resp = f_get_stim_trig_resp(
        data_echo[n_fl]['firing_rates'],    # firing rates
        data_echo[n_fl]['stim_times'],      # stimulus onset frames
        trial_frames=trial_frames_tuning)     # frames to be extracted
    
    print('computing stats dset %d/%d' % (n_fl+1, len(data_echo)))
    resp_cells = f_compute_tuning(
        stim_trig_resp,
        data_echo[n_fl]['trial_types'],
        np.arange(1,11), 
        plot_t_tuning,
        num_samp=2000,
        z_thresh = 3,
        sig_resp_win = [0, 1.2])
    
    resp_cells_all.append(resp_cells)
print('Done')

#%% ---- corrrlation analysis CaIm echo data    ----

trial_frames, plot_t = f_get_frames(
    trial_win = [-0.05, .95],         # sec relative to stim onset (pre, post)
    frame_rate = frame_rate)

corr_vals_all = []
stim_trig_resp_all = []
for n_fl in range(len(data_echo)):
    
    stim_trig_resp = f_get_stim_trig_resp(
        data_echo[n_fl]['firing_rates'],    # firing rates
        data_echo[n_fl]['stim_times'],      # stimulus onset frames
        trial_frames=trial_frames)          # frames to be extracted
    
    stim_trig_resp_all.append(stim_trig_resp)
    
    corr_vals = f_compute_correlation(
        stim_trig_resp,                     # stimulus triggered response matrix
        data_echo[n_fl]['trial_types'],     # vector of trial types
        np.arange(1,11),                    # trials types to analyze (1-10)
        resp_cells_all[n_fl],               # logical vector of responsive cells
        metric='correlation',               # cosine, correlation
        min_resp_cells=5)                   # minimum number of responsive cells
    
    corr_vals_all.append(corr_vals)

corr_vals = np.vstack(corr_vals_all)

if 0:
    plt.figure()
    plt.imshow(corr_vals)
    
    plt.figure()
    plt.plot(plot_t, np.mean(stim_trig_resp[90,:,data_echo[n_fl]['trial_types']==5], axis=0))

#%%
#  ---- plot correlation for trials 1-10 across ISI ----
fig = f_plot_fig_isi_corr_trials(
    corr_vals,
    isi_list,
    colormap='jet',
    metric_tag = 'Correlation')

# ---- save figure ----
if save_figs:    
    f_save_fig(fig, path=fig_dir, name_tag='Corr fig')
    
#%%
mouse_list = np.array(f_get_mouse_id(data_echo))
mouse_tag = np.unique(mouse_list)[5]
n_freq = 5

isi_uq = np.unique(isi_list)
SI_list = []

for n_isi in range(len(isi_uq)):
    mouse_idx2 = np.where((mouse_list == mouse_tag) & (isi_list == isi_uq[n_isi]))[0][0]
    trial_idx = data_echo[mouse_idx2]['trial_types'] == n_freq
    
    stim_trig_resp = stim_trig_resp_all[mouse_idx2][:,:,trial_idx]
    
    SI = f_compute_correlation_mat(stim_trig_resp, subtract_mean=True, add_noise_sigma=1e-5, metric='cosine')
    SI_list.append(SI)
    
#%%

fig = f_plot_fig_SI_mat(SI_list, isi_uq, title_tag = 'mouse %s, freq %d' % (mouse_tag, n_freq))

# ---- save figure ----
if save_figs:    
    f_save_fig(fig, path=fig_dir, name_tag='Corr SI mat')
    
