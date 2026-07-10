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
import sd_utils as sd

#%%
# ---- Save figures as along the way ---- 
save_figs = True  # can turn on or off to save figures or not
fig_dir = 'C:/Users/ys2605/Desktop/stuff/papers/AC_paper_protocol/figures/python_corr'    # edit this
seed = 0           # integer for reproducible results, or None for random

#%% ---- loading echo data ----
data_dir = 'F:/AC_data/caiman_data_echo/'    # edit this

# loading raw firing rates, trial types, and stimuli times from echo dataset
data_echo = sd.load_caim_data_mat(
    data_dir,                            # data directory
    ext_list=['mat'],                    # data extension
    tags=['cont', '_processed_data'],    # data tags
    num_files=None,                      # limit number of files to load
    deconvolution='smoothdfdt',          # deconvolution oasis or smoothdfdt
    smooth_std_duration=0.150)           # in sec

frame_rate = 1000/np.mean(sd.get_values(data_echo, 'volume_period')) # in Hz; edit this for external datasets
isi_list = np.array(sd.get_values(data_echo, 'isi'))

#%% ---- calculate cell tuning and extract responsive cells ----

trial_frames_tuning, plot_t_tuning = sd.get_frames(
    trial_win=[-1, 2],         # sec relative to stim onset (pre, post)
    frame_rate=frame_rate)

resp_cells_all = []
for n_fl in range(len(data_echo)):  
    # computing stimulus triggered average (neurons, frames, trials)
    stim_trig_resp = sd.get_stim_trig_resp(
        data_echo[n_fl]['firing_rates'],    # firing rates
        data_echo[n_fl]['stim_times'],      # stimulus onset frames
        trial_frames=trial_frames_tuning)     # frames to be extracted
    
    print('computing stats dset %d/%d' % (n_fl+1, len(data_echo)))
    resp_cells = sd.compute_tuning(
        stim_trig_resp,
        data_echo[n_fl]['trial_types'],
        np.arange(1,11), 
        plot_t_tuning,
        num_samp=2000,
        z_thresh = 3,
        sig_resp_win = [0, 1.2],
        seed=seed)

    resp_cells_all.append(resp_cells)
print('Done')

#%% ---- correlation analysis CaIm echo data    ----

trial_frames, plot_t = sd.get_frames(
    trial_win = [-0.05, .95],         # sec relative to stim onset (pre, post)
    frame_rate = frame_rate)

corr_vals_all = []
stim_trig_resp_all = []
for n_fl in range(len(data_echo)):
    
    stim_trig_resp = sd.get_stim_trig_resp(
        data_echo[n_fl]['firing_rates'],    # firing rates
        data_echo[n_fl]['stim_times'],      # stimulus onset frames
        trial_frames=trial_frames)          # frames to be extracted
    
    stim_trig_resp_all.append(stim_trig_resp)
    
    corr_vals = sd.compute_correlation(
        stim_trig_resp,                     # stimulus triggered response matrix
        data_echo[n_fl]['trial_types'],     # vector of trial types
        np.arange(1,11),                    # trials types to analyze (1-10)
        resp_cells_all[n_fl],               # logical vector of responsive cells
        metric='correlation',               # cosine, correlation
        min_resp_cells=5,                   # minimum number of responsive cells
        seed=seed)
    
    corr_vals_all.append(corr_vals)

corr_vals = np.vstack(corr_vals_all)

if 0:
    plt.figure()
    plt.imshow(corr_vals)
    
    plt.figure()
    plt.plot(plot_t, np.mean(stim_trig_resp[90,:,data_echo[n_fl]['trial_types']==5], axis=0))

#%%
#  ---- plot correlation for trials 1-10 across ISI ----
import importlib
importlib.reload(sd)

fig, ax, groups = sd.plot_fig_isi_corr_trials(
    corr_vals,
    isi_list,
    colormap='jet',
    metric_tag = 'Correlation')

# statistical comparison: prints result + draws bracket on the panel (scale auto-detected)
# test: 'ttest' (Welch), 'mannwhitney', 'wilcoxon'/'ttest_rel' (paired); alternative: 'greater'/'less'/'two-sided'
sd.stat_compare(ax[0], groups, 'ISI 0.5', 'ISI 1', test='mannwhitney', alternative='two-sided')
sd.stat_compare(ax[0], groups, 'ISI 0.5', 'ISI 2', test='mannwhitney', alternative='two-sided')
sd.stat_compare(ax[0], groups, 'ISI 0.5', 'ISI 4', test='mannwhitney', alternative='two-sided')

# ---- save figure ----
if save_figs:
    sd.save_fig(fig, path=fig_dir, name_tag='Corr fig')
    
#%%
mouse_list, mouse_tag = sd.get_example_mouse(data_echo, example_mouse='M4372')
n_freq = 5

isi_uq = np.unique(isi_list)
SI_list = []

for n_isi in range(len(isi_uq)):
    mouse_idx2 = np.where((mouse_list == mouse_tag) & (isi_list == isi_uq[n_isi]))[0][0]
    trial_idx = data_echo[mouse_idx2]['trial_types'] == n_freq
    
    stim_trig_resp = stim_trig_resp_all[mouse_idx2][:,:,trial_idx]
    
    SI = sd.compute_correlation_mat(stim_trig_resp, subtract_mean=True, add_noise_sigma=1e-5, metric='cosine', seed=seed)
    SI_list.append(SI)
    
#%%

fig = sd.plot_fig_SI_mat(SI_list, isi_uq, title_tag = 'mouse %s, freq %d' % (mouse_tag, n_freq))

# ---- save figure ----
if save_figs:    
    sd.save_fig(fig, path=fig_dir, name_tag='Corr SI mat')
    
