# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 12:00:33 2025

@author: yuiry shymkiv; ys2605@columbia.edu
"""

# ---- importing functions ----
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

# importing slow dynamics pipeline
pipeline_dir = 'C:/Users/ys2605/Desktop/stuff/slow_dynamics_analysis'    # edit this  
sys.path.append(pipeline_dir + '/functions')
import sd_utils as sd
import sd_decoder as dec

#%%
save_figs = False  # can turn on or off to save figure or not
fig_dir = 'C:/Users/ys2605/Desktop/stuff/papers/AC_paper_protocol/figures/python'    # edit this
seed = 0           # integer for reproducible results, or None for random

#%% ---- loading mismatch datasets ----
# for external datasets these steps need to be modified
data_dir = 'F:/AC_data/caiman_data_missmatch/'   # edit this 

# loading raw firing rates, trial types, and stimuli times from oddball dataset
data_ob = sd.load_caim_data_mat(
    data_dir,                            # data directory
    ext_list=['mat'],                    # data extension
    tags=['ammn', '_processed_data'],    # data tags
    num_files=20,                        # limit number of files to load
    deconvolution='oasis',               # deconvolution oasis or smoothdfdt
    smooth_std_duration=0.1)             # in sec

frame_rate = 1000/np.mean(sd.get_values(data_ob, 'volume_period')) # in Hz; edit this for external datasets

#%% ---- plot example raster ----
fig = plt.figure()
plt.imshow(data_ob[0]['firing_rates'][:,500:5000], aspect='auto', cmap='gist_yarg', vmin=0, vmax=.5, interpolation='none')
plt.title('Raster')
plt.ylabel('Neurons')
plt.xlabel('Frames')

# ---- save figure ----
if save_figs:    
    sd.save_fig(fig, path=fig_dir, name_tag='Raster')

#%% ---- extracting trials using stimulus times ----
# compute trial window frames (assuming all datasets have similar frame rate)
trial_frames, plot_t = sd.get_frames(
    trial_win=[-1,3],         # sec relative to stim onset (pre, post)
    frame_rate=frame_rate)

# computing stimulus triggered average (neurons, frames, trials)
stim_trig_resp_all = []
for n_fl in range(len(data_ob)):  
    stim_trig_resp = sd.get_stim_trig_resp(
        data_ob[n_fl]['firing_rates'],    # firing rates
        data_ob[n_fl]['stim_times'],      # stimulus onset frames
        trial_frames=trial_frames)        # frames to be extracted
    
    stim_trig_resp_all.append(stim_trig_resp)
    
#%% ---- define diagonal binwise decoder ----
def f_diag_decoder(stim_trig_resp, trial_types, seed=None):
    # the same responses are passed twice in X_all; the control is the label-shuffled
    # version in Y_all (trial labels shuffled), not a second, different dataset
    X_all = [stim_trig_resp,
             stim_trig_resp]

    Y_all = [trial_types,
             dec.shuffle_trials(trial_types, seed=seed)]   # label-shuffled control
    
    dec_data = dec.run_binwise_dec(
        X_all,
        Y_all,
        train_test_method='diag',   # options: full, diag, train_at_stim, test_at_stim
        pca_var_frac=1,             # reduce data with pca
        num_cv=5,                   # cross validation
        normalize=False,
        add_noise_sigma=1e-5,       # for stability 
        get_train_coeffs=True,
        seed=seed)

    return dec_data

#%% run for all datasets
dec_data_all = []
for n_fl in range(len(data_ob)):
    print('Training dataset %d/%d' % (n_fl+1, len(stim_trig_resp_all)))

    trial_types = data_ob[n_fl]['trial_types']
    trial_types_use = trial_types<10

    dec_data = f_diag_decoder(
        stim_trig_resp_all[n_fl][:,:,trial_types_use],
        trial_types[trial_types_use],
        seed=seed)

    dec_data_all.append(dec_data)
    
print('Done')

#%% ---- plotting decoder results ----

fig_diag = dec.plot_diag_binwise_dec(
    dec_data_all,
    plot_t=plot_t,
    plot_legend=('Data', 'Shuff'),
    plot_start=-1,
    plot_end=3,
    title_tag = 'Ca Im data binwise decoder',
    colors = ['blue', 'black'])


# ---- save figure ----
if save_figs:    
    sd.save_fig(fig_diag, path=fig_dir, name_tag='')

#%% ---- define full decoding space binwise decoder ----
def f_full_decoder(stim_trig_resp, trial_types, seed=None):
    # the same responses are passed twice in X_all; the control is the label-shuffled
    # version in Y_all (trial labels shuffled), not a second, different dataset
    X_all = [stim_trig_resp,
             stim_trig_resp]

    Y_all = [trial_types,
             dec.shuffle_trials(trial_types, seed=seed)]   # label-shuffled control
    
    dec_data = dec.run_binwise_dec(
        X_all,
        Y_all,
        train_test_method='full',   # options: full, diag, train_at_stim, test_at_stim
        pca_var_frac=1,             # reduce data with pca
        num_cv=5,                   # cross validation
        normalize=False,
        add_noise_sigma=1e-5,       # for stability 
        log=True,
        seed=seed)
    return dec_data

#%% ---- run example dataset ---- 
n_fl = 0 

trial_types = data_ob[n_fl]['trial_types']
trial_types_use = trial_types<10

dec_data_full = f_full_decoder(
    stim_trig_resp_all[n_fl][:,:,trial_types_use],
    trial_types[trial_types_use],
    seed=seed)

print('Done')

#%%

figs_full = dec.plot_full_binwise_dec(dec_data_full,
                    plot_t=plot_t,
                    plot_legend=('Data', 'Shuff'),
                    plot_start=-1,
                    plot_end=2,
                    clim=[0, 1],
                    clim_width_ratio = 10,
                    figsize=(17, 3.5),
                    title_tag='Freq single trial decoder')

if save_figs:
    sd.save_fig(figs_full, path=fig_dir, name_tag='')


#%%
if save_figs:
    sd.save_fig(figs_full['diag'], path=fig_dir, name_tag='')
    sd.save_fig(figs_full['fixed_train'], path=fig_dir, name_tag='')
    sd.save_fig(figs_full['fixed_test'], path=fig_dir, name_tag='')

#%% correlation stuff

trial_frames, plot_t = sd.get_frames(trial_win = [0.03, .95], frame_rate=1000/np.mean(sd.get_values(data_ob, 'volume_period')))

stim_trig_resp_corr_all = []
for n_fl in range(len(data_ob)):  
    # computing stimulus triggered average (neurons, frames, trials)
    stim_trig_resp = sd.get_stim_trig_resp(data_ob[n_fl]['firing_rates'], data_ob[n_fl]['stim_times'], trial_frames = trial_frames)
    stim_trig_resp_corr_all.append(stim_trig_resp)
    
#%% corr analysis

trials_analyze = np.arange(1,11)
corr_vals = np.zeros((len(stim_trig_resp_corr_all), len(trials_analyze)))

for n_fl in range(len(stim_trig_resp_corr_all)):
    stim_trig_resp = stim_trig_resp_corr_all[n_fl]
    trial_types = data_ob[n_fl]['trial_types']
    
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
plt.imshow(data_ob['ca_traces'], aspect='auto', cmap='gray')

fig1, ax1 = plt.subplots()
im1 = ax1.imshow(data_ob['firing_rates'][:,data_ob['vid_cuts_trace']], aspect='auto', cmap='gray', clim=[0, 0.2])
ax1.set_title('Firing rates raster; dataset %d' % (n_fl)) 
ax1.set_ylabel('Neurons')
ax1.set_xlabel('Frames')
cbar = fig1.colorbar(im1)
cbar.set_label('Firing rate')

#sd.save_fig(fig1, path=fig_dir, name_tag='')

plt.figure()
plt.plot(data_ob['firing_rates'][1,:])
plt.plot(data_ob['ca_traces'][1,:]/np.max(data_ob['ca_traces'][1,:]))


x = np.mean(stim_trig_resp_all[n_fl][:,data_ob[n_fl]['trial_types'] == 4,:], axis=1)
plt.figure()
plt.imshow(x.T)
plt.ylabel('neurons')
plt.xlabel('frames')

plt.figure()
plt.plot(np.mean(x, axis=1))

#%%
#plt.set_cmap('viridis')     # cividis, viridis, 

