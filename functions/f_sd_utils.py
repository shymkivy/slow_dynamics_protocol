# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 13:32:23 2025

@author: ys2605
"""

import os
import h5py
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from datetime import datetime



#%%
def f_get_fnames_from_dir(dir_path, ext_list = [], tags = None):

    f_list = os.listdir(dir_path)
    f_list2 = []
    for fil1 in f_list:
        if len(ext_list):
            for ext1 in ext_list:
                if fil1.endswith(ext1):
                    f_list2.append(fil1)
        else:
            f_list2.append(fil1)
    
    if tags is not None:
        if type(tags) is str:
            tags = [tags]
        
        f_list_out = []
        for fil1 in f_list2:
            has_tag = True
            for tag in tags:
                if tag not in fil1:
                    has_tag = False
            if has_tag:
                f_list_out.append(fil1)
            
    else:
        f_list_out = f_list2
    
    return f_list_out


#%%

def f_load_firing_rates(fpath, data_tag = 'results_cnmf_sort.mat', proc_tag = 'processed_data.mat', deconvolution='oasis', smooth_std_duration=0.1):
    
    # deconvolution methods are either oasis (caiman default) or smoothdftd - smoothed, rectified first derivative
    
    fpath2 = fpath.removesuffix('_results_cnmf_sort.mat')
    
    dir_path, fname = os.path.split(fpath2)
    
    fname_core = fname
    if data_tag in fname_core:
        fname_core = fname_core.removesuffix(data_tag)
    if proc_tag in fname_core:
        fname_core = fname_core.removesuffix(proc_tag)
    
    flist_data = f_get_fnames_from_dir(dir_path, ext_list = ['.mat'], tags = [fname_core, data_tag])
    fname_proc = f_get_fnames_from_dir(dir_path, ext_list = ['.mat'], tags = [fname_core, proc_tag])
    
    do_load = False
    if len(fname_proc):
        if len(flist_data):
            do_load = True
        else:
            print(fname_core + " data file with " + data_tag +  " tag not found, skipping")
    else:
        print(fname_core + " proc file with " + proc_tag +  " tag not found, skipping")
    
    data_out = {'files_loaded':     do_load,
                'flist_data':       flist_data,
                'flist_proc':       fname_proc}
    if do_load:
        
        f_proc = h5py.File(dir_path + '/' + fname_proc[0], 'r')
        vid_cuts_trace = f_proc[f_proc['data']['file_cuts_params'][0][0]]['vid_cuts_trace'][()].flatten().astype(bool)
        trial_types = f_proc['data']['trial_types'][()].flatten().astype(int)
        stim_times = f_proc[f_proc['data']['stim_times_frame'][0][0]][()].flatten().astype(int)
        
        if 'volume_period' in f_proc['data']['frame_data'].keys():
            data_out['volume_period'] = f_proc['data']['frame_data']['volume_period'][()].flatten()[0]
        if 'isi' in f_proc['data']['stim_params'].keys():
            data_out['isi'] = f_proc['data']['stim_params']['isi'][()].flatten()[0]
        if 'MMN_orientations' in f_proc['data'].keys():
            data_out['MMN_ori'] = f_proc['data']['MMN_orientations'][()].flatten().astype(int)
        if 'MMN_freq' in f_proc['data']['stim_params'].keys():
            data_out['MMN_ori'] = f_proc['data']['stim_params']['MMN_freq'][()].flatten().astype(int)

        f_proc.close()
        
        firing_rates_all = []
        dset_idx_all = []
        
        for n_fl in range(len(flist_data)):
            fname_data = flist_data[n_fl]
            f = h5py.File(dir_path + '/' + fname_data, 'r')
        
            d_est = f['est']
            d_proc = f['proc']
            
            comp_acc = d_proc['comp_accepted'][()].flatten().astype(bool)
            
            if deconvolution == 'oasis':
                firing_rates_cut = d_est['S'][()][:,comp_acc].T

            elif deconvolution == 'smoothdfdt':
                C = d_est['C'][()]
                YrA = d_est['YrA'][()]
                ca_traces_cut = (C + YrA)[:,comp_acc].T
                firing_rates_cut = f_smooth_dfdt(ca_traces_cut, sigma_frames=1000/data_out['volume_period']*0.1, do_smooth=True)
            
            firing_rates_cut = f_gauss_smooth(firing_rates_cut, sigma_frames=1000/data_out['volume_period']*smooth_std_duration)
            
            peak_rate = np.max(firing_rates_cut, axis=1)[:,None]
            firing_rates_cutn = firing_rates_cut/peak_rate
            
            firing_rates = np.zeros((firing_rates_cutn.shape[0], vid_cuts_trace.shape[0]))
            firing_rates[:, vid_cuts_trace] = firing_rates_cutn
            
            firing_rates_all.append(firing_rates)
            dset_idx_all.append(np.ones(firing_rates.shape[0], dtype=int)*(n_fl))
            f.close()
        
        firing_rates = np.vstack(firing_rates_all)
        dset_idx = np.hstack(dset_idx_all)
        
        data_out['firing_rates'] = firing_rates
        data_out['trial_types'] = trial_types
        data_out['stim_times'] = stim_times
        data_out['vid_cuts_trace'] = vid_cuts_trace
        data_out['files_loaded'] = do_load
        data_out['dset_idx'] = dset_idx
        
    return data_out
    

def f_smooth_dfdt(data, do_smooth=True, sigma_frames=1, rectify=True, normalize=True):
    
    num_cells, num_frames = data.shape
    
    firing_rates = np.zeros((num_cells, num_frames));
    
    if sigma_frames == 0:
        do_smooth=False
    
    if do_smooth:
        s_fr = np.ceil(sigma_frames).astype(int)
        x = np.linspace(-3*s_fr, 3*s_fr, s_fr*6+1)
        gauss_kernel = np.exp(-x**2 / (2 * sigma_frames**2))
    
    for n_cell in range(num_cells):
        temp_data = np.diff(data[n_cell,:], prepend=0)
        
        if do_smooth:
            temp_data = np.convolve(temp_data, gauss_kernel, mode='same')
        
        if rectify:
            temp_data = np.maximum(temp_data, 0)
        
        if normalize:
            temp_data = temp_data - np.mean(temp_data)
            temp_data = temp_data/np.max(temp_data)
            
        firing_rates[n_cell,:] = temp_data;
    
    return firing_rates

def f_gauss_smooth(firing_rates, sigma_frames=1):
    
    if sigma_frames:
        num_cells, num_frames = firing_rates.shape
        
        s_fr = np.ceil(sigma_frames).astype(int)
        x = np.linspace(-3*s_fr, 3*s_fr, s_fr*6+1)
        gauss_kernel = np.exp(-x**2 / (2 * sigma_frames**2))
        
        firing_rates_sm = np.zeros((num_cells, num_frames));
        for n_cell in range(num_cells):
            firing_rates_sm[n_cell,:] = np.convolve(firing_rates[n_cell,:], gauss_kernel, mode='same')
    else:
        firing_rates_sm = firing_rates
    
    return firing_rates_sm


def f_get_frames(trial_win = [-0.05, .95], frame_rate = 30):
    # anchor at 0
    frame_start = np.ceil(trial_win[0] * frame_rate)
    frame_end = np.ceil(trial_win[1] * frame_rate)
    trial_frames = [int(frame_start), int(frame_end)]
    plot_t = np.arange(frame_start/frame_rate, frame_end/frame_rate, 1/frame_rate) 
    return trial_frames, plot_t

def f_get_stim_trig_resp(firing_rates, stim_times, trial_frames = [-29, 85]):
    # input: cells x time
    
    num_cells = firing_rates.shape[0]
    num_trials = len(stim_times)
    
    win_size = trial_frames[1] - trial_frames[0]
    
    stim_trig_resp = np.zeros((num_cells, win_size, num_trials))
    
    for n_tr in range(num_trials):
        cur_frame = round(stim_times[n_tr])
        stim_trig_resp[:,:,n_tr] = firing_rates[:,(cur_frame+trial_frames[0]):(cur_frame+trial_frames[1])]
    
    return stim_trig_resp
    
def f_save_fig(fig, path='/', name_tag=''):
    
    plt.rcParams['svg.fonttype'] = 'none'
    name1 = fig.axes[0].title.get_text()
    now1 = datetime.now()
    
    date_tag = '%d_%d_%d_%dh_%dm' % (now1.year, now1.month, now1.day, now1.hour, now1.minute)
    
    fig.savefig('%s/%s_%s%s.svg' % (path, name1, date_tag, name_tag))
    fig.savefig('%s/%s_%s%s.png' % (path, name1, date_tag, name_tag), dpi=1200)


def f_get_trial_peak(trial_ave, peak_size=3):
    num_cells, num_bins = trial_ave.shape
    
    pad_left = np.floor((peak_size-1)/2).astype(int)
    pad_right = np.ceil((peak_size-1)/2).astype(int)
    
    peak_locs = np.argmax(trial_ave,axis=1)
    
    peak_start = peak_locs - pad_left
    peak_end = peak_locs + pad_right + 1
    
    idx_fix_start = peak_start < 0
    if sum(idx_fix_start):
        peak_start_to_fix = peak_start[idx_fix_start]
        peak_start[idx_fix_start] = peak_start[idx_fix_start] - peak_start_to_fix
        peak_end[idx_fix_start] = peak_end[idx_fix_start] - peak_start_to_fix
    
    idx_fix_end = peak_end > num_bins
    if sum(idx_fix_end):
        peak_end_to_fix = peak_end[idx_fix_end]
        peak_end[idx_fix_end] = peak_end[idx_fix_end] - peak_end_to_fix + num_bins
        peak_start[idx_fix_end] = peak_start[idx_fix_end] - peak_end_to_fix + num_bins
    
    peak_vals = np.zeros(num_cells)
    for n_cell in range(num_cells):
        peak_vals[n_cell] = np.mean(trial_ave[n_cell,peak_start[n_cell]:peak_end[n_cell]])
    
    return peak_vals, peak_locs


def f_compute_tuning(stim_trig_resp, trial_types, trials_analyze, plot_t, num_samp=2000, z_thresh = 3, sig_resp_win = [0, 1.5]):
    
    tt_use_idx = np.sum(trial_types == trials_analyze[:,None],axis=0).astype(bool)
    trial_types_use = trial_types[tt_use_idx]
    stim_trig_resp_use = stim_trig_resp[:,:,tt_use_idx]
    
    num_cells, _, num_trials = stim_trig_resp_use.shape
    num_tt = len(trials_analyze)
    
    # get data
    peak_vals = np.full((num_cells, num_tt), np.nan)
    peak_locs = np.full((num_cells, num_tt), np.nan)
    for n_tt in range(num_tt):
        tt1_idx = trial_types_use == trials_analyze[n_tt]
        if len(tt1_idx):
            trial_ave1 = np.mean(stim_trig_resp_use[:,:,tt1_idx], axis=2)
            peak_vals[:,n_tt], peak_locs[:,n_tt] = f_get_trial_peak(trial_ave1, peak_size=3)
    
    # make shuffled dist
    trials_per_stim = np.zeros(num_tt, dtype=int)
    for n_tt in range(num_tt):
        trials_per_stim[n_tt] = np.sum(trial_types_use == trials_analyze[n_tt])
    trials_per_stim_ave = np.round(np.mean(trials_per_stim)).astype(int)
    
    samp_peak_vals = np.full((num_cells, num_samp), np.nan)
    samp_peak_locs = np.full((num_cells, num_samp), np.nan)
    rng = np.random.default_rng()
    for n_cell in range(num_cells):
        random_integers = rng.integers(low=0, high=num_trials, size=(trials_per_stim_ave, num_samp))
        samp_trial_ave = np.mean(stim_trig_resp_use[n_cell,:,random_integers], axis=0)
        samp_peak_vals[n_cell,:], samp_peak_locs[n_cell,:] = f_get_trial_peak(samp_trial_ave, peak_size=3)

    idx1 = ~np.isnan(peak_locs[0,:])
    peak_locs_t = np.full((num_cells, num_tt), np.nan)
    peak_locs_t[:,idx1] = plot_t[peak_locs[:,idx1].astype(int)]
    
    peak_in_resp_win = np.logical_and(peak_locs_t >= sig_resp_win[0], peak_locs_t <= sig_resp_win[1])
    
    peak_prcntle = norm.cdf(z_thresh)*100
    prc_thresh = np.percentile(samp_peak_vals, peak_prcntle, axis=1)
    resp_cells_peak = np.zeros((num_cells, num_tt), dtype=bool)
    
    resp_cells_peak[:,idx1] = np.logical_and(peak_vals[:,idx1] > prc_thresh[:,None], peak_in_resp_win[:,idx1])
    
    return resp_cells_peak

