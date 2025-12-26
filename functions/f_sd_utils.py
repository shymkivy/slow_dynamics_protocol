# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 13:32:23 2025

@author: ys2605
"""

import os
import h5py
import numpy as np
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

def f_load_firing_rates(fpath, data_tag = 'results_cnmf_sort.mat', proc_tag = 'processed_data.mat'):
    
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
            data_out['volume_period'] = f_proc['data']['frame_data']['volume_period'][()].flatten()
        if 'isi' in f_proc['data']['stim_params'].keys():
            data_out['isi'] = f_proc['data']['stim_params']['isi'][()].flatten()
        if 'MMN_orientations' in f_proc['data'].keys():
            data_out['MMN_ori'] = f_proc['data']['MMN_orientations'][()].flatten().astype(int)
        if 'MMN_freq' in f_proc['data']['stim_params'].keys():
            data_out['MMN_ori'] = f_proc['data']['stim_params']['MMN_freq'][()].flatten().astype(int)

        f_proc.close()
        
        ca_traces_all = []
        dset_idx_all = []
        
        for n_fl in range(len(flist_data)):
            fname_data = flist_data[n_fl]
            f = h5py.File(dir_path + '/' + fname_data, 'r')
        
            d_est = f['est']
            d_proc = f['proc']
        
            C = d_est['C'][()]
            YrA = d_est['YrA'][()]
            
            comp_acc = d_proc['comp_accepted'][()].flatten().astype(bool)
    
            ca_traces_cut = (C + YrA)[:,comp_acc].T
            ca_traces = np.zeros((ca_traces_cut.shape[0], vid_cuts_trace.shape[0]))
            ca_traces[:, vid_cuts_trace] = ca_traces_cut
            
            ca_traces_all.append(ca_traces)
            dset_idx_all.append(np.ones(ca_traces.shape[0], dtype=int)*(n_fl))
            f.close()
        
        ca_traces = np.vstack(ca_traces_all)
        dset_idx = np.hstack(dset_idx_all)
        
        data_out['ca_traces'] = ca_traces
        data_out['trial_types'] = trial_types
        data_out['stim_times'] = stim_times
        data_out['vid_cuts_trace'] = vid_cuts_trace
        data_out['files_loaded'] = do_load
        data_out['dset_idx'] = dset_idx
        
    return data_out
    

def f_smooth_dfdt(data, do_smooth=True, sigma_frames=1, normalize=True, rectify=True):
    
    num_cells, num_frames = data.shape
    
    firing_rates = np.zeros((num_cells, num_frames));
    
    s_fr = np.ceil(sigma_frames).astype(int)
    x = np.linspace(-3*s_fr, 3*s_fr, s_fr*6+1)
    gauss_kernel = np.exp(-x**2 / (2 * sigma_frames**2))
    
    for n_cell in range(num_cells):
        temp_data = np.diff(data[n_cell,:], prepend=0)
        
        if do_smooth:
            temp_data = np.convolve(temp_data, gauss_kernel, mode='same')
        
        if normalize:
            temp_data = temp_data - np.mean(temp_data)
            temp_data = temp_data/np.max(temp_data)
            
        if rectify:
            temp_data = np.maximum(temp_data, 0)

        firing_rates[n_cell,:] = temp_data;
    
    return firing_rates


def f_get_frames(trial_win = [-0.05, .95], frame_rate = 30):
    # anchor at 0
    frame_start = np.floor(trial_win[0] * frame_rate)
    frame_end = np.ceil(trial_win[1] * frame_rate)
    trial_frames = [int(frame_start), int(frame_end)]
    plot_t = np.arange(frame_start/frame_rate, frame_end/frame_rate, 1/frame_rate) 
    return trial_frames, plot_t

def f_get_stim_trig_resp(firing_rates, stim_times, trial_frames = [-29, 85]):
    
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


