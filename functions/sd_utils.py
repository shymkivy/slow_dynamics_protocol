# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 13:32:23 2025

@author: ys2605
"""

import os
import h5py
import numpy as np
from scipy.stats import norm, wilcoxon, mannwhitneyu, ttest_ind, ttest_rel, f_oneway
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.signal import correlate #, correlation_lags

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter
from datetime import datetime
from sd_decoder import plot_diag_binwise_dec, plot_full_binwise_dec

#%%
def get_fnames_from_dir(data_dir, ext_list = [], tags = None, f_list=None):
    # list files in data_dir, keeping those with an ext_list extension AND containing every tag; f_list overrides the listing

    if f_list is None:
        if not os.path.isdir(data_dir):
            raise FileNotFoundError('Data directory not found: %s' % data_dir)
        f_list = os.listdir(data_dir)
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

def load_caim_data_mat(data_dir, ext_list = [], tags = None, num_files=None, data_tag = 'results_cnmf_sort.mat', proc_tag = 'processed_data.mat', deconvolution='oasis', smooth_std_duration=0.1, norm_first=False):

    # load matched caiman .mat sessions from data_dir into a list of dataset dicts (one per session);
    # main fields: firing_rates (neurons x time), trial_types, stim_times, volume_period, isi.
    # deconvolution methods are either oasis (caiman default) or smoothdftd - smoothed, rectified first derivative
    # norm_first: False (default) = smooth then peak-normalize (original); True = peak-normalize raw S then smooth (MATLAB order)
    # smooth_std_duration in sec
    
    
    dir_files = get_fnames_from_dir(data_dir)
    flist = get_fnames_from_dir(data_dir, ext_list=ext_list, tags=tags, f_list=dir_files)

    if len(flist) == 0:
        print('Warning: no files found in %s matching ext %s and tags %s' % (data_dir, ext_list, tags))
        return []

    if num_files is None:
        num_files = len(flist)
    else:
        num_files = np.min([len(flist), num_files])

    data_out = []
    
    for n_fl in range(num_files):

        fname_core = flist[n_fl]
        if data_tag in fname_core:
            fname_core = fname_core.removesuffix(data_tag)
        if proc_tag in fname_core:
            fname_core = fname_core.removesuffix(proc_tag)
        
        flist_data = get_fnames_from_dir(data_dir, ext_list = ['.mat'], tags = [fname_core, data_tag], f_list=dir_files)
        flist_proc = get_fnames_from_dir(data_dir, ext_list = ['.mat'], tags = [fname_core, proc_tag], f_list=dir_files)
        
        do_load = False
        if len(flist_proc):
            if len(flist_data):
                do_load = True
            else:
                print(fname_core + " data file with " + data_tag +  " tag not found, skipping")
        else:
            print(fname_core + " proc file with " + proc_tag +  " tag not found, skipping")
        
        if do_load:
            
            data_slice = {'flist_data':       flist_data,
                          'flist_proc':       flist_proc}
            
            f_proc = h5py.File(data_dir + '/' + flist_proc[0], 'r')
            vid_cuts_trace = f_proc[f_proc['data']['file_cuts_params'][0][0]]['vid_cuts_trace'][()].flatten().astype(bool)
            trial_types = f_proc['data']['trial_types'][()].flatten().astype(int)
            stim_times = f_proc[f_proc['data']['stim_times_frame'][0][0]][()].flatten().astype(int)
            
            if 'volume_period' in f_proc['data']['frame_data'].keys():
                data_slice['volume_period'] = f_proc['data']['frame_data']['volume_period'][()].flatten()[0]
            if data_slice.get('volume_period', 0) <= 0:
                print('Warning: %s missing or invalid volume_period in _processed_data; using default 33.3 ms (~30 Hz).' % fname_core)
                data_slice['volume_period'] = 33.3   # ms, default ~30 Hz
            if 'isi' in f_proc['data']['stim_params'].keys():
                data_slice['isi'] = f_proc['data']['stim_params']['isi'][()].flatten()[0]
            if 'MMN_orientations' in f_proc['data'].keys():
                data_slice['MMN_ori'] = f_proc['data']['MMN_orientations'][()].flatten().astype(int)
            if 'MMN_freq' in f_proc['data']['stim_params'].keys():
                data_slice['MMN_ori'] = f_proc['data']['stim_params']['MMN_freq'][()].flatten().astype(int)
    
            f_proc.close()
            
            firing_rates_all = []
            dset_idx_all = []
            
            for n_fl in range(len(flist_data)):
                fname_data = flist_data[n_fl]
                
                f = h5py.File(os.path.join(data_dir, fname_data), 'r')
            
                d_est = f['est']
                d_proc = f['proc']
                
                comp_acc = d_proc['comp_accepted'][()].flatten().astype(bool)
                
                if deconvolution == 'oasis':
                    firing_rates_cut = d_est['S'][()][:,comp_acc].T
    
                elif deconvolution == 'smoothdfdt':
                    C = d_est['C'][()]
                    YrA = d_est['YrA'][()]
                    ca_traces_cut = (C + YrA)[:,comp_acc].T
                    firing_rates_cut = smooth_dfdt(ca_traces_cut, sigma_frames=1000/data_slice['volume_period']*0.1, do_smooth=True)
                
                sigma_fr = 1000/data_slice['volume_period']*smooth_std_duration
                if norm_first:
                    # MATLAB order: peak-normalize the raw deconvolved S first, then smooth
                    peak_rate = np.max(firing_rates_cut, axis=1)[:,None]
                    peak_rate[peak_rate == 0] = 1
                    firing_rates_cutn = gauss_smooth(firing_rates_cut/peak_rate, sigma_frames=sigma_fr)
                else:
                    # original order: smooth, then peak-normalize the smoothed trace
                    firing_rates_cut = gauss_smooth(firing_rates_cut, sigma_frames=sigma_fr)
                    peak_rate = np.max(firing_rates_cut, axis=1)[:,None]
                    peak_rate[peak_rate == 0] = 1
                    firing_rates_cutn = firing_rates_cut/peak_rate
                
                firing_rates = np.zeros((firing_rates_cutn.shape[0], vid_cuts_trace.shape[0]))
                firing_rates[:, vid_cuts_trace] = firing_rates_cutn
                
                firing_rates_all.append(firing_rates)
                dset_idx_all.append(np.ones(firing_rates.shape[0], dtype=int)*(n_fl))
                f.close()
            
            firing_rates = np.vstack(firing_rates_all)
            dset_idx = np.hstack(dset_idx_all)
            
            data_slice['fname_core'] = fname_core
            data_slice['firing_rates'] = firing_rates
            data_slice['trial_types'] = trial_types
            data_slice['stim_times'] = stim_times
            data_slice['vid_cuts_trace'] = vid_cuts_trace
            data_slice['files_loaded'] = do_load
            data_slice['dset_idx'] = dset_idx
        
            data_out.append(data_slice)

    if len(data_out):
        mouse_ids = list(np.unique(get_mouse_id(data_out)))
        print('Loaded %d datasets from %d mice: %s' % (len(data_out), len(mouse_ids), mouse_ids))

    return data_out

def load_caim_data_mat2(*args, **kwargs):
    # same as load_caim_data_mat but with the MATLAB preprocessing order:
    # peak-normalize the raw deconvolved S first, then smooth (norm_first=True)
    kwargs['norm_first'] = True
    return load_caim_data_mat(*args, **kwargs)

def h5_load_group(group, keys=None):
    # load the given keys (default: all) of an open h5py group into a plain dict
    if keys is None:
        keys = group.keys()
        
    data = {}
    for key1 in keys:
        data[key1] = group[key1][()]
    
    return data

def get_values(data_out, key):
    # collect data_slice[key] across the datasets in the list (skips slices missing key)

    values = []
    
    for data_slice in data_out:
        if key in data_slice.keys():
            values.append(data_slice[key])
            
    return values

def get_mouse_id(data_echo):
    # mouse id per dataset = text before the first underscore of fname_core (e.g. 'M4372')

    fnames = get_values(data_echo, 'fname_core')
    
    mouse_id = []
    for fname in fnames:
        mouse_id.append(fname.split('_')[0])

    return mouse_id

def get_example_mouse(data_echo, example_mouse='M4372'):
    # return (per-dataset mouse-id array, example_mouse); error if example_mouse is not among the loaded datasets

    mouse_list = np.array(get_mouse_id(data_echo))
    mouse_uq = np.unique(mouse_list)

    if example_mouse not in mouse_uq:
        raise ValueError('Example mouse %s not found among loaded datasets. The trial-to-trial '
                         'similarity demo requires all six variable-ISI (echo) datasets: '
                         'M226, M4264, M4265, M4266, M4371, M4372. Loaded mice: %s'
                         % (example_mouse, list(mouse_uq)))

    return mouse_list, example_mouse

def smooth_dfdt(data, do_smooth=True, sigma_frames=1, rectify=True, normalize=True):
    # smoothdfdt firing-rate estimate: per-neuron smoothed first derivative, optionally rectified and peak-normalized

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

def gauss_smooth(firing_rates, sigma_frames=1):
    # gaussian smoothing of each neuron's trace (sigma in frames; 0 = no smoothing)

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

def normalize(rates):
    # assumes neurons x time
    
    rates_n = rates - np.min(rates, axis=1)[:, None]
    
    # ignoring cells that are always zero
    max_rates = np.max(rates_n, axis=1)
    has_max = max_rates > 0
    
    rates_n = rates_n[has_max,:] / max_rates[has_max][:,None]
    
    return rates_n
    

def get_frames(trial_win = [-0.05, .95], frame_rate = 30):
    # anchor at 0
    frame_start = np.ceil(trial_win[0] * frame_rate)
    frame_end = np.ceil(trial_win[1] * frame_rate)
    trial_frames = [int(frame_start), int(frame_end)]
    plot_t = np.round(np.arange(frame_start/frame_rate, frame_end/frame_rate, 1/frame_rate), decimals=4)
    return trial_frames, plot_t

def get_stim_trig_resp(firing_rates, stim_times, trial_frames = [-29, 85]):
    # input: cells x time
    
    num_cells, T = firing_rates.shape
    num_trials = len(stim_times)
    
    win_size = trial_frames[1] - trial_frames[0]
    
    stim_trig_resp = np.zeros((num_cells, win_size, num_trials))
    
    for n_tr in range(num_trials):
        cur_frame = round(stim_times[n_tr]-1) # correct for matlab to python 
        
        raw_start = cur_frame + trial_frames[0]
        raw_end = cur_frame + trial_frames[1]

        src_start = np.max([raw_start, 0])
        src_end = np.min([raw_end, T])

        if src_end > src_start:
            dst_start = src_start - raw_start
            stim_trig_resp[:, dst_start:dst_start + (src_end - src_start), n_tr] = firing_rates[:, src_start:src_end]
    
    return stim_trig_resp
    
def save_fig(fig, path='/', name_tag=''):
    # save the figure as svg + png (1200 dpi), named from the first-axis title + date + name_tag

    plt.rcParams['svg.fonttype'] = 'none'
    name1 = fig.axes[0].title.get_text()
    now1 = datetime.now()
    
    date_tag = '%d_%d_%d_%dh_%dm' % (now1.year, now1.month, now1.day, now1.hour, now1.minute)
    
    fig.savefig('%s/%s_%s%s.svg' % (path, name1, date_tag, name_tag))
    fig.savefig('%s/%s_%s%s.png' % (path, name1, date_tag, name_tag), dpi=1200)


def get_trial_peak(trial_ave, peak_size=3):
    # per-neuron peak response: mean over a peak_size-frame window centred on each neuron's argmax; returns (peak_vals, peak_locs)
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


def compute_tuning(stim_trig_resp, trial_types, trials_analyze, plot_t, num_samp=2000, z_thresh = 3, sig_resp_win = [0, 1.5], seed=None):
    # find stimulus-responsive cells per trial type by comparing each cell's peak response to a shuffled null
    # (z_thresh over num_samp shuffles, within sig_resp_win); returns a (cells x trial types) responsive-cell mask

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
        if np.any(tt1_idx):
            trial_ave1 = np.mean(stim_trig_resp_use[:,:,tt1_idx], axis=2)
            peak_vals[:,n_tt], peak_locs[:,n_tt] = get_trial_peak(trial_ave1, peak_size=3)
    
    # make shuffled dist
    trials_per_stim = np.zeros(num_tt, dtype=int)
    for n_tt in range(num_tt):
        trials_per_stim[n_tt] = np.sum(trial_types_use == trials_analyze[n_tt])
    trials_per_stim_ave = np.round(np.mean(trials_per_stim)).astype(int)
    
    samp_peak_vals = np.full((num_cells, num_samp), np.nan)
    samp_peak_locs = np.full((num_cells, num_samp), np.nan)
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    for n_cell in range(num_cells):
        random_integers = rng.integers(low=0, high=num_trials, size=(trials_per_stim_ave, num_samp))
        samp_trial_ave = np.mean(stim_trig_resp_use[n_cell,:,random_integers], axis=0)
        samp_peak_vals[n_cell,:], samp_peak_locs[n_cell,:] = get_trial_peak(samp_trial_ave, peak_size=3)

    idx1 = ~np.isnan(peak_locs[0,:])
    peak_locs_t = np.full((num_cells, num_tt), np.nan)
    peak_locs_t[:,idx1] = plot_t[peak_locs[:,idx1].astype(int)]
    
    peak_in_resp_win = np.logical_and(peak_locs_t >= sig_resp_win[0], peak_locs_t <= sig_resp_win[1])
    
    peak_prcntle = norm.cdf(z_thresh)*100
    prc_thresh = np.percentile(samp_peak_vals, peak_prcntle, axis=1)
    resp_cells_peak = np.zeros((num_cells, num_tt), dtype=bool)
    
    resp_cells_peak[:,idx1] = np.logical_and(peak_vals[:,idx1] > prc_thresh[:,None], peak_in_resp_win[:,idx1])
    
    return resp_cells_peak


def compute_correlation(stim_trig_resp, trial_types, trials_analyze, resp_cells=None, min_resp_cells=5, subtract_mean=False, add_noise_sigma=1e-5, metric='correlation', cell_select='resp_marg', drop_zero_cells=True, seed=None):
    # cell_select: 'resp_marg' (union of responsive cells; = MATLAB 'Resp marg'),
    #              'resp_split' (per-frequency responsive cells; = MATLAB 'Resp split'), or 'all'
    # drop_zero_cells: True drops cells with non-positive mean across trials (Python-only; MATLAB keeps all)
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    corr_vals = np.full((len(trials_analyze)), np.nan)
    
    if subtract_mean:
        stim_trig_resp = stim_trig_resp - np.mean(stim_trig_resp)

    if add_noise_sigma:   # add some uncorrelated noise to add stability
        stim_trig_resp = stim_trig_resp + rng.normal(0, add_noise_sigma, size=stim_trig_resp.shape)
    
    if resp_cells is not None:
        resp_marg = np.sum(resp_cells, axis=1).astype(bool)
    for n_tn in range(len(trials_analyze)):
        
        if resp_cells is not None:
            trig = sum(resp_cells[:,n_tn]) > min_resp_cells
        else:
            trig = 1
        
        if trig:
            
            tn1 = trials_analyze[n_tn]
            tr_idx = trial_types == tn1
            
            stim_trig_resp2 = stim_trig_resp[:,:,tr_idx]
            # cell selection mode
            if resp_cells is None or cell_select == 'all':
                cell_mask = np.ones(stim_trig_resp2.shape[0], dtype=bool)
            elif cell_select == 'resp_split':
                cell_mask = resp_cells[:,n_tn].astype(bool)     # per-frequency responsive cells
            else:  # 'resp_marg'
                cell_mask = resp_marg                           # union of responsive cells
            stim_trig_resp3 = stim_trig_resp2[cell_mask,:,:]
            stim_trig_resp4 = np.mean(stim_trig_resp3, axis=1)

            if drop_zero_cells:
                act_idx = np.mean(stim_trig_resp4, axis=1) > 0
                stim_trig_resp5 = stim_trig_resp4[act_idx,:]
            else:
                stim_trig_resp5 = stim_trig_resp4
            
            distances = squareform(pdist(stim_trig_resp5.T, metric=metric))     # cosine, correlation
            SI = 1 - distances
            SI2 = np.tril(SI, k=-1)
            SI2_vals = SI2[SI2.astype(bool)]
            corr_vals[n_tn] = np.mean(SI2_vals) if len(SI2_vals) else np.nan
            
            # if 0:
            #     if tn1==4:
            #         plt.figure()
            #         plt.imshow(SI)
            #         plt.title('isi = ' + str(data_echo[n_fl]['isi']))
                    
    return corr_vals
        
def compute_correlation_mat(stim_trig_resp, subtract_mean=False, add_noise_sigma=1e-5, metric='correlation', seed=None):
    # trial-by-trial similarity matrix (1 - pairwise distance) of each trial's mean stimulus response
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    if subtract_mean:
        stim_trig_resp = stim_trig_resp - np.mean(stim_trig_resp)

    if add_noise_sigma:   # add some uncorrelated noise to add stability
        stim_trig_resp = stim_trig_resp + rng.normal(0, add_noise_sigma, size=stim_trig_resp.shape)
    
    stim_trig_resp2 = np.mean(stim_trig_resp, axis=1)
    
    distances = squareform(pdist(stim_trig_resp2.T, metric=metric))     # cosine, correlation
    SI = 1 - distances
    
    return SI

#%%
def get_network_distance(firing_rates):
    # euclidean distance of the population vector from its time-averaged baseline at each frame: d(t) = ||r(t) - mean_t r||
    base_pop_vec = np.mean(firing_rates, axis=1)
    tr_dist = cdist(np.reshape(base_pop_vec, (1,len(base_pop_vec))), firing_rates.T, 'euclidean')[0]

    return tr_dist

def get_trace_tau(trace, sm_bin = 0):
    # intrinsic timescale of a trace = first autocorrelation lag (in frames) where it drops below 0.5; returns (tau, autocorr)

    #sm_bin = 10#round(1/params['dt'])*50;
    #trial_len = out_temp_all.shape[1]
    
    
    tracen = trace - np.mean(trace)
    trace_std = np.std(tracen)
    if trace_std == 0:
        return np.nan, np.full(len(trace), np.nan)
    tracen = tracen/trace_std

    corr1 = correlate(tracen, tracen)/len(tracen)
    
    #lags = correlation_lags(len(tracen), len(tracen))
    
    if sm_bin:
        kernel = np.ones(sm_bin)/sm_bin
        corr1_sm = np.convolve(corr1, kernel, mode='same')
        
        corr1_smn = corr1_sm - np.mean(corr1_sm)
        corr1_smn = corr1_smn/np.max(corr1_smn)
    else:
        corr1_smn = corr1
    
    corr1_smn2 = corr1_smn[len(trace)-1:]
    
    # plt.figure(); plt.plot(corr1)
    
    below = np.where(corr1_smn2 < 0.5)[0]
    tau_corr = below[0] if len(below) else np.nan
    
    # x = np.arange(corr_len)+1
    # y = corr1[num_trials2*num_run:num_trials2*num_run+corr_len]
    
    # yn = y - np.min(y)+0.01
    # yn = yn/np.max(yn)
    
    # fit = np.polyfit(x, np.log(yn), 1)  
    
    # y_fit = np.exp(x*fit[0]+fit[1])
    
    # tau_corr = np.log(1/2)/fit[0]*params['dt']
    
    # x = np.random.rand(1000)
    # corrx = correlate(x, x)
    # plt.figure(); plt.plot(corrx)
    
    return tau_corr, corr1_smn2


#%%

def load_rnn_test(data_dir, fname_data, fname_params, max_net_load = 999, limit_network_types=[], flatten_runs = False, max_trial_types = 10, max_trials = 500, cut_zero_trials = False, num_initial_trials_skip = 0, seed=None):
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    # input is the spectrogram inputs
    # target is the index when oddball trial happens
    # loaded rates shape is (time, run, neurons) - inside converted to
    # output rates are converted to (runs, neurons, time)
    # flatten runs makes output rates 2D (neurons, time)
    # max trial types and max trials only apply during flattened runs
    # returns a list of dicts (one per network), matching the calcium-data format;
    # main field 'firing_rates' is (neurons, time) if flatten_runs else (runs, neurons, time)
    
    
    data_path = os.path.join(data_dir, fname_data)
    params_path = os.path.join(data_dir, fname_params)

    for p in (data_path, params_path):
        if not os.path.isfile(p):
            raise FileNotFoundError('RNN test file not found: %s' % p)

    if '_ob_data' in fname_data:
        print('Note: the oddball (_ob_data) RNN file is large (tens of GB) and is loaded fully into memory '
              'before subsetting, so max_net_load does not reduce the peak memory. Ensure sufficient RAM, '
              'or use the smaller control file (_cont_data).')

    test_data_load = np.load(data_path, allow_pickle=True).item()
    data_key = list(test_data_load.keys())[0]
    test_data_load_all = test_data_load[data_key]
    
    # for n1 in range(len(test_data_load_all)):
    #     for n2 in range(len(test_data_load_all[n1])):
    #         del test_data_load_all[n1][n2]['input']
    #         del test_data_load_all[n1][n2]['target']
    #         del test_data_load_all[n1][n2]['output']
    # test_data_load_all[0][0].keys()
    # test_data_load_all[0][0]['rates'].shape
    # np.save(data_dir + 'RNN_test_data_2024_5_24_9h_42m_ob_data2.npy', test_data_load)
    
    deets_load = np.load(params_path, allow_pickle=True).item()
    params_all = deets_load['params_all']
    params_test = deets_load['params_test']
    # deets_load['ob_data'].keys()
    # del deets_load['ob_data']['input_oddball']
    # del deets_load['ob_data']['target_oddball_freq']
    # del deets_load['cont_data']['input_control']
    # del deets_load['cont_data']['target_control']
    # np.save(data_dir + 'RNN_test_data_2024_5_24_9h_42m2_params.npy', deets_load)
     
    data_all = []
    
    if len(limit_network_types) == 0:
        limit_network_types = list(np.unique(deets_load['rnn_leg']))
    use_net_type_lim = True
    
    
    if 'ob' in data_key:
        is_ob = True
    else:
        is_ob = False
    
    # oddball tained, cont trained, and untrained
    for n_net in range(len(test_data_load_all)):
        
        num_rnn = np.min([len(test_data_load_all[n_net]), max_net_load])
        
        if use_net_type_lim:
            if deets_load['rnn_leg'][n_net] in limit_network_types: 
        
                for n_rnn in range(num_rnn):
                    
                    net1 = test_data_load_all[n_net][n_rnn]
                    
                    T, num_runs, num_neurons = net1['rates'].shape
                    
                    # get stim times
                    stim_times_runs = []
                    trial_types_runs = []
                    if is_ob:
                        trial_types_red_dd_runs = []
                    for n_run in range(num_runs):
                        
                        # deets_load['ob_data'].keys() 
                        # deets_load['ob_data']['trials_oddball_freq']
                        if 1:
                            stim_on_trace = 1 - deets_load['ob_data']['target_oddball_ctx3'][:,n_run,0]
                        else:
                            stim_trace = np.max(net1['input'][:,n_run], axis=1)
                            stim_trace_n = stim_trace - np.percentile(stim_trace, 20)
                            stim_trace_n = stim_trace_n / np.max(stim_trace_n)
                            stim_on_trace = (stim_trace_n > 0.5).astype(int)
        
                        stim_onset_trace = np.diff(stim_on_trace, prepend=[0]) > 0
                        stim_times = np.where(stim_onset_trace)[0]
                        stim_times_runs.append(stim_times)
                        
                        if is_ob:
                            ob_deets = deets_load['ob_data']
                            trial_idx = ob_deets['trials_oddball_ctx3'][:,n_run] > 0
                            trials_types_red_dd = ob_deets['trials_oddball_ctx3'][:,n_run][trial_idx] - 1
                            trials_types = ob_deets['trials_oddball_freq'][:, n_run][trial_idx]
                            red_dd_seq = ob_deets['red_dd_seq']
                            
                            trial_types_red_dd_runs.append(trials_types_red_dd)
                            
                        else:
                            cont_deets = deets_load['cont_data']
                            trial_idx = cont_deets['trials_control_freq'][:,n_run] > 0
                            trials_types = cont_deets['trials_control_freq'][:, n_run][trial_idx]
                        
                        trial_types_runs.append(trials_types)
                    
                    stim_times_runs2 = np.vstack(stim_times_runs)
                    trial_types_runs2 = np.vstack(trial_types_runs)
                    if is_ob:
                        trial_types_red_dd_runs2 = np.vstack(trial_types_red_dd_runs)
                    
                    if cut_zero_trials or num_initial_trials_skip > 0:
                        
                        trial_len = round((params_test['stim_duration'] + params_test['isi_duration']) / params_test['dt'])
                        if T % trial_len != 0:
                            raise ValueError('RNN trace length T=%d is not divisible by trial_len=%d; cannot reshape into whole trials (check stim_duration/isi_duration/dt).' % (T, trial_len))
                        rates3d = net1['rates']
                        rates4d = np.reshape(rates3d, (round(T/trial_len), trial_len, num_runs, num_neurons), order='C')
                        
                        if cut_zero_trials:
                            num_skip = params_test['num_prepend_zeros'] + num_initial_trials_skip
                        else:
                            num_skip = num_initial_trials_skip

                        if num_skip >= rates4d.shape[0]:
                            raise ValueError('num_initial_trials_skip (+ prepended zeros) = %d exceeds the %d available trials; reduce num_initial_trials_skip.' % (num_skip, rates4d.shape[0]))

                        rates4d_cut = rates4d[num_skip:,:,:]
                        rates2 = np.reshape(rates4d_cut, ((round(T/trial_len) - num_skip) * trial_len, num_runs, num_neurons), order='C')
                        
                        stim_times_runs2 = stim_times_runs2[:,num_skip:]
                        trial_types_runs2 = trial_types_runs2[:,num_skip:]
                        
                    else:
                        rates2 = net1['rates']
                        
                    T, num_runs, num_neurons = rates2.shape

                    if flatten_runs:
                        stim_times_runs_flat = (stim_times_runs2 + np.arange(num_runs)[:,None] * T).flatten()
                        trial_types_runs_flat = trial_types_runs2.flatten()
                        if is_ob:
                            trial_types_red_dd_runs2 = trial_types_red_dd_runs2.flatten()
                        rates = np.reshape(rates2, (T * num_runs, num_neurons), order='F').T
                        
                        trials_uq = np.unique(trial_types_runs_flat)
                        num_tt = len(trials_uq)
                        
                        num_stim_all = len(stim_times_runs_flat)
                        
                        if max_trial_types < num_tt:
                            step = num_tt/10
                            trial_sel = np.arange(step/2, num_tt, step=step, dtype=int)
                            sel_trial_idx = np.sum(trial_types_runs_flat[:,None] == trials_uq[trial_sel][None,:], axis=1).astype(bool)
                            sel_trials = np.where(sel_trial_idx)[0]
                        else:
                            sel_trials = np.arange(num_stim_all)
        
                        num_stim_all2 = len(sel_trials)
                        
                        # limit trials to max number
                        if max_trials < num_stim_all:
                            trial_idx = rng.choice(sel_trials, size=np.min([max_trials, num_stim_all2]), replace=False)
                            trial_idx.sort()
                            
                        else:
                            trial_idx = sel_trials
                            
                        stim_times_runs3 = stim_times_runs_flat[trial_idx]
                        trial_types_runs3 = trial_types_runs_flat[trial_idx]
                        if is_ob:
                            trial_types_red_dd_runs3 = trial_types_red_dd_runs2[trial_idx]    
                        
                    else:
                        stim_times_runs3 = stim_times_runs2
                        trial_types_runs3 = trial_types_runs2
                        if is_ob:
                            trial_types_red_dd_runs3 = trial_types_red_dd_runs2
                        rates = np.transpose(rates2, (1, 2, 0))
            
                
                    data_slice = {}
                    data_slice['training'] = deets_load['rnn_leg'][n_net]
                    data_slice['test_data_key'] = data_key
                    data_slice['firing_rates'] = rates
                    data_slice['stim_times'] = stim_times_runs3
                    data_slice['trial_types'] = trial_types_runs3
                    if is_ob:
                        data_slice['trial_types_red_dd'] = trial_types_red_dd_runs3
                        data_slice['red_dd_seq'] = red_dd_seq
                    data_slice['num_runs'] = num_runs
                    data_slice['volume_period'] = params_test['dt']*1000   # for frame rate equivalent
                    data_slice['params_train'] = params_all[n_net][n_rnn]
                    data_slice['params_test'] =  params_test
                    
                    data_all.append(data_slice)
            
                    del net1

    if len(data_all):
        trainings = [d['training'] for d in data_all]
        uq, cnt = np.unique(trainings, return_counts=True)
        print('Loaded %d RNN networks: %s' % (len(data_all), dict(zip(list(uq), [int(c) for c in cnt]))))
    else:
        print('Warning: no RNN networks loaded (file %s, limit_network_types=%s)' % (fname_data, limit_network_types))

    return data_all
    
#%%

def get_network_intrinsic_timescales(firing_rates, frame_rate):
    # per recording (or RNN run): network tau from the baseline-distance autocorrelation and per-neuron tau; both in seconds

    if len(firing_rates.shape) == 2:
        rates3d = firing_rates[None,:,:]
    else:
        rates3d = firing_rates
        
    num_runs, num_neurons, _ = rates3d.shape
    
    tau_net = np.zeros(num_runs)
    tau_cell = np.full(shape=(num_runs, num_neurons), fill_value=np.nan)
    
    for n_run in range(num_runs):
        
        tr_dist = get_network_distance(rates3d[n_run,:,:])
        
        tau_net1, _ = get_trace_tau(tr_dist, sm_bin = 0)
        tau_net[n_run] = tau_net1/frame_rate
        
        for n_nr in range(num_neurons):
      
            neur = rates3d[n_run,n_nr,:]
            if np.sum(neur) > 0.1:
                tau_neur1, _ = get_trace_tau(neur, sm_bin = 0)
                tau_cell[n_run, n_nr] = tau_neur1/frame_rate
                
    return tau_net, tau_cell

#%%


def plot_fig_raster(firing_rates_ob, firing_rates_rnn, frame_rate=None, frame_rate_rnn=None):
    # ---- plot example raster ----
    fig, ax = plt.subplots(1,2, figsize=(12,5), layout='constrained')
    fig.set_constrained_layout_pads(w_pad=0.2, h_pad=0.3, hspace=0, wspace=0)
    fig.text(0.015, .89, 'A', fontsize=18)
    fig.text(0.515, .89, 'B', fontsize=18)
    
    num_cells, num_t = firing_rates_ob.shape
    
    if frame_rate is not None:
        x_end = num_t/frame_rate
        x_lab = 'Time (sec)'
    else:
        x_end = num_t
        x_lab = 'Frames'
    
    ax[0].imshow(firing_rates_ob,
               aspect='auto',
               cmap='gist_yarg',
               vmin=0,
               vmax=.5,
               extent=[0, x_end, 1, num_cells],
               interpolation='none')
    ax[0].set_title('CaIm control data')
    ax[0].set_ylabel('Neurons')
    ax[0].set_xlabel(x_lab)
    
    num_cells_rnn, num_t_rnn = firing_rates_rnn.shape
    
    if frame_rate_rnn is not None:
        x_end = num_t_rnn/frame_rate_rnn
        x_lab = 'Time (sec)'
    else:
        x_end = num_t_rnn
        x_lab = 'Frames'

    ax[1].imshow(normalize(firing_rates_rnn),
               aspect='auto',
               cmap='gist_yarg',
               vmin=0,
               vmax=1,
               extent=[0, x_end, 1, num_cells_rnn],
               interpolation='none')
    ax[1].set_title('RNN control test data')
    ax[1].set_ylabel('Neurons')
    ax[1].set_xlabel(x_lab)
    
    return fig

def plot_fig_diag_decoder(dec_data_all, dec_data_all_rnn, training_type, plot_t=None, plot_t_rnn=None, add_sig=True):
    # ---- Plotting diagonal decoder results ----
    # add_sig=True overlays binwise data-vs-shuffle significance (see plot_diag_binwise_dec)
    fig_diag, ax = plt.subplots(1,2, figsize=(12,5), layout='constrained')
    fig_diag.set_constrained_layout_pads(w_pad=0.2, h_pad=0.1, hspace=0, wspace=0)
    fig_diag.text(0.015, .87, 'A', fontsize=18)
    fig_diag.text(0.515, .87, 'B', fontsize=18)
    fig_diag.suptitle('Binwise decoder')

    plot_diag_binwise_dec(
        dec_data_all,
        plot_t=plot_t,
        plot_legend=('Data', 'Shuff'),
        plot_start=-1,                    # plot window start
        plot_end=3,                       # plot window end
        axis = ax[0],
        title_tag='CaIm data',
        add_sig=add_sig)

    plot_diag_binwise_dec(
        np.array(dec_data_all_rnn)[training_type == 'ob trained'],
        plot_t=plot_t_rnn,
        plot_start=-1,
        plot_end=8,
        axis=ax[1],
        title_tag='RNN test data',
        colors = ['limegreen', 'darkgreen'],
        add_sig=add_sig)

    plot_diag_binwise_dec(
        np.array(dec_data_all_rnn)[training_type == 'freq trained'],
        plot_t=plot_t_rnn,
        plot_start=-1,
        plot_end=8,
        axis=ax[1],
        colors = ['orange', 'saddlebrown'],
        add_sig=add_sig)

    # legend from the mean-trace lines only (exclude the '.-' binwise-significance lines)
    main_lines = [l for l in ax[1].get_lines() if l.get_marker() in ('', 'None', None)]
    ax[1].legend(main_lines, ['Oddball trained', 'Oddball trained shuff', 'Freq trained', 'Freq trained shuff'])
    
    return fig_diag

def plot_fig_full_decoder_caim(dec_data_full, plot_t, clim=[0, 0.5]):
    
    # ---- Plotting full decoder results CaIm data ----
    fig_full, ax = plt.subplots(1, 3, gridspec_kw={'width_ratios': [20, 20, 1]}, figsize=(12, 5.7), layout='constrained')
    fig_full.set_constrained_layout_pads(w_pad=0.01, h_pad=0.1, hspace=0, wspace=0)
    fig_full.text(0.01, .88, 'A', fontsize=16)
    fig_full.text(0.47, .88, 'B', fontsize=16)
    fig_full.suptitle('CaIm data full space decoder', fontsize=14)

    plot_full_binwise_dec(
        dec_data_full,
        plot_t=plot_t,
        plot_legend=('Data', 'Shuff'),
        plot_start=-1,
        plot_end=2,
        clim=clim,
        axis=ax,
        title_tag='')
    
    return fig_full

def plot_fig_full_decoder_RNN(dec_data_full_ob, dec_data_full_freq, plot_t):
    
    # ---- Plotting full decoder results RNN data ----
    fig_full, ax = plt.subplots(2, 3, gridspec_kw={'width_ratios': [20, 20, 1]}, figsize=(12,11), layout='constrained')
    fig_full.set_constrained_layout_pads(w_pad=0.05, h_pad=0.1, hspace=0, wspace=0)
    fig_full.text(0.005, .94, 'A', fontsize=16)
    fig_full.text(0.465, .94, 'B', fontsize=16)
    fig_full.text(0.005, .46, 'C', fontsize=16)
    fig_full.text(0.465, .46, 'D', fontsize=16)
    fig_full.suptitle('RNN neurons full space decoder', fontsize=14)

    plot_full_binwise_dec(
        dec_data_full_ob,
        plot_t=plot_t,
        plot_legend=('data', 'shuff'),
        plot_start=-1,
        plot_end=10,
        clim=[0, 1],
        axis=ax[0,:],
        title_tag='RNN Oddball trained')

    plot_full_binwise_dec(
        dec_data_full_freq,
        plot_t=plot_t,
        plot_legend=('data', 'shuff'),
        plot_start=-1,
        plot_end=10,
        clim=[0, 1],
        axis=ax[1,:],
        title_tag='RNN Control freq trained')
    
    return fig_full

def plot_fig_full_decoder_comb(dec_data_full, dec_data_full_ob, dec_data_full_freq, plot_t, plot_t_rnn):
    # ---- plotting full decoder results ----
    fig_full, ax = plt.subplots(3, 3, gridspec_kw={'width_ratios': [10, 10, 1]}, figsize=(15, 12))
    fig_full.text(0.09, .88, 'A', fontsize=16)
    fig_full.text(0.51, .88, 'B', fontsize=16)

    plot_full_binwise_dec(
        dec_data_full,
        plot_t=plot_t,
        plot_legend=('Data', 'Shuff'),
        plot_start=-1,
        plot_end=2,
        clim=[0, 0.5],
        axis=ax[0,:],
        title_tag='CaIm data')


    plot_full_binwise_dec(
        dec_data_full_ob,
        plot_t=plot_t_rnn,
        plot_legend=('Oddball trained', 'Shuff'),
        plot_start=-1,
        plot_end=10,
        clim=[0, 1],
        axis=ax[1,:],
        title_tag='RNN Ob trained')


    plot_full_binwise_dec(
        dec_data_full_freq,
        plot_t=plot_t_rnn,
        plot_legend=('Freq trained', 'Shuff'),
        plot_start=-1,
        plot_end=10,
        clim=[0, 1],
        axis=ax[2,:],
        title_tag='RNN Freq trained')
    
    return fig_full

#%%

def plot_fig_isi_corr_trials(corr_vals, isi_list, colormap='jet', metric_tag = None):
    # trial-to-trial correlation vs ISI: panel A = mean over frequencies, panel B = per-frequency lines; returns (fig, ax, groups)

    fig, ax = plt.subplots(1,2, figsize=(12,5))
    fig.text(0.07, .88, 'A', fontsize=16)
    fig.text(0.51, .88, 'B', fontsize=16)
    
    idx_uq = np.unique(isi_list)
    num_trials = corr_vals.shape[1]
    
    col1 = plt.colormaps[colormap](np.linspace(0, 1, num_trials))
    
    corr_tn_all = np.zeros((num_trials, len(idx_uq)))
    for n_tn in range(num_trials):
        corr_tn = np.full(len(idx_uq), np.nan)
        for n_isi in range(len(idx_uq)):
            idx1 = (idx_uq[n_isi] == np.array(isi_list)).flatten()
            if np.sum(~np.isnan(corr_vals[idx1,n_tn])):
                corr_tn[n_isi] = np.nanmean(corr_vals[idx1,n_tn])
                corr_tn_all[n_tn, n_isi] = np.nanmean(corr_vals[idx1,n_tn])
        if np.sum(~np.isnan(corr_tn)):
            ax[1].plot(idx_uq, corr_tn, '-o', color=col1[n_tn])
    
    if metric_tag is not None:
        ax[1].set_ylabel(metric_tag) 
    else:
        ax[1].set_ylabel('Correlation') 
    ax[1].set_xlabel('ISI duration (sec)')
    freqs = np.logspace(np.log10(2), np.log10(76.9), num_trials)   # kHz, log-spaced 2 -> 76.9 (x1.5/step)
    ax[1].legend(['%g' % round(f, 1) for f in freqs], loc='upper right', title='Freq (kHz)')
    ax[1].set_title('Individual freqs.')
    
    mean1 = np.nanmean(corr_tn_all, axis=0)
    sem1 = np.nanstd(corr_tn_all, axis=0)/np.sqrt(np.sum(~np.isnan(corr_tn_all), axis=0) - 1)
    
    ax[0].errorbar(idx_uq, mean1, sem1, fmt='-o', color='k', capsize=4)

    ax[0].set_ylabel('Correlation')
    ax[0].set_xlabel('ISI duration (sec)')
    ax[0].set_title('Average over freqs.')

    # per-ISI groups for stat_compare; each entry is (data_array, x-position = ISI value).
    # here we include every individual (dataset x frequency) correlation value at each ISI
    # (not the per-dataset mean), so all the individual points enter the test. Pass ax[0], e.g.
    #   sd.stat_compare(ax[0], groups, 'ISI 0.5', 'ISI 1', test='mannwhitney', alternative='two-sided')
    groups = {}
    for isi in idx_uq:
        sel = (np.array(isi_list) == isi).flatten()
        d = corr_vals[sel, :].flatten()              # all datasets x frequencies at this ISI
        groups['ISI %g' % isi] = (d[np.isfinite(d)], float(isi))

    return fig, ax, groups

def plot_fig_isi_corr_trials2(corr_vals, isi_list, colormap='jet', metric_tag=None):
    # nicer version: panel A = individual (dataset x freq) points + mean line + shaded SEM;
    # panel B = per-frequency lines. Returns (fig, ax, groups) like plot_fig_isi_corr_trials.
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    fig.text(0.07, .88, 'A', fontsize=16)
    fig.text(0.51, .88, 'B', fontsize=16)

    idx_uq = np.unique(isi_list)
    num_trials = corr_vals.shape[1]
    isi_arr = np.array(isi_list).flatten()
    rng = np.random.default_rng(0)   # reproducible jitter

    # ---- panel B: per-frequency correlation vs ISI ----
    col1 = plt.colormaps[colormap](np.linspace(0, 1, num_trials))
    corr_tn_all = np.full((num_trials, len(idx_uq)), np.nan)
    for n_tn in range(num_trials):
        for n_isi in range(len(idx_uq)):
            vals = corr_vals[idx_uq[n_isi] == isi_arr, n_tn]
            if np.sum(~np.isnan(vals)):
                corr_tn_all[n_tn, n_isi] = np.nanmean(vals)
        ax[1].plot(idx_uq, corr_tn_all[n_tn, :], '-o', color=col1[n_tn], markersize=4, linewidth=1.2)
    ax[1].set_xlabel('ISI duration (sec)')
    ax[1].set_ylabel(metric_tag if metric_tag is not None else 'Correlation')
    ax[1].set_title('Individual freqs.')
    ax[1].legend([str(n + 1) for n in range(num_trials)], loc='upper right', fontsize=8, ncol=2, frameon=False)

    # ---- panel A: individual points + mean line + shaded SEM ----
    means = np.full(len(idx_uq), np.nan)
    sems = np.full(len(idx_uq), np.nan)
    for n_isi in range(len(idx_uq)):
        pts = corr_vals[idx_uq[n_isi] == isi_arr, :].flatten()
        pts = pts[np.isfinite(pts)]
        if len(pts) == 0:
            continue
        jit = (rng.random(len(pts)) - 0.5) * 0.08
        ax[0].plot(np.full(len(pts), idx_uq[n_isi]) + jit, pts, '.', color='0.7',
                   markersize=4, alpha=0.5, zorder=1)
        means[n_isi] = np.mean(pts)
        sems[n_isi] = np.std(pts) / np.sqrt(len(pts) - 1) if len(pts) > 1 else np.nan

    ax[0].fill_between(idx_uq, means - sems, means + sems, color='steelblue', alpha=0.3, zorder=2)
    ax[0].plot(idx_uq, means, '-o', color='steelblue', markersize=6, linewidth=2, zorder=3)
    ax[0].set_xlabel('ISI duration (sec)')
    ax[0].set_ylabel(metric_tag if metric_tag is not None else 'Correlation')
    ax[0].set_title('Average over freqs.')

    for a in ax:
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)

    # ---- groups (all individual dataset x freq points per ISI) for stat_compare ----
    groups = {}
    for isi in idx_uq:
        d = corr_vals[isi_arr == isi, :].flatten()
        groups['ISI %g' % isi] = (d[np.isfinite(d)], float(isi))

    return fig, ax, groups

def plot_fig_SI_mat(SI_list, isi_uq, title_tag = ''):
    # plot the per-ISI trial-by-trial similarity matrices side by side with a shared colorbar
    fig, ax = plt.subplots(1,len(isi_uq)+1, gridspec_kw={'width_ratios': list(np.ones(len(isi_uq))*10) + [1]}, figsize=(12, 2.8))
    for n_isi in range(len(SI_list)):
        ax1 = ax.flatten()[n_isi]
        im = ax1.imshow(SI_list[n_isi], vmin=0, vmax=.7)
    
        ax1.set_title('isi %s sec' % (isi_uq[n_isi]))
    if title_tag is not None:
        fig.suptitle(title_tag)
    ax.flatten()[0].set_ylabel('trials')
    ax.flatten()[0].set_xlabel('trials')
    fig.colorbar(im, cax=ax.flatten()[-1])
    ax.flatten()[-1].set_ylabel('cosine similarity')
    
    return fig


def plot_fig_tau_networks_comb3(tau_ob_net_all, tau_rnn_net_all, training_type, tau_ob_cell_all, tau_rnn_cell_all, do_log=True):
    # legacy two-panel network/neuron tau figure (superseded by plot_fig_tau_networks_comb)

    fig, ax, = plt.subplots(1, 2, sharey=True, figsize=(12,5))

    fig.text(0.075, .88, 'A', fontsize=16)
    fig.text(0.495, .88, 'B', fontsize=16)

    data_all = [np.array(tau_ob_net_all).flatten()] + tau_rnn_net_all
    labels_all = np.array(['Caim data'] + list(training_type))
    
    plot_int_violin2(data_all,
                       net_labels = labels_all, 
                       axis=ax[0],
                       points=1000,
                       mean_std=True,
                       showmeans=True,
                       showmedians=False,
                       quantile = [0.05, 0.95],
                       colors=['blue', 'green', 'orange', 'gray'],
                       do_log=True)
    
    tau_rnn_cell_all2 = []
    for n_net in range(len(tau_rnn_cell_all)):
        tau_rnn_cell_all2.append(np.nanmean(tau_rnn_cell_all[n_net], axis=0))
        
    data_cell_all = [np.hstack(tau_ob_cell_all).flatten()] + tau_rnn_cell_all2
    
    plot_int_violin2(data_cell_all,
                       net_labels = labels_all, 
                       axis=ax[1],
                       points=1000,
                       mean_std=True,
                       showmeans=True,
                       showmedians=False,
                       quantile = [0.05, 0.95],
                       colors=['blue', 'green', 'orange', 'gray'],
                       do_log=True)
    
    ax[1].yaxis.set_tick_params(labelleft=True)
    ax[0].set_title('Network tau')
    ax[1].set_title('Neuron tau')

    return fig

def get_tau_groups(tau_ob_net_all, tau_rnn_net_all, training_type, tau_ob_cell_all, tau_rnn_cell_all, gap=1.0):
    # build the 8 tau groups (network block + neuron block) shared by the plot and the stats.
    # returns:
    #   groups: dict {label: (data_array, x_position)} e.g. 'CaIm net', 'CaIm neuron', ...
    #   meta:   dict with 'labels', 'x_labels', 'positions', 'colors', 'n_net', 'data'
    tt = np.array(training_type)
    uq_types = list(dict.fromkeys(tt.tolist()))    # RNN types in order of appearance
    short = {'ob trained': 'Ob', 'freq trained': 'Freq', 'untrained': 'Untr'}
    sh = lambda l: short.get(l, l)

    # network tau, one group per dataset type
    net_groups = [np.array(tau_ob_net_all).flatten()]
    net_names = ['CaIm']
    for u in uq_types:
        net_groups.append(np.hstack([np.asarray(tau_rnn_net_all[i]).flatten()
                                     for i in range(len(tau_rnn_net_all)) if tt[i] == u]))
        net_names.append(sh(u))

    # neuron tau (mean over runs per network, then pooled per type)
    tau_rnn_cell_mean = [np.nanmean(tau_rnn_cell_all[n], axis=0) for n in range(len(tau_rnn_cell_all))]
    neuron_groups = [np.hstack(tau_ob_cell_all).flatten()]
    for u in uq_types:
        neuron_groups.append(np.hstack([np.asarray(tau_rnn_cell_mean[i]).flatten()
                                       for i in range(len(tau_rnn_cell_mean)) if tt[i] == u]))

    data_all = [g[np.isfinite(g)] for g in (net_groups + neuron_groups)]
    labels = [n + ' net' for n in net_names] + [n + ' neuron' for n in net_names]
    x_labels = net_names + net_names
    base_colors = ['blue', 'green', 'orange', 'gray']
    colors = base_colors[:len(net_names)] + base_colors[:len(net_names)]

    n_net = len(net_names)
    positions = list(range(n_net)) + [p + n_net + gap for p in range(n_net)]   # e.g. [0,1,2,3, 5,6,7,8]

    groups = {labels[i]: (data_all[i], positions[i]) for i in range(len(labels))}
    meta = {'labels': labels, 'x_labels': x_labels, 'positions': positions,
            'colors': colors, 'n_net': n_net, 'data': data_all}
    return groups, meta


def plot_fig_tau_networks_comb2(tau_ob_net_all, tau_rnn_net_all, training_type, tau_ob_cell_all, tau_rnn_cell_all, do_log=True):
    # all 8 groups (network block + neuron block) on one shared-y axis.
    # returns (fig, ax, groups); pass ax + groups to stat_compare() to add significance brackets.
    groups, meta = get_tau_groups(tau_ob_net_all, tau_rnn_net_all, training_type, tau_ob_cell_all, tau_rnn_cell_all)
    data_all = meta['data']
    positions = meta['positions']
    colors_all = meta['colors']
    x_labels = meta['x_labels']
    n_net = meta['n_net']
    num = len(data_all)

    fig, ax1 = plt.subplots(1, 1, figsize=(12, 5))

    parts = ax1.violinplot(data_all, positions=positions, showmeans=False, showextrema=False,
                          quantiles=[[0.05, 0.95]] * num, points=1000)
    if 'cquantiles' in parts:
        parts['cquantiles'].set_color('k')
    for i in range(num):
        parts['bodies'][i].set_facecolor(colors_all[i])
        parts['bodies'][i].set_edgecolor(colors_all[i])
    for i in range(num):
        y = data_all[i]
        ax1.plot(positions[i], np.mean(y), '_', color='black', mew=2, markersize=30)
        ax1.errorbar(positions[i], np.mean(y), np.std(y), fmt='o', color='black', markersize=4, linewidth=2, capsize=8)

    ax1.set_xticks(positions)
    ax1.set_xticklabels(x_labels, rotation=0)
    if do_log:
        ax1.set_yscale('log')
    ax1.set_ylabel('Tau (sec)')
    ax1.set_title('Network and neuron intrinsic timescales')

    # group underlines + labels beneath (x in data coords, y in axes fraction)
    trans = ax1.get_xaxis_transform()
    net_pos = positions[:n_net]
    neuron_pos = positions[n_net:]
    for xs, name, col in [(net_pos, 'Networks', 'black'), (neuron_pos, 'Neurons', 'red')]:
        ax1.plot([xs[0] - 0.45, xs[-1] + 0.45], [-0.13, -0.13], transform=trans,
                 color=col, lw=4, clip_on=False)
        ax1.text(np.mean(xs), -0.19, name, transform=trans, ha='center', va='top',
                 fontsize=13, fontweight='bold', color=col)

    fig.subplots_adjust(bottom=0.26)

    return fig, ax1, groups


def plot_fig_tau_networks_comb(tau_ob_net_all, tau_rnn_net_all, training_type, tau_ob_cell_all, tau_rnn_cell_all, do_log=True):
    # cleaner version of comb2: violin body + a slim inner boxplot (median + IQR + whiskers),
    # instead of the mean/SD dash + 5/95 quantile lines. Returns (fig, ax, groups) for stat_compare().
    groups, meta = get_tau_groups(tau_ob_net_all, tau_rnn_net_all, training_type, tau_ob_cell_all, tau_rnn_cell_all)
    data_all = meta['data']
    positions = meta['positions']
    colors_all = meta['colors']
    x_labels = meta['x_labels']
    n_net = meta['n_net']
    num = len(data_all)

    fig, ax1 = plt.subplots(1, 1, figsize=(12, 5))

    # violin bodies only (no extrema / quantile lines)
    parts = ax1.violinplot(data_all, positions=positions, showmeans=False, showextrema=False,
                           showmedians=False, points=1000)
    for i in range(num):
        parts['bodies'][i].set_facecolor(colors_all[i])
        parts['bodies'][i].set_edgecolor(colors_all[i])
        parts['bodies'][i].set_alpha(0.55)

    # slim inner boxplot: median + IQR + whiskers, no outliers
    ax1.boxplot(data_all, positions=positions, widths=0.12, showfliers=False, patch_artist=True,
                medianprops=dict(color='black', linewidth=1.5),
                boxprops=dict(facecolor='white', edgecolor='black', linewidth=1.0),
                whiskerprops=dict(color='black', linewidth=1.0),
                capprops=dict(color='black', linewidth=1.0))

    ax1.set_xticks(positions)
    ax1.set_xticklabels(x_labels, rotation=0)
    if do_log:
        ax1.set_yscale('log')
        # plain-number tick labels (0.1, 1, 10, ...) instead of 10^0 exponential form
        ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: ('%g' % v)))
        ax1.yaxis.set_minor_formatter(NullFormatter())
    ax1.set_ylabel('Tau (sec)')
    ax1.set_title('Network and neuron intrinsic timescales')

    # group underlines + labels beneath (x in data coords, y in axes fraction)
    trans = ax1.get_xaxis_transform()
    net_pos = positions[:n_net]
    neuron_pos = positions[n_net:]
    for xs, name, col in [(net_pos, 'Networks', 'black'), (neuron_pos, 'Neurons', 'red')]:
        ax1.plot([xs[0] - 0.45, xs[-1] + 0.45], [-0.13, -0.13], transform=trans,
                 color=col, lw=4, clip_on=False)
        ax1.text(np.mean(xs), -0.19, name, transform=trans, ha='center', va='top',
                 fontsize=13, fontweight='bold', color=col)

    fig.subplots_adjust(bottom=0.26)

    return fig, ax1, groups


def p_to_star(p):
    # p-value to significance stars: *** <0.001, ** <0.01, * <0.05, else n.s.
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'n.s.'


def draw_sig_bracket(ax, x1, x2, p, star=True, color='black', y=None):
    # draw a significance bracket + label between x1 and x2 in the space above the data,
    # extending the y-axis to fit. Stacked calls space evenly (pixel gaps) on log or linear.
    sig = p_to_star(p)
    def _yoff(y_data, pts):
        x0 = ax.get_xlim()[0]
        yd = ax.transData.transform((x0, y_data))[1] + pts * ax.figure.dpi / 72.0
        return ax.transData.inverted().transform((x0, yd))[1]

    step_pts = 15      # spacing between stacked brackets
    margin_pts = 24    # space above the topmost bracket/star and the plot edge
    last = getattr(ax, '_sig_last_y', None)
    if y is not None:
        y_line = y
    elif last is None:
        y_line = _yoff(ax.get_ylim()[1], step_pts)
    else:
        y_line = _yoff(last, step_pts)
    ax._sig_last_y = y_line
    y_tick = _yoff(y_line, -5)
    ax.set_ylim(top=_yoff(y_line, margin_pts))
    ax.plot([x1, x1, x2, x2], [y_tick, y_line, y_line, y_tick], color=color, lw=1.5, clip_on=False)
    # '*' renders high in its box (va='top'); text like 'n.s.'/'p=..' sits low (va='bottom')
    label = sig if star else ('p=%.3g' % p)
    if set(label) <= set('*'):
        va, dy = 'top', 7
    else:
        va, dy = 'bottom', 3
    ax.annotate(label, xy=((x1 + x2) / 2.0, y_line), xytext=(0, dy), textcoords='offset points',
                ha='center', va=va, color=color, fontsize=12, annotation_clip=False)


def stat_compare(ax, groups, name1, name2, test='mannwhitney', alternative='two-sided',
                 stat_file=None, y=None, star=True, color='black'):
    # compare two groups from get_tau_groups(); print result, append to a stats file,
    # and draw a significance bracket on the same panel (ax) between the two groups.
    #   test: 'mannwhitney' (unpaired) or 'wilcoxon' (paired; needs equal-length data)
    #   alternative: 'two-sided', 'greater', or 'less' (name1 vs name2)
    d1, x1 = groups[name1]
    d2, x2 = groups[name2]
    d1 = np.asarray(d1, dtype=float); d1 = d1[np.isfinite(d1)]
    d2 = np.asarray(d2, dtype=float); d2 = d2[np.isfinite(d2)]

    if test in ('wilcoxon', 'paired'):
        if len(d1) != len(d2):
            raise ValueError('wilcoxon needs paired equal-length data (got %d and %d); '
                             'use test="mannwhitney" or pass paired per-dataset values' % (len(d1), len(d2)))
        stat, p = wilcoxon(d1, d2, alternative=alternative)
        n_str = 'n=%d pairs' % len(d1)
    elif test in ('mannwhitney', 'mannwhitneyu', 'mwu'):
        stat, p = mannwhitneyu(d1, d2, alternative=alternative)
        n_str = 'n1=%d; n2=%d' % (len(d1), len(d2))
    elif test in ('ttest', 'ttest_ind', 't'):
        stat, p = ttest_ind(d1, d2, equal_var=False, alternative=alternative)   # Welch's t-test
        n_str = 'n1=%d; n2=%d' % (len(d1), len(d2))
    elif test in ('ttest_rel', 'ttest_paired', 'paired_t'):
        if len(d1) != len(d2):
            raise ValueError('paired t-test needs equal-length data (got %d and %d)' % (len(d1), len(d2)))
        stat, p = ttest_rel(d1, d2, alternative=alternative)
        n_str = 'n=%d pairs' % len(d1)
    else:
        raise ValueError('unknown test: %s (use "mannwhitney", "wilcoxon", "ttest", or "ttest_rel")' % test)

    sig = p_to_star(p)
    print('%s vs %s | %s (%s) | %s | stat=%.4g | p=%.4g | %s'
          % (name1, name2, test, alternative, n_str, stat, p, sig))

    if stat_file is not None:
        new = not os.path.isfile(stat_file)
        with open(stat_file, 'a') as f:
            if new:
                f.write('group1,group2,test,alternative,n,statistic,p_value,significance\n')
            f.write('%s,%s,%s,%s,%s,%.6g,%.6g,%s\n'
                    % (name1, name2, test, alternative, n_str, stat, p, sig))

    # draw the significance bracket on the panel
    draw_sig_bracket(ax, x1, x2, p, star=star, color=color, y=y)

    return {'name1': name1, 'name2': name2, 'test': test, 'alternative': alternative,
            'statistic': stat, 'p': p, 'significance': sig}


def _bh_fdr(p):
    # Benjamini-Hochberg FDR-adjusted p-values (matches MATLAB f_FDR_correction.m)
    p = np.asarray(p, dtype=float)
    n = p.size
    order = np.argsort(p)                                  # ascending
    ranked = p[order] * n / (np.arange(n) + 1)             # p * n / rank
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]     # monotone from the top
    ranked = np.minimum(ranked, 1.0)
    out = np.empty(n)
    out[order] = ranked
    return out


def isi_stats(groups, ax=None, draw='consecutive', color='black', posthoc='tukey'):
    # Stats for the ISI correlation figure: one-way ANOVA across ALL ISI groups + post-hoc.
    # groups: dict {label: (data, x_position)} from plot_fig_isi_corr_trials.
    # posthoc: 'tukey' (Tukey HSD) OR 'fisher_fdr' = MATLAB f_dv_plot_anova1: Fisher LSD using the
    #          pooled ANOVA error MSE, t with df=n1+n2-1, two-tailed, then Benjamini-Hochberg FDR.
    # draw: None, 'consecutive' (0.5-1,1-2,2-4), 'first' (vs first group), or 'all'.
    labels = list(groups)
    data = [np.asarray(groups[k][0], dtype=float) for k in labels]
    data = [d[np.isfinite(d)] for d in data]
    ncat = len(data)

    F, p_anova = f_oneway(*data)
    print('One-way ANOVA across %d groups: F=%.4g, p=%.4g' % (ncat, F, p_anova))

    if posthoc in ('fisher_fdr', 'fisher', 'matlab'):
        from scipy.stats import t as _tdist
        ns = np.array([len(d) for d in data])
        means = np.array([d.mean() for d in data])
        N = int(ns.sum())
        MSE = float(sum(((d - d.mean()) ** 2).sum() for d in data)) / (N - ncat)   # pooled ANOVA error
        pairs = [(i, j) for i in range(ncat) for j in range(i + 1, ncat)]
        praw = np.array([2 * _tdist.sf(abs((means[i] - means[j]) / np.sqrt(MSE / ns[i] + MSE / ns[j])),
                                       ns[i] + ns[j] - 1) for (i, j) in pairs])
        padj = _bh_fdr(praw)
        pmat = np.ones((ncat, ncat))
        for k, (i, j) in enumerate(pairs):
            pmat[i, j] = pmat[j, i] = padj[k]
        method = 'Fisher LSD (pooled MSE) + Benjamini-Hochberg FDR'
    else:
        try:
            from scipy.stats import tukey_hsd
            pmat = np.asarray(tukey_hsd(*data).pvalue)
            method = 'Tukey HSD'
        except Exception:
            m = ncat * (ncat - 1) // 2
            pmat = np.ones((ncat, ncat))
            for i in range(ncat):
                for j in range(i + 1, ncat):
                    _, pp = ttest_ind(data[i], data[j], equal_var=False)
                    pmat[i, j] = pmat[j, i] = min(pp * m, 1.0)
            method = 'pairwise Welch t-test + Bonferroni'

    print('Post-hoc (%s):' % method)
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            print('  %s vs %s: p=%.4g %s' % (labels[i], labels[j], pmat[i, j], p_to_star(pmat[i, j])))

    if ax is not None and draw:
        if draw == 'consecutive':
            pairs = [(i, i + 1) for i in range(len(labels) - 1)]
        elif draw == 'first':
            pairs = [(0, j) for j in range(1, len(labels))]
        elif draw == 'all':
            pairs = [(i, j) for i in range(len(labels)) for j in range(i + 1, len(labels))]
        else:
            pairs = []
        for (i, j) in pairs:
            draw_sig_bracket(ax, groups[labels[i]][1], groups[labels[j]][1], pmat[i, j], color=color)

    return {'labels': labels, 'F': F, 'p_anova': p_anova, 'pvalue': pmat, 'method': method}

def plot_fig_tau_networks(tau_ob_net_all, tau_rnn_net_all, training_type):
    # two-panel network-tau figure (CaIm | RNN by training type) drawn as violins

    fig, ax, = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [1, 3]}, figsize=(6,5))

    fig.text(0.02, .89, 'A', fontsize=16)
    fig.text(0.32, .89, 'B', fontsize=16)
    fig.suptitle('Network Tau')
    
    plot_int_violin2(tau_ob_net_all,
                       net_labels = None, 
                       title_tag = 'CaIm',
                       axis=ax[0],
                       points=1000,
                       mean_std=True,
                       showmeans=True,
                       showmedians=False,
                       quantile = [0.05, 0.95],
                       colors=['blue', 'magenta', 'green'],
                       do_log=True)
    
    
    plot_int_violin2(tau_rnn_net_all,
                       net_labels = training_type, 
                       title_tag = 'RNN',
                       axis=ax[1],
                       points=1000,
                       mean_std=True,
                       showmeans=True,
                       showmedians=False,
                       quantile = [0.05, 0.95],
                       colors=['blue', 'magenta', 'green'],
                       do_log=True)
    
    ax[1].set_ylabel(None)
    
    return fig

def plot_fig_tau_networks2(tau_ob_net_all, tau_rnn_net_all, training_type):
    # single-panel network-tau violins (CaIm data + each RNN training type on one axis)

    fig, ax, = plt.subplots(1, 1, figsize=(6,5))

    fig.text(0.02, .89, 'A', fontsize=16)
    fig.text(0.32, .89, 'B', fontsize=16)
    fig.suptitle('Network Tau')

    data_all = [np.array(tau_ob_net_all).flatten()] + tau_rnn_net_all
    labels_all = np.array(['Caim data'] + list(training_type))
    
    plot_int_violin2(data_all,
                       net_labels = labels_all, 
                       axis=ax,
                       points=1000,
                       mean_std=True,
                       showmeans=True,
                       showmedians=False,
                       quantile = [0.05, 0.95],
                       colors=['blue', 'green', 'orange', 'gray'],
                       do_log=True)
    
    return fig

#%%

def plot_cat_data2(y_data_in, rnn_leg, title_tag = '', do_log=False):
    # scatter of each category's values with a mean +/- std marker

    num_cat = len(y_data_in)
    
    plt.figure()
    ax1 = plt.subplot(111)
    ax1.bar(rnn_leg, np.zeros(num_cat))
    for n_net in range(num_cat):
        
        y_data = np.concatenate(y_data_in[n_net])
        
        x_data = ((np.random.rand(len(y_data)))-0.5)/5+n_net
        
        ax1.plot(x_data, y_data, '.', color='gray')
        ax1.plot(n_net, np.mean(y_data), '_', color='black', mew=2, markersize=40)
        ax1.errorbar(n_net, np.mean(y_data), np.std(y_data), fmt='o', color='black', mew=2, markersize=5, linewidth=2, capsize=10)
    ax1.set_title(title_tag)
    if do_log:
        ax1.set_yscale('log')

def plot_cat_data_violin(y_data_in, rnn_leg, title_tag = '', points=100, mean_std=True, showmeans=False, showmedians=False, quantile = [], colors=[], do_log=False):
    # violin plot of categorical groups, optional mean +/- std overlay and log y-axis

    num_cat = len(y_data_in)
    
    plt.figure()
    ax1 = plt.subplot(111)
    ax1.bar(rnn_leg, np.zeros(num_cat))
    parts = ax1.violinplot(y_data_in, positions=range(num_cat), showmeans=showmeans, showextrema=False, showmedians=showmedians, quantiles=[quantile, quantile, quantile], points=points)
    for key in ['cmeans', 'cmedians', 'cquantiles']:
        if key in parts:
            parts[key].set_color('k')

    if len(colors):
        for n_net in range(num_cat):
            pc = parts['bodies'][n_net]
            pc.set_facecolor(colors[n_net])
            pc.set_edgecolor(colors[n_net])
    if mean_std:
        for n_net in range(num_cat):
            y_data = y_data_in[n_net]
            ax1.plot(n_net, np.mean(y_data), '_', color='black', mew=2, markersize=40)
            ax1.errorbar(n_net, np.mean(y_data), np.std(y_data), fmt='o', color='black', mew=2, markersize=5, linewidth=2, capsize=10)
    ax1.set_title(title_tag)
    if do_log:
        ax1.set_yscale('log')
        
def plot_cat_data_bar(y_data_in, rnn_leg, title_tag = '', do_sem=True, colors=[]):
    # bar plot (mean +/- sem or std) of categorical groups

    num_cat = len(y_data_in)
    
    plt.figure()
    plt.bar(rnn_leg, np.zeros(num_cat))
    for n_net in range(num_cat):
        y_data = y_data_in[n_net]
        if do_sem:
            stds = np.std(y_data)/np.sqrt(len(y_data)-1)
        else:
            stds = np.std(y_data)
        if len(colors):
            plt.bar(n_net, np.mean(y_data), color=colors[n_net], alpha=0.5, edgecolor=colors[n_net])
        else:
            plt.bar(n_net, np.mean(y_data))
        plt.plot(n_net, np.mean(y_data), '_', color='black', mew=2, markersize=40)
        plt.errorbar(n_net, np.mean(y_data), stds, fmt='o', color='black', mew=2, markersize=5, linewidth=2, capsize=10)
    plt.title(title_tag)

def plot_int_violin2(tau_net_list, net_labels = None, data_lab = ['CaIm data'], title_tag = '', axis=None, points=100, mean_std=True, showmeans=False, showmedians=False, quantile = [0.05, 0.95], colors=['blue', 'green', 'orange', 'gray'], do_log=False):
    # violin plot of tau distributions grouped by net_labels, with optional mean +/- std overlay and log y-axis
    
    num_net = len(tau_net_list)
    
    net_type_list = []
    net_idx = np.zeros(num_net, dtype=int)
    rnn_leg = []
    quantiles_all = []
    if net_labels is not None:
        _, idx1 = np.unique(net_labels, return_index=True)
        idx1.sort()
        net_uq = net_labels[idx1]
        for n_net in range(len(net_uq)):
            net_idx[net_labels == net_uq[n_net]] = n_net
            temp_bin = []
            for n_net2 in range(num_net):
                if net_uq[n_net] == net_labels[n_net2]:
                    temp_bin.append(tau_net_list[n_net2].flatten())

            net_type_list.append(np.hstack(temp_bin).flatten())
            rnn_leg.append(net_uq[n_net].capitalize())
            quantiles_all.append(quantile)
            
    else:
        net_uq = data_lab
        net_type_list.append(np.array(tau_net_list).flatten())
        quantiles_all.append(quantile)
        rnn_leg = data_lab
        
            
    num_net_types = len(net_uq)
    
    if axis is None:
        plt.figure()
        ax1 = plt.subplot(111)
    else:
        ax1 = axis
        
    ax1.bar(rnn_leg, np.zeros(num_net_types))
    parts = ax1.violinplot(net_type_list, positions=range(num_net_types), showmeans=showmeans, showextrema=False, showmedians=showmedians, quantiles=quantiles_all, points=points)
    for key in ['cmeans', 'cmedians', 'cquantiles']:
        if key in parts:
            parts[key].set_color('k')

    if len(colors):
        for n_net in range(num_net_types):
            pc = parts['bodies'][n_net]
            pc.set_facecolor(colors[n_net])
            pc.set_edgecolor(colors[n_net])
    if mean_std:
        for n_net in range(num_net_types):
            y_data = net_type_list[n_net]
            ax1.plot(n_net, np.mean(y_data), '_', color='black', mew=2, markersize=40)
            ax1.errorbar(n_net, np.mean(y_data), np.std(y_data), fmt='o', color='black', mew=2, markersize=5, linewidth=2, capsize=10)
    ax1.set_title(title_tag)
    
    if do_log:
        ax1.set_yscale('log')
        ax1.set_ylabel('Tau (sec)')
    else:
        ax1.set_ylabel('Tau (sec)')
    

