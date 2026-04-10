# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 13:32:23 2025

@author: ys2605
"""

import os
import h5py
import numpy as np
from scipy.stats import norm
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.signal import correlate #, correlation_lags

import matplotlib.pyplot as plt
from datetime import datetime

#%%
def f_get_fnames_from_dir(data_dir, ext_list = [], tags = None):

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

def f_load_caim_data_mat(data_dir, ext_list = [], tags = None, num_files=None, data_tag = 'results_cnmf_sort.mat', proc_tag = 'processed_data.mat', deconvolution='oasis', smooth_std_duration=0.1):
    
    # deconvolution methods are either oasis (caiman default) or smoothdftd - smoothed, rectified first derivative
    # smooth_std_duration in sec
    
    
    flist = f_get_fnames_from_dir(data_dir, ext_list=ext_list, tags=tags)
    
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
        
        flist_data = f_get_fnames_from_dir(data_dir, ext_list = ['.mat'], tags = [fname_core, data_tag])
        flist_proc = f_get_fnames_from_dir(data_dir, ext_list = ['.mat'], tags = [fname_core, proc_tag])
        
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
                    firing_rates_cut = f_smooth_dfdt(ca_traces_cut, sigma_frames=1000/data_slice['volume_period']*0.1, do_smooth=True)
                
                firing_rates_cut = f_gauss_smooth(firing_rates_cut, sigma_frames=1000/data_slice['volume_period']*smooth_std_duration)
                
                peak_rate = np.max(firing_rates_cut, axis=1)[:,None]
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
        
    return data_out

def f_h5_load_group(group, keys=None):
    if keys is None:
        keys = group.keys()
        
    data = {}
    for key1 in keys:
        data[key1] = group[key1][()]
    
    return data

def f_get_values(data_out, key):
    
    values = []
    
    for data_slice in data_out:
        if key in data_slice.keys():
            values.append(data_slice[key])
            
    return values

def f_get_mouse_id(data_echo):
    
    fnames = f_get_values(data_echo, 'fname_core')
    
    mouse_id = []
    for fname in fnames:
        mouse_id.append(fname[:fname.find('_')])
        
    return mouse_id

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

def f_normalize(rates):
    # assumes neurons x time
    
    rates_n = rates - np.min(rates, axis=1)[:, None]
    
    # ignoring cells that are always zero
    max_rates = np.max(rates_n, axis=1)
    has_max = max_rates > 0
    
    rates_n = rates_n[has_max,:] / max_rates[has_max][:,None]
    
    return rates_n
    

def f_get_frames(trial_win = [-0.05, .95], frame_rate = 30):
    # anchor at 0
    frame_start = np.ceil(trial_win[0] * frame_rate)
    frame_end = np.ceil(trial_win[1] * frame_rate)
    trial_frames = [int(frame_start), int(frame_end)]
    plot_t = np.round(np.arange(frame_start/frame_rate, frame_end/frame_rate, 1/frame_rate), decimals=4)
    return trial_frames, plot_t

def f_get_stim_trig_resp(firing_rates, stim_times, trial_frames = [-29, 85]):
    # input: cells x time
    
    num_cells, T = firing_rates.shape
    num_trials = len(stim_times)
    
    win_size = trial_frames[1] - trial_frames[0]
    
    stim_trig_resp = np.zeros((num_cells, win_size, num_trials))
    
    for n_tr in range(num_trials):
        cur_frame = round(stim_times[n_tr]-1) # correct for matlab to python 
        
        start1 = (cur_frame+trial_frames[0])
        end1 = np.min([cur_frame+trial_frames[1], T])
        
        stim_trig_resp[:,:end1-start1,n_tr] = firing_rates[:,start1:end1]
    
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


def f_compute_correlation(stim_trig_resp, trial_types, trials_analyze, resp_cells=None, min_resp_cells=5, subtract_mean=False, add_noise_sigma=1e-5, metric='correlation'):
    
    corr_vals = np.full((len(trials_analyze)), np.nan)
    
    if subtract_mean:
        stim_trig_resp = stim_trig_resp - np.mean(stim_trig_resp)

    if add_noise_sigma:   # add some uncorrelated noise to add stability
        stim_trig_resp = stim_trig_resp + np.random.normal(0, add_noise_sigma, size=stim_trig_resp.shape)
    
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
            if resp_cells is not None:
                stim_trig_resp3 = stim_trig_resp2[resp_marg,:,:]
            else:
                stim_trig_resp3 = stim_trig_resp2
            stim_trig_resp4 = np.mean(stim_trig_resp3, axis=1)
            
            act_idx = np.mean(stim_trig_resp4, axis =1) > 0
            
            stim_trig_resp5 = stim_trig_resp4[act_idx,:]
            
            distances = squareform(pdist(stim_trig_resp5.T, metric=metric))     # cosine, correlation
            SI = 1 - distances
            SI2 = np.tril(SI, k=-1)
            corr_vals[n_tn] = np.mean(SI2[SI2.astype(bool)])
            
            # if 0:
            #     if tn1==4:
            #         plt.figure()
            #         plt.imshow(SI)
            #         plt.title('isi = ' + str(data_echo[n_fl]['isi']))
                    
    return corr_vals
        
def f_compute_correlation_mat(stim_trig_resp, subtract_mean=False, add_noise_sigma=1e-5, metric='correlation'):
    
    if subtract_mean:
        stim_trig_resp = stim_trig_resp - np.mean(stim_trig_resp)

    if add_noise_sigma:   # add some uncorrelated noise to add stability
        stim_trig_resp = stim_trig_resp + np.random.normal(0, add_noise_sigma, size=stim_trig_resp.shape)
    
    stim_trig_resp2 = np.mean(stim_trig_resp, axis=1)
    
    distances = squareform(pdist(stim_trig_resp2.T, metric=metric))     # cosine, correlation
    SI = 1 - distances
    
    return SI

#%%
def f_get_network_distance(firing_rates):

    base_pop_vec = np.mean(firing_rates, axis=1)
    tr_dist = cdist(np.reshape(base_pop_vec, (1,len(base_pop_vec))), firing_rates.T, 'euclidean')[0]

    return tr_dist

def f_get_trace_tau(trace, sm_bin = 0):
    
    #sm_bin = 10#round(1/params['dt'])*50;
    #trial_len = out_temp_all.shape[1]
    
    
    tracen = trace - np.mean(trace)
    tracen = tracen/np.std(tracen)
    
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
    
    tau_corr = np.where(corr1_smn2 < 0.5)[0][0]
    
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

def f_load_rnn_test(data_dir, fname_data, fname_params, max_net_load = 999, limit_network_types=[], flatten_runs = False, max_trial_types = 10, max_trials = 500, cut_zero_trials = False, num_initial_trials_skip = 0):
    # input is the spectrogram inputs
    # target is the index when oddball trial happens
    # loaded rates shape is (time, run, neurons) - inside converted to 
    # output rates are converted to (runs, neurons, time)
    # flatten runs makes output rates 2D (neurons, time)
    # max trial types and max trials only apply during flattened runs
    
    
    test_data_load = np.load(data_dir + fname_data, allow_pickle=True).item()
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
    
    deets_load = np.load(data_dir + fname_params, allow_pickle=True).item()
    params_all = deets_load['params_all']
    params_test = deets_load['params_test']
    # deets_load['ob_data'].keys()
    # del deets_load['ob_data']['input_oddball']
    # del deets_load['ob_data']['target_oddball_freq']
    # del deets_load['cont_data']['input_control']
    # del deets_load['cont_data']['target_control']
    # np.save(data_dir + 'RNN_test_data_2024_5_24_9h_42m2_params.npy', deets_load)
     
    data_all = []
    
    use_net_type_lim = False
    for n_net in range(len(test_data_load_all)):
        if deets_load['rnn_leg'][n_net] in deets_load['rnn_leg']:
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
                            stim_on_tace = 1 - deets_load['ob_data']['target_oddball_ctx3'][:,n_run,0]
                        else:
                            stim_trace = np.max(net1['input'][:,n_run], axis=1)
                            stim_trace_n = stim_trace - np.percentile(stim_trace, 20)
                            stim_trace_n = stim_trace_n / np.max(stim_trace_n)
                            stim_on_tace = (stim_trace_n > 0.5).astype(int)
        
                        stim_onset_trace = np.diff(stim_on_tace, prepend=[0]) > 0
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
                        rates3d = net1['rates']
                        rates4d = np.reshape(rates3d, (round(T/trial_len), trial_len, num_runs, num_neurons), order='C')
                        
                        if cut_zero_trials:
                            num_skip = params_test['num_prepend_zeros'] + num_initial_trials_skip
                        else:
                            num_skip = num_initial_trials_skip
                        
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
                            trial_idx = np.random.choice(sel_trials, size=np.min([max_trials, num_stim_all2]), replace=False)
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
            
    return data_all
    
#%%

def f_plot_diag_binwise_dec(dec_data_list, plot_t=None, plot_legend=None, plot_start=-1, plot_end=5, axis=None, title_tag='', colors = ['blue', 'black']):
    
    
    if type(dec_data_list) is not list:
        if type(dec_data_list) is not np.ndarray:
            dec_data_list = [dec_data_list]
    
    plt_start2 = np.argmin(np.abs(plot_start - plot_t))
    plt_end2 = np.argmin(np.abs(plot_end - plot_t))
    
    plot_t2 = plot_t[plt_start2:plt_end2]
    
    # label='Alpha 0.6

    num_dsets = len(dec_data_list)
    
    num_t, _, num_dec = dec_data_list[0]['performance'].shape
    traces_all = np.zeros((num_dsets, num_dec, len(plot_t2)))
    
    for n_d in range(num_dsets):
        perform_train_test = dec_data_list[n_d]['performance']
        num_t, _, num_dec = perform_train_test.shape
        for n_dec in range(num_dec):
            traces_all[n_d, n_dec, :] = perform_train_test[plt_start2:plt_end2,:,n_dec].flatten()
            
    if axis is None:
        fig1, axis = plt.subplots()
    
    leg_lines = []
    for n_dec in range(num_dec):
        mean_trace = np.mean(traces_all[:, n_dec, :], axis=0)
        sem_trace = np.std(traces_all[:, n_dec, :], axis=0)/np.max([np.sqrt(num_dsets-1), 1])
        l1, = axis.plot(plot_t2, mean_trace, color=colors[n_dec])
        leg_lines.append(l1)
        axis.fill_between(plot_t2, mean_trace-sem_trace, mean_trace+sem_trace, color=colors[n_dec], alpha=0.2)
    
    if plot_legend is not None:
        axis.legend(handles=leg_lines, labels=plot_legend)
    axis.set_xlabel('Time (sec)')
    axis.set_ylabel('Performance')
    if len(title_tag):
        axis.set_title(title_tag)
           
    return axis

def f_plot_full_binwise_dec(dec_data, plot_t=None, plot_legend=None, plot_start=-1, plot_end=5, axis=None, title_tag='', clim=[0, 1], clim_width_ratio = 10, figsize=(17, 3.5)):
    
    plt_start2 = np.argmin(np.abs(plot_start - plot_t))
    plt_end2 = np.argmin(np.abs(plot_end - plot_t))
    
    plot_t2 = plot_t[plt_start2:plt_end2]
    
    # label='Alpha 0.6

    perform_train_test = dec_data['performance']
    num_t, _, num_dec = perform_train_test.shape
    
    if axis is None:
        fig, ax1 = plt.subplots(1, num_dec+1, gridspec_kw={'width_ratios': list(np.ones(num_dec)*clim_width_ratio) + [1]}, figsize=figsize)
    else:
        ax1 = axis
    
    for n_dec in range(num_dec):
        perform_train_test[plt_start2:plt_end2,plt_start2:plt_end2,n_dec]
    
    for n_dec in range(num_dec):
        
        im1 = ax1[n_dec].imshow(perform_train_test[plt_start2:plt_end2,plt_start2:plt_end2,n_dec], extent=(np.min(plot_t2), np.max(plot_t2), np.max(plot_t2), np.min(plot_t2)), clim=clim)
        #ax1.set_clim([0, 1])
        ax1[n_dec].set_xlabel('Test time (sec)')
        ax1[n_dec].set_ylabel('Train time (sec)')
        if len(title_tag):
            ax1[n_dec].set_title('%s %s' % (title_tag, if_get_leg(dec_data, plot_legend)[n_dec]))
        else:
            axis[n_dec].set_title(plot_legend[n_dec])
    if len(ax1) > num_dec:
        plt.colorbar(im1, cax=axis.flatten()[-1])
        ax1.flatten()[-1].set_ylabel('Performance')
        
        #cbar = fig.colorbar(im1)
        #cbar.set_label('Performance')   
    if axis is None:
        return fig
    
def if_get_leg(dec, f_leg):
    if 'legend' in dec:
        leg = dec['legend']
    elif f_leg is not None:
        leg = f_leg
    
    return leg

def f_get_network_intrinsic_timescales(firing_rates, frame_rate):
    
    if len(firing_rates.shape) == 2:
        rates3d = firing_rates[None,:,:]
    else:
        rates3d = firing_rates
        
    num_runs, num_neurons, _ = rates3d.shape
    
    tau_net = np.zeros(num_runs)
    tau_cell = np.full(shape=(num_runs, num_neurons), fill_value=np.nan)
    
    for n_run in range(num_runs):
        
        tr_dist = f_get_network_distance(rates3d[n_run,:,:])
        
        tau_net1, _ = f_get_trace_tau(tr_dist, sm_bin = 0)
        tau_net[n_run] = tau_net1/frame_rate
        
        for n_nr in range(num_neurons):
      
            neur = rates3d[n_run,n_nr,:]
            if np.sum(neur) > 0.1:
                tau_neur1, _ = f_get_trace_tau(neur, sm_bin = 0)
                tau_cell[n_run, n_nr] = tau_neur1/frame_rate
                
    return tau_net, tau_cell

#%%


def f_plot_fig_raster(firing_rates_ob, firing_rates_rnn, frame_rate=None, frame_rate_rnn=None):
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

    ax[1].imshow(f_normalize(firing_rates_rnn),
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

def f_plot_fig_diag_decoder(dec_data_all, dec_data_all_rnn, training_type, plot_t=None, plot_t_rnn=None):
    # ---- Plotting diagonal decoder results ----
    fig_diag, ax = plt.subplots(1,2, figsize=(12,5), layout='constrained')
    fig_diag.set_constrained_layout_pads(w_pad=0.2, h_pad=0.1, hspace=0, wspace=0)
    fig_diag.text(0.015, .87, 'A', fontsize=18)
    fig_diag.text(0.515, .87, 'B', fontsize=18)
    fig_diag.suptitle('Binwise decoder')

    f_plot_diag_binwise_dec(
        dec_data_all,
        plot_t=plot_t,
        plot_legend=('Data', 'Shuff'),
        plot_start=-1,                    # plot window start
        plot_end=3,                       # plot window end
        axis = ax[0],
        title_tag='CaIm data')

    f_plot_diag_binwise_dec(
        np.array(dec_data_all_rnn)[training_type == 'ob trained'],
        plot_t=plot_t_rnn,
        plot_start=-1,
        plot_end=8,
        axis=ax[1],
        title_tag='RNN test data',
        colors = ['limegreen', 'darkgreen'])

    f_plot_diag_binwise_dec(
        np.array(dec_data_all_rnn)[training_type == 'freq trained'],
        plot_t=plot_t_rnn,
        plot_start=-1,
        plot_end=8,
        axis=ax[1],
        colors = ['orange', 'saddlebrown'])

    ax[1].legend(ax[1].get_lines(), ['Oddball trained', 'Oddball trained shuff', 'Freq trained', 'Freq trained shuff'])
    
    return fig_diag

def f_plot_fig_full_decoder_caim(dec_data_full, plot_t, clim=[0, 0.5]):
    
    # ---- Rlotting full decoder results CaIm data ----
    fig_full, ax = plt.subplots(1, 3, gridspec_kw={'width_ratios': [20, 20, 1]}, figsize=(12, 5.7), layout='constrained')
    fig_full.set_constrained_layout_pads(w_pad=0.01, h_pad=0.1, hspace=0, wspace=0)
    fig_full.text(0.01, .88, 'A', fontsize=16)
    fig_full.text(0.47, .88, 'B', fontsize=16)
    fig_full.suptitle('CaIm data full space decoder', fontsize=14)

    f_plot_full_binwise_dec(
        dec_data_full,
        plot_t=plot_t,
        plot_legend=('Data', 'Shuff'),
        plot_start=-1,
        plot_end=2,
        clim=clim,
        axis=ax,
        title_tag='')
    
    return fig_full

def f_plot_fig_full_decoder_RNN(dec_data_full_ob, dec_data_full_freq, plot_t):
    
    # ---- Rlotting full decoder results RNN data ----
    fig_full, ax = plt.subplots(2, 3, gridspec_kw={'width_ratios': [20, 20, 1]}, figsize=(12,11), layout='constrained')
    fig_full.set_constrained_layout_pads(w_pad=0.05, h_pad=0.1, hspace=0, wspace=0)
    fig_full.text(0.005, .94, 'A', fontsize=16)
    fig_full.text(0.465, .94, 'B', fontsize=16)
    fig_full.text(0.005, .46, 'C', fontsize=16)
    fig_full.text(0.465, .46, 'D', fontsize=16)
    fig_full.suptitle('RNN neurons full space decoder', fontsize=14)

    f_plot_full_binwise_dec(
        dec_data_full_ob,
        plot_t=plot_t,
        plot_legend=('data', 'shuff'),
        plot_start=-1,
        plot_end=10,
        clim=[0, 1],
        axis=ax[0,:],
        title_tag='RNN Oddball trained')

    f_plot_full_binwise_dec(
        dec_data_full_freq,
        plot_t=plot_t,
        plot_legend=('data', 'shuff'),
        plot_start=-1,
        plot_end=10,
        clim=[0, 1],
        axis=ax[1,:],
        title_tag='RNN Control freq trained')
    
    return fig_full

def f_plot_fig_full_decoder_comb(dec_data_full, dec_data_full_ob, dec_data_full_freq, plot_t, plot_t_rnn):
    # ---- plotting full decoder results ----
    fig_full, ax = plt.subplots(3, 3, gridspec_kw={'width_ratios': [10, 10, 1]}, figsize=(15, 12))
    fig_full.text(0.09, .88, 'A', fontsize=16)
    fig_full.text(0.51, .88, 'B', fontsize=16)

    f_plot_full_binwise_dec(
        dec_data_full,
        plot_t=plot_t,
        plot_legend=('Data', 'Shuff'),
        plot_start=-1,
        plot_end=2,
        clim=[0, 0.5],
        axis=ax[0,:],
        title_tag='CaIm data')


    f_plot_full_binwise_dec(
        dec_data_full_ob,
        plot_t=plot_t_rnn,
        plot_legend=('Oddball trained', 'Shuff'),
        plot_start=-1,
        plot_end=10,
        clim=[0, 1],
        axis=ax[1,:],
        title_tag='RNN Ob trained')


    f_plot_full_binwise_dec(
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

def f_plot_fig_isi_corr_trials(corr_vals, isi_list, colormap='jet', metric_tag = None):
    
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
    ax[1].legend([str(n+1) for n in range(10)], loc='upper right')
    ax[1].set_title('Individual freqs.')
    
    mean1 = np.nanmean(corr_tn_all, axis=0)
    sem1 = np.nanstd(corr_tn_all, axis=0)/np.sqrt(np.sum(~np.isnan(corr_tn_all), axis=0) - 1)
    
    ax[0].plot(idx_uq, mean1, '-o', color='k')
    ax[0].errorbar(idx_uq, mean1, sem1, color='k')
    
    ax[0].set_ylabel('Correlation')  
    ax[0].set_xlabel('ISI duration (sec)')
    ax[0].set_title('Average over freqs.')
    
    return fig

def f_plot_fig_SI_mat(SI_list, isi_uq, title_tag = ''):
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


def f_plot_fig_tau_networks_comb(tau_ob_net_all, tau_rnn_net_all, training_type, tau_ob_cell_all, tau_rnn_cell_all, do_log=True):
    
    fig, ax, = plt.subplots(1, 2, sharey=True, figsize=(12,5))

    fig.text(0.075, .88, 'A', fontsize=16)
    fig.text(0.495, .88, 'B', fontsize=16)

    data_all = [np.array(tau_ob_net_all).flatten()] + tau_rnn_net_all
    labels_all = np.array(['Caim data'] + list(training_type))
    
    f_plot_int_violin2(data_all,
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
    
    f_plot_int_violin2(data_cell_all,
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

def f_plot_fig_tau_networks(tau_ob_net_all, tau_rnn_net_all, training_type):
    
    fig, ax, = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [1, 3]}, figsize=(6,5))

    fig.text(0.02, .89, 'A', fontsize=16)
    fig.text(0.32, .89, 'B', fontsize=16)
    fig.suptitle('Network Tau')
    
    f_plot_int_violin2(tau_ob_net_all,
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
    
    
    f_plot_int_violin2(tau_rnn_net_all,
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

def f_plot_fig_tau_networks2(tau_ob_net_all, tau_rnn_net_all, training_type):
    
    fig, ax, = plt.subplots(1, 1, figsize=(6,5))

    fig.text(0.02, .89, 'A', fontsize=16)
    fig.text(0.32, .89, 'B', fontsize=16)
    fig.suptitle('Network Tau')

    data_all = [np.array(tau_ob_net_all).flatten()] + tau_rnn_net_all
    labels_all = np.array(['Caim data'] + list(training_type))
    
    f_plot_int_violin2(data_all,
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

def f_plot_cat_data2(y_data_in, rnn_leg, title_tag = '', do_log=False):
    
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

def f_plot_cat_data_violin(y_data_in, rnn_leg, title_tag = '', points=100, mean_std=True, showmeans=False, showmedians=False, quantile = [], colors=[], do_log=False):
    
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
        
def f_plot_cat_data_bar(y_data_in, rnn_leg, title_tag = '', do_sem=True, colors=[]):
    
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

def f_plot_int_violin2(tau_net_list, net_labels = None, data_lab = ['CaIm data'], title_tag = '', axis=None, points=100, mean_std=True, showmeans=False, showmedians=False, quantile = [0.05, 0.95], colors=['blue', 'green', 'orange', 'gray'], do_log=False):
    
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
        ax1.set_ylabel('Log tau (sec)')
    else:
        ax1.set_ylabel('Tau (sec)')
    

