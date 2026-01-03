# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 13:20:25 2025

@author: ys2605
"""

import numpy as np

import time

from sklearn import svm
from sklearn.metrics import accuracy_score
from scipy import linalg

import matplotlib.pyplot as plt
#from matplotlib import gridspec
from matplotlib.patches import Rectangle

#%% decoder make cross validation groups

def f_make_cv_groups(num_trials, num_cv_groups):
    
    cv_tr_idx = np.arange(num_trials)
    np.random.shuffle(cv_tr_idx)  
    
    test_groups = np.zeros((num_cv_groups, num_trials), dtype=bool)
    
    for n_cv in range(num_cv_groups):
    
        start1 = np.floor(n_cv*(num_trials/num_cv_groups)).astype(int)
        end1 = np.floor((n_cv+1)*(num_trials/num_cv_groups)).astype(int)
        
        test_groups[n_cv][cv_tr_idx[start1:end1]] = 1
    
    return test_groups

#%%
def f_sample_trial_data_dec(rates_in, stim_loc, trial_types):
    
    num_dec = len(rates_in)
    num_tt = len(trial_types)
    
    # fiirst sample stim
    rates_out = []
    
    for n_dec in range(num_dec):
        
        trial_len, num_trials, num_batch, num_neurons = rates_in[n_dec].shape
        
        dd_red_samp_rates = np.zeros((trial_len, num_tt, num_batch, num_neurons))
        has_data = np.ones((num_batch), dtype=bool)
        
        for n_run in range(num_batch):
            
            run_data1 = stim_loc[:,n_run]
            
            tt_loc_all = np.zeros(num_tt, dtype=int)
            for n_tt in range(num_tt):
                tt_loc = np.where(run_data1 == trial_types[n_tt])[0]
                if tt_loc.shape[0]:
                    tt_loc_all[n_tt] = np.random.choice(tt_loc, 1)[0]
                else:
                    has_data[n_run] = 0
    
            if has_data[n_run]:
                for n_tt in range(num_tt):
                    dd_red_samp_rates[:, n_tt, n_run, :] = rates_in[n_dec][:, tt_loc_all[n_tt], n_run, :]
            
        
        dd_red_samp_rates2 = dd_red_samp_rates[:,:,has_data,:]
        dd_red_samp_rates3 = np.reshape(dd_red_samp_rates2, (trial_len, num_tt*num_batch, num_neurons), order='F')
        rates_out.append(dd_red_samp_rates3)
    
    # ctreate y
    num_train = np.sum(has_data)
    
    y_data_templ = np.reshape(np.arange(num_tt)+1, (num_tt,1), order='F')
    y_data = np.repeat(y_data_templ, num_train, axis=1)
    y_data2 = np.reshape(y_data, (num_tt*num_batch), order='F')
    
    return rates_out, y_data2

#%%

def f_run_binwise_dec(X_all, Y_all, train_test_method='diag', pca_var_frac=1, num_cv=5, fixed_time=0, add_noise_sigma=0.1, normalize = True, get_train_coeffs=False, log=False):
    num_dset = len(X_all)
    
    num_cells, num_t_bins, num_trials = X_all[0].shape
    
    if train_test_method == 'full':
        train_t_idx = np.arange(num_t_bins)
        test_t_idx = np.repeat(np.arange(num_t_bins).reshape((1,num_t_bins)), num_t_bins, axis=0)
    elif train_test_method == 'diag':
        train_t_idx = np.arange(num_t_bins)
        test_t_idx = np.arange(num_t_bins).reshape((num_t_bins,1))
    elif train_test_method == 'fixed_train_time':
        train_t_idx = (np.ones(1) * fixed_time).astype(int)
        test_t_idx = np.arange(num_t_bins)
    elif train_test_method == 'fixed_test_time':
        train_t_idx = np.arange(num_t_bins)
        test_t_idx = (np.ones(num_t_bins) * fixed_time).astype(int)

    perform_train_test1 = np.zeros((train_t_idx.shape[0], test_t_idx[0].shape[0], num_cv, num_dset))
    
    if get_train_coeffs:
        n_classes = len(np.unique(Y_all))
        num_ovo_dec = int(n_classes * (n_classes-1) / 2)
        train_coeffs = np.zeros((num_cells, train_t_idx.shape[0], num_cv, num_ovo_dec, num_dset))
        train_intercepts = np.zeros((train_t_idx.shape[0], num_cv, num_ovo_dec, num_dset))
    
    start_t = time.time()

    for n_dset in range(num_dset):
        X_use = X_all[n_dset]
        
        if add_noise_sigma:
            X_use = X_use + np.random.normal(0, add_noise_sigma, size=X_use.shape)
        
        if normalize:
            for n_cell in range(num_cells):
                X_use1 = X_use[n_cell,:,:]
                X_use2 = X_use1 - np.mean(X_use1)
                X_use[n_cell,:,:] = X_use2/np.std(X_use2)

        if pca_var_frac < 1 and pca_var_frac !=0:
            
            X_use2d = np.reshape(X_use, (num_cells, num_t_bins*num_trials), order = 'F')
            
            U, S, Vh = linalg.svd(X_use2d.T, full_matrices=False)
            cum_var = np.cumsum(S**2/np.sum(S**2))
            comp_include_idx = np.where(cum_var > pca_var_frac)[0][0]+1
            #X_rec = np.dot(U, Vh*S[:, None])
            X_LD2d = np.dot(X_use2d.T, Vh.T).T
            X_use2 = np.reshape(X_LD2d[:comp_include_idx,:], (comp_include_idx, num_t_bins, num_trials), order = 'F')
        else:
            X_use2 = X_use
            
        y_data2 = Y_all[n_dset]
        
        for n_bin in range(train_t_idx.shape[0]): # 

            if log:
                print('dset %d of %d; train bin %d/%d' % (n_dset+1, num_dset, n_bin, train_t_idx.shape[0]))    
        
            n_bin2 = train_t_idx[n_bin]
            
            X_train = X_use2[:,n_bin2,:]
            
            cv_groups = f_make_cv_groups(num_trials, num_cv)
            
            for n_cv in range(num_cv):
                test_idx = cv_groups[n_cv]
                train_idx = ~test_idx
                
                X_train2 = X_train[:,train_idx]
                
                svc = svm.SVC(kernel='linear', C=1,gamma='auto') # ,verbose=True , probability=True
                svc.fit(X_train2.T, y_data2[train_idx])
                
                if get_train_coeffs:
                    train_coeffs[:,n_bin,n_cv,:,n_dset] = svc.coef_.T 
                    train_intercepts[n_bin,n_cv,:,n_dset] = svc.intercept_
                
                for n_test in range(test_t_idx[n_bin].shape[0]):
                    n_test2 = test_t_idx[n_bin][n_test]
                    X_test = X_use2[:,n_test2,:]
                    
                    test_pred1 = svc.predict(X_test[:,test_idx].T)
                    perform_train_test1[n_bin, n_test, n_cv, n_dset] = accuracy_score(y_data2[test_idx], test_pred1)
                    # perform_train_test1[n_bin, n_test, n_cv, n_dset] = np.sum(y_data2[test_idx] == test_pred1)/test_pred1.shape[0]
                    # perform_train_test1[n_bin, n_test, n_cv, n_dset] = svc.score(X_test[:,test_idx].T, y_data2[test_idx])
                    if 0:
                        plt.figure()
                        plt.imshow(svc.coef_)
                        
                        plt.figure()
                        plt.plot(svc.coef_.T)
                        
                        plt.figure()
                        plt.plot(svc.coef_[:,15])
                        
                        # decision using svc
                        # X = X_test[:,test_idx].T
                        # X = svc._validate_for_predict(X)
                        # X = svc._compute_kernel(X)
                        # adec = svc._dense_decision_function(X)
                        
                        # decisison using dot prod
                        # adec = np.dot(X_test[:,test_idx].T, svc.coef_.T) + svc.intercept_
                        
                        # 45 is the number of decoders created to do multiclass classification of 10 classes
                        # first is class 1 vs class 2
                        # n_samples, _ =adec.shape
                        # predictions = adec < 0
                        # confidences = -adec
                        # n_classes = 10
                        # votes = np.zeros((n_samples, n_classes))
                        # sum_of_confidences = np.zeros((n_samples, n_classes))
                        # k = 0
                        # for i in range(n_classes):
                        #    for j in range(i + 1, n_classes):
                        #        sum_of_confidences[:, i] -= confidences[:, k]
                        #        sum_of_confidences[:, j] += confidences[:, k]
                        #        votes[predictions[:, k] == 0, i] += 1
                        #        votes[predictions[:, k] == 1, j] += 1
                        #        k+=1
                        # transformed_confidences = sum_of_confidences / (3 * (np.abs(sum_of_confidences) + 1))
                        # return votes + transformed_confidences
                        
                        # this value used to compute decision, same as votes + transfomed confidences
                        # a = svc.decision_function(X_test[:,test_idx].T)
                        
                        # this shows confidence in each trial decoding (probability=True has to be used, computationally expensive)
                        # b = svc.predict_proba(X_test[:,test_idx].T)
                                      
                        # plt.figure()
                        # plt.plot(svc.coef_[:9,:].T)
                        
    perform_train_test2 = np.mean(perform_train_test1, axis=2)
    
    run_duration = time.time() - start_t
    if log:
        print('done; %.2f sec' % run_duration)
    
    dec_out = {'train_test_method':     train_test_method,
               'pca_var_frac':          pca_var_frac,
               'num_cv':                num_cv,
               'fixed_time':            fixed_time,
               'add_noise_sigma':       add_noise_sigma,
               'normalize':             normalize,
               'performance':           perform_train_test2,
               }
    
    if get_train_coeffs:
        dec_out['train_coeffs'] = train_coeffs
        dec_out['train_intercepts'] = train_intercepts
        dec_out['num_classes'] = n_classes
        dec_out['num_ovo_dec'] = num_ovo_dec
    
    return dec_out

def f_shuffle_trials(trials):
    trials_idx = np.arange(trials.shape[0])
    np.random.shuffle(trials_idx)
    trials_shuff = trials[trials_idx]
    
    return trials_shuff

#%%

def f_plot_binwise_dec(dec_data_list, plot_t=None, plot_legend=None, plot_start=-1, plot_end=5, fixed_time=0.25, title_tag=''):
    
    if type(dec_data_list) is not list:
        dec_data_list = [dec_data_list]
    
    plt_start2 = np.argmin(np.abs(plot_start - plot_t))
    plt_end2 = np.argmin(np.abs(plot_end - plot_t))
    plot_cond2 = np.argmin(np.abs(fixed_time - plot_t))

    plot_t2 = plot_t[plt_start2:plt_end2]
    
    colors1 = ['lightblue', 'gray']
    colors_mean = ['blue', 'black']
    
    if len(title_tag):
        title_tag2 = '%s; ' % title_tag
    else:
        title_tag2 = title_tag
    
    figures = {}
    
    num_dsets = len(dec_data_list)
    
    train_test_method = dec_data_list[0]['train_test_method']
    
    if train_test_method == 'full':
        
        for n_d in range(num_dsets):
            perform_train_test = dec_data_list[n_d]['performance']
            num_t, _, num_dec = perform_train_test.shape
        
            figures['full'] = []
            for n_dec in range(num_dec):
                
                fig1, ax1 = plt.subplots()
                im1 = ax1.imshow(perform_train_test[plt_start2:plt_end2,plt_start2:plt_end2,n_dec], extent=(np.min(plot_t2), np.max(plot_t2), np.max(plot_t2), np.min(plot_t2)), clim=[0, 1])
                #ax1.set_clim([0, 1])
                ax1.set_xlabel('Test time')
                ax1.set_ylabel('Train time')
                if len(title_tag):
                    ax1.set_title('%s; %s' % (title_tag, plot_legend[n_dec]))
                else:
                    ax1.set_title(plot_legend[n_dec])
                cbar = fig1.colorbar(im1)
                cbar.set_label('Performance')
                figures['full'].append(fig1)

        
            fig1, ax1 = plt.subplots()
            for n_dec in range(num_dec):
                ax1.plot(plot_t2, np.diag(perform_train_test[plt_start2:plt_end2,plt_start2:plt_end2,n_dec]))
            ax1.legend(plot_legend)
            ax1.set_xlabel('Time (sec)')
            ax1.set_ylabel('Performance')
            ax1.set_title('%sbinwise, %s' % (title_tag2, train_test_method))
            figures['diag'] = fig1
            
            fig1, ax1 = plt.subplots()
            for n_dec in range(num_dec):
                ax1.plot(plot_t2, perform_train_test[plot_cond2,plt_start2:plt_end2,n_dec])
            ax1.legend(plot_legend)
            ax1.set_xlabel('Time (sec)')
            ax1.set_ylabel('Performance')
            ax1.set_title('%sbinwise, %s, train at %.2fsec, test variable' % (title_tag2, train_test_method, plot_t[plot_cond2]))
            figures['fixed_train'] = fig1
            
            fig1, ax1 = plt.subplots()
            for n_dec in range(num_dec):
                ax1.plot(plot_t2, perform_train_test[plt_start2:plt_end2,plot_cond2,n_dec])
            ax1.legend(plot_legend)
            ax1.set_xlabel('Time (sec)')
            ax1.set_ylabel('Performance')
            ax1.set_title('%sbinwise, %s, train variable, test at %.2fsec' % (title_tag2, train_test_method, plot_t[plot_cond2]))
            figures['fixed_test'] = fig1
        
    elif train_test_method == 'diag':
        
        fig1, ax1 = plt.subplots()
        
        num_t, _, num_dec = dec_data_list[0]['performance'].shape
        traces_all = np.zeros((num_dsets, num_dec, len(plot_t2)))
        
        for n_d in range(num_dsets):
            perform_train_test = dec_data_list[n_d]['performance']
            num_t, _, num_dec = perform_train_test.shape

            for n_dec in range(num_dec):
                traces_all[n_d, n_dec, :] = perform_train_test[plt_start2:plt_end2,:,n_dec].flatten()
                ax1.plot(plot_t2, perform_train_test[plt_start2:plt_end2,:,n_dec], color=colors1[n_dec])
                
        ax1.legend(plot_legend)
        ax1.set_xlabel('Time (sec)')
        ax1.set_ylabel('Performance')
        ax1.set_title('%sbinwise, %s' % (title_tag2, train_test_method))
        figures['diag'] = fig1
        
        fig1, ax1 = plt.subplots()
        leg_lines = []
        for n_dec in range(num_dec):
            mean_trace = np.mean(traces_all[:, n_dec, :], axis=0)
            sem_trace = np.std(traces_all[:, n_dec, :], axis=0)/np.max([np.sqrt(num_dsets-1), 1])
            l1, = ax1.plot(plot_t2, mean_trace, color=colors_mean[n_dec])
            leg_lines.append(l1)
            ax1.fill_between(plot_t2, mean_trace-sem_trace, mean_trace+sem_trace, color=colors_mean[n_dec], alpha=0.2)
        
        ax1.legend(handles=leg_lines, labels=plot_legend)
        ax1.set_xlabel('Time (sec)')
        ax1.set_ylabel('Performance')
        ax1.set_title('%sbinwise, %s' % (title_tag2, train_test_method))
        figures['diag_mean'] = fig1
        
    elif train_test_method == 'fixed_train_time':
        1
    elif train_test_method == 'fixed_test_time':
        1
        
    return figures

#%%
def f_run_one_shot_dec(x_data, y_data, trial_stim_on=[], shuff_stim_type=[], shuff_bins=[], stim_on_train=True, num_cv=5, equalize_y_input=True):
    
    num_dec = len(x_data)
    trial_len, num_trials, _ = x_data[0].shape
    
    # y_data_templ = np.zeros((trial_len, num_tt, 1), dtype=int, order = 'F')
    # for n_tt in range(num_tt):
    #     y_data_templ[trial_stim_on,n_tt] = n_tt + 1
        
    
    if len(shuff_stim_type):
        shuff_stim_type2 = np.array(shuff_stim_type).astype(bool)
    else:
        shuff_stim_type2 = np.zeros(num_dec, dtype=bool)
    
    if len(shuff_bins):
        shuff_bins2 = np.array(shuff_bins).astype(bool)
    else:
        shuff_bins2 = np.zeros(num_dec, dtype=bool)
        
    if len(trial_stim_on):
        trial_stim_on2 = np.reshape(trial_stim_on.astype(int), (trial_len, 1), order='F')
        y_data3 = np.reshape(y_data, (1, num_trials), order='F')
        y_data2 = trial_stim_on2*y_data3
    else:
        y_data2 = y_data
    
    if stim_on_train:
        y_train = y_data2[trial_stim_on,:]
        #title_tag7 = 'stim on train'
    else:
        y_train = y_data2
        #title_tag7 = 'stim plus isi train'
        
    trial_len_train = y_train.shape[0]
    train_cat = np.unique(y_train)
    train_tt = np.unique(y_data)
    
    num_cat = train_cat.shape[0]
    num_tt = train_tt.shape[0]
    
    perform1_total = np.zeros((num_cv, num_dec))
    perform1_binwise = np.zeros((trial_len, num_tt, num_cv, num_dec))
    perform1_y_is_cat = np.zeros((trial_len, num_tt, num_cat, num_cv, num_dec))
    
    for n_dec in range(num_dec):
        print('decoder %d/%d' % (n_dec+1, num_dec))
        cv_groups = f_make_cv_groups(num_trials, num_cv)
        
        x_data1 = x_data[n_dec]
        
        num_cells = x_data1.shape[2]
        
        if stim_on_train:
            x_train = x_data1[trial_stim_on,:,:]
            #title_tag7 = 'stim on train'
        else:
            x_train = x_data1
            #title_tag7 = 'stim plus isi train'
        
        for n_cv in range(num_cv):
            test_idx = cv_groups[n_cv]
            train_idx = ~test_idx
            
            num_train = np.sum(train_idx)
            num_test = np.sum(test_idx)
            
            y_train2 = y_train[:,train_idx]
            
            if shuff_stim_type2[n_dec]:
                shuff_idx = np.arange(y_train2.shape[1])
                np.random.shuffle(shuff_idx)
                y_train3 = y_train2[:,shuff_idx]
            else:
                y_train3 = y_train2
                 
            y_train4 = np.reshape(y_train3, (trial_len_train*num_train), order = 'F')
            
            x_train2 = x_train[:,:,:]
            x_train3 = x_train2[:,train_idx,:]
            x_train4 = np.reshape(x_train3, (trial_len_train*num_train, num_cells), order = 'F')
            
            if equalize_y_input:
                counts1 = np.zeros(num_cat, dtype=int)
                for n_cat in range(num_cat):
                    counts1[n_cat] = np.sum(y_train4 == train_cat[n_cat])
                use_count = np.min(counts1)
            
                train_use = np.zeros(trial_len_train*num_train, dtype=bool)
                for n_cat in range(num_cat):
                    idx1 = y_train4 == train_cat[n_cat]
                    idx2 = np.where(idx1)[0]
                    np.random.shuffle(idx2)
                    train_use[idx2[:use_count]]=1
                    
                y_train5 = y_train4[train_use]
                x_train5 = x_train4[train_use,:]
            else:
                y_train5 = y_train4
                x_train5 = x_train4
            
            print('%d y total' % (y_train5.shape[0]))
            
            if shuff_bins2[n_dec]:
                shuff_idx = np.arange(y_train5.shape[0])
                np.random.shuffle(shuff_idx)
                y_train6 = y_train5[shuff_idx]
            else:
                y_train6 = y_train5    
            
            svc = svm.SVC(kernel='linear', C=1, gamma='auto')
            svc.fit(x_train5, y_train6)
            
            # now testing
            x_data2 = x_data1[:,:,:]
            x_test = x_data2[:,test_idx,:]
            x_test2 = np.reshape(x_test, (trial_len*num_test, num_cells), order = 'F')
            
            y_test_tt = y_data[test_idx]
            y_test = y_data2[:,test_idx]
            
            test_pred1 = svc.predict(x_test2)
            test_pred2 = np.reshape(test_pred1, (trial_len, num_test), order = 'F')
            
            #test_perform = test_pred1 == y_test2
            #test_perform2 = np.reshape(test_perform, (trial_len, 2, num_test), order = 'F')
            test_perform2 = y_test == test_pred2
            
            perform1_total[n_cv, n_dec] = np.mean(test_perform2[trial_stim_on,:])
            
            for n_tt in range(num_tt):
                idx1 = y_test_tt == train_tt[n_tt]
                perform1_binwise[:, n_tt, n_cv, n_dec] = np.mean(test_perform2[:, idx1], axis=1)
            
            for n_tt in range(num_tt):
                idx1 = y_test_tt == train_tt[n_tt]
                for n_cat in range(num_cat):
                    test_perform3 = test_pred2[:,idx1] == train_cat[n_cat]
                    perform1_y_is_cat[:, n_tt, n_cat, n_cv, n_dec] = np.mean(test_perform3, axis=1)
            
    
    perform_final = np.mean(perform1_total,axis=0)
    perform_binwise = np.mean(perform1_binwise, axis=2)
    perform_y_is_cat = np.mean(perform1_y_is_cat, axis=3)
    
    return perform_final, perform_binwise, perform_y_is_cat

#%%
def f_plot_one_shot_dec_bycat(perform_final, perform_binwise, perform_y_is_cat, plot_t1, leg_all, trial_labels, trial_colors):
    #y_is_ddfin = np.mean(y_is_dd, axis=2)
    #test_is_ddfin = test_is_cat_final[:,:,:,1]
    trial_len, num_tt, num_cat, num_dec = perform_y_is_cat.shape
    
    if num_cat > num_tt:
        title_tag7 = 'stim plus isi train'
    else:
        title_tag7 = 'stim on train'
    
    ylims = [-0.05, 1.05]
    
    leg_all2 = []
    for n_dec in range(num_dec):
        leg_all2.append('%s, %.1f%%' % (leg_all[n_dec], perform_final[n_dec]*100))
    
    plt.figure()
    for n_tt in range(num_tt):
        ax1 = plt.subplot(1,num_tt,n_tt+1)
        if type(trial_colors) == list:
            color1 = trial_colors[n_tt]
        else:
            color1 = trial_colors[n_tt,:]
            
        ax1.add_patch(Rectangle((plot_t1[5], ylims[0]), 0.5, ylims[1]-ylims[0], edgecolor=color1, facecolor=color1, linewidth=1))
        #ax1.plot(plot_t1, y_is_ddfin[:,0])
        for n_dec in range(num_dec):
            if num_cat > num_tt:
                n_cat = n_tt+1
            else:
                n_cat = n_tt
            ax1.plot(plot_t1, perform_y_is_cat[:,n_tt,n_cat, n_dec])
        plt.ylim(ylims)
        if not n_tt:
            plt.ylabel('stim probability')
            plt.xlabel('time (sec)')
        else:
            plt.tick_params(axis='y', labelleft=False)
        
        if len(trial_labels):
            label1 = trial_labels[n_tt]
        else:
            label1 = 'input %d' % n_tt
        plt.title(label1)
    plt.legend(ax1.lines, leg_all) # ['actual'] + 
    plt.suptitle('stim type decoding, one shot train, %s' % (title_tag7))
    
    plt.figure()
    ax3 = plt.subplot(1,1,1)
    ax3.add_patch(Rectangle((plot_t1[5], ylims[0]), 0.5, ylims[1]-ylims[0], edgecolor='lightgray', facecolor='lightgray', linewidth=1))
    for n_dec in range(num_dec):
        ax3.plot(plot_t1, np.mean(perform_binwise[:,:,n_dec],axis=1))
    ax3.plot(plot_t1, np.ones(plot_t1.shape[0]), '--', color='black')
    ax3.plot(plot_t1, np.ones(plot_t1.shape[0])/num_cat, '--', color='gray')
    plt.ylim(ylims)
    plt.ylabel('decoder performance')
    plt.xlabel('time (sec)')
    plt.title('performance')
    #plt.legend(('deviant', 'redundant'))
    plt.legend(ax3.lines, leg_all2+ ['max', 'chance']) # ['actual'] + 
    plt.suptitle('stim type decoding, one shot train, %s' % (title_tag7))

def f_plot_one_shot_dec_avecat(perform_final, perform_binwise, perform_y_is_cat, plot_t1, net_idx, leg_net):
    #y_is_ddfin = np.mean(y_is_dd, axis=2)
    #test_is_ddfin = test_is_cat_final[:,:,:,1]
    trial_len, num_tt, num_cat, num_dec = perform_y_is_cat.shape
    
    colors1 = ['tab:blue', 'tab:orange', 'tab:green']
    
    if num_cat > num_tt:
        title_tag7 = 'stim plus isi train'
    else:
        title_tag7 = 'stim on train'
    
    ylims = [-0.05, 1.05]
    
    # leg_all2 = []
    # for n_dec in range(num_dec):
    #     leg_all2.append('%s, %.1f%%' % (leg_all[n_dec], perform_final[n_dec]*100))
    
    leg_idx = np.array([np.where(net_idx==0)[0][0], np.where(net_idx==1)[0][0], np.where(net_idx==2)[0][0]]).astype(int)
    
    
    perform_y_is_cat2 = np.zeros((trial_len, num_tt, num_dec))
    for n_dec in range(num_dec):
        for n_tt in range(num_tt):
            if num_cat > num_tt:
                n_cat = n_tt+1
            else:
                n_cat = n_tt
            
            perform_y_is_cat2[:,n_tt,n_dec] = perform_y_is_cat[:,n_tt, n_cat, n_dec]
            
    
    perform_y_is_cat3 = np.mean(perform_y_is_cat2, axis=1)
    
    plt.figure()
    ax1 = plt.subplot(1,1,1)
    ax1.add_patch(Rectangle((plot_t1[5], ylims[0]), 0.5, ylims[1]-ylims[0], edgecolor='lightgray', facecolor='lightgray', linewidth=1))
    for n_dec in range(num_dec):
        
        ax1.plot(plot_t1, perform_y_is_cat3[:, n_dec], color=colors1[net_idx[n_dec]])
    
    plt.ylabel('stim probability')
    plt.xlabel('time (sec)')
    
    # for n_dec in range(num_dec):
    #     if num_cat > num_tt:
    #         n_cat = n_tt+1
    #     else:
    #         n_cat = n_tt
    #     ax1.plot(plot_t1, perform_y_is_cat[:,n_cat, n_dec])
        
    # for n_tt in range(num_tt):
    #     ax1 = plt.subplot(1,num_tt,n_tt+1)
    #     if type(trial_colors) == list:
    #         color1 = trial_colors[n_tt]
    #     else:
    #         color1 = trial_colors[n_tt,:]
            
    #     ax1.add_patch(Rectangle((plot_t1[5], ylims[0]), 0.5, ylims[1]-ylims[0], edgecolor=color1, facecolor=color1, linewidth=1))
    #     #ax1.plot(plot_t1, y_is_ddfin[:,0])
    #     for n_dec in range(num_dec):
    #         if num_cat > num_tt:
    #             n_cat = n_tt+1
    #         else:
    #             n_cat = n_tt
    #         ax1.plot(plot_t1, perform_y_is_cat[:,n_tt,n_cat, n_dec])
    #     plt.ylim(ylims)
    #     if not n_tt:
    #         plt.ylabel('stim probability')
    #         plt.xlabel('time (sec)')
    #     else:
    #         plt.tick_params(axis='y', labelleft=False)
        
    #     if len(trial_labels):
    #         label1 = trial_labels[n_tt]
    #     else:
    #         label1 = 'input %d' % n_tt
    #     plt.title(label1)
    #plt.legend(ax1.lines, leg_all) # ['actual'] + 
    plt.legend(np.array(ax1.lines)[leg_idx], leg_net) # ['actual'] + 
    plt.suptitle('stim type decoding, one shot train, %s' % (title_tag7))
    

def f_plot_one_shot_dec_bycat2(perform_final, perform_binwise, perform_y_is_cat, plot_t1, net_idx, leg_net, trial_labels, trial_colors):
    #y_is_ddfin = np.mean(y_is_dd, axis=2)
    #test_is_ddfin = test_is_cat_final[:,:,:,1]
    trial_len, num_tt, num_cat, num_dec = perform_y_is_cat.shape
    
    colors1 = ['tab:blue', 'tab:orange', 'tab:green']
    
    if num_cat > num_tt:
        title_tag7 = 'stim plus isi train'
    else:
        title_tag7 = 'stim on train'
    
    ylims = [-0.05, 1.05]
    
    # leg_all2 = []
    # for n_dec in range(num_dec):
    #     leg_all2.append('%s, %.1f%%' % (leg_all[n_dec], perform_final[n_dec]*100))
    
    leg_idx = np.array([np.where(net_idx==0)[0][0], np.where(net_idx==1)[0][0], np.where(net_idx==2)[0][0]]).astype(int)
    
    plt.figure()
    for n_tt in range(num_tt):
        
        ax1 = plt.subplot(1,num_tt,n_tt+1)
        if type(trial_colors) == list:
            color1 = trial_colors[n_tt]
        else:
            color1 = trial_colors[n_tt,:]
            
        ax1.add_patch(Rectangle((plot_t1[5], ylims[0]), 0.5, ylims[1]-ylims[0], edgecolor=color1, facecolor=color1, linewidth=1))
        #ax1.plot(plot_t1, y_is_ddfin[:,0])
        
        for n_dec in range(num_dec):
            if num_cat > num_tt:
                n_cat = n_tt+1
            else:
                n_cat = n_tt
            ax1.plot(plot_t1, perform_y_is_cat[:,n_tt,n_cat, n_dec], color=colors1[net_idx[n_dec]])
        plt.ylim(ylims)
        if not n_tt:
            plt.ylabel('stim probability')
            plt.xlabel('time (sec)')
        else:
            plt.tick_params(axis='y', labelleft=False)
        
        if len(trial_labels):
            label1 = trial_labels[n_tt]
        else:
            label1 = 'input %d' % n_tt
        plt.title(label1)
    plt.legend(np.array(ax1.lines)[leg_idx], leg_net) # ['actual'] + 
    plt.suptitle('stim type decoding, one shot train, %s' % (title_tag7))
    
    plt.figure()
    ax3 = plt.subplot(1,1,1)
    ax3.add_patch(Rectangle((plot_t1[5], ylims[0]), 0.5, ylims[1]-ylims[0], edgecolor='lightgray', facecolor='lightgray', linewidth=1))
    for n_dec in range(num_dec):
        ax3.plot(plot_t1, np.mean(perform_binwise[:,:,n_dec],axis=1), color=colors1[net_idx[n_dec]])
    ax3.plot(plot_t1, np.ones(plot_t1.shape[0]), '--', color='black')
    ax3.plot(plot_t1, np.ones(plot_t1.shape[0])/num_cat, '--', color='gray')
    plt.ylim(ylims)
    plt.ylabel('decoder performance')
    plt.xlabel('time (sec)')
    plt.title('performance')
    #plt.legend(('deviant', 'redundant'))
    plt.legend(np.array(ax3.lines)[np.hstack((leg_idx, np.array([-2, -1])))], leg_net+ ['max', 'chance']) # ['actual'] + 
    plt.suptitle('stim type decoding, one shot train, %s' % (title_tag7))

#%%

def f_plot_one_shot_dec_iscat(perform_final, perform_binwise, perform_y_is_cat, plot_t1, leg_all, trial_labels, trial_colors, cat_plot=1):
    #y_is_ddfin = np.mean(y_is_dd, axis=2)
    #test_is_ddfin = test_is_cat_final[:,:,:,1]
    trial_len, num_tt, num_cat, num_dec = perform_y_is_cat.shape
    
    if num_cat > num_tt:
        title_tag7 = 'stim plus isi train'
    else:
        title_tag7 = 'stim on train'
    
    ylims = [-0.05, 1.05]
    
    leg_all2 = []
    for n_dec in range(num_dec):
        leg_all2.append('%s, %.1f%%' % (leg_all[n_dec], perform_final[n_dec]*100))
    
    tt_list = np.arange(num_tt)
    
    tt_list2 = np.hstack((tt_list[tt_list==cat_plot], tt_list[tt_list!=cat_plot]))
    
    plt.figure()
    for n_tt in range(num_tt):
        ax1 = plt.subplot(1,num_tt,n_tt+1)
        if type(trial_colors) == list:
            color1 = trial_colors[n_tt]
        else:
            color1 = trial_colors[n_tt,:]
        ax1.add_patch(Rectangle((plot_t1[5], ylims[0]), 0.5, ylims[1]-ylims[0], edgecolor=color1, facecolor=color1, linewidth=1))
        #ax1.plot(plot_t1, y_is_ddfin[:,0])
        for n_dec in range(num_dec):
            # if num_cat > num_tt:
            #     n_cat = n_tt+1
            # else:
            #     n_cat = n_tt
            ax1.plot(plot_t1, perform_y_is_cat[:,tt_list2[n_tt],cat_plot, n_dec])
        plt.ylim(ylims)
        if not n_tt:
            plt.ylabel('stim probability')
            plt.xlabel('time (sec)')
        else:
            plt.tick_params(axis='y', labelleft=False)
        
        if len(trial_labels):
            label1 = trial_labels[n_tt]
        else:
            label1 = 'input %d' % n_tt
        plt.title(label1)
    plt.legend(ax1.lines, leg_all) # ['actual'] + 
    plt.suptitle('stim type decoding, one shot train, %s' % (title_tag7))
    
    plt.figure()
    ax3 = plt.subplot(1,1,1)
    ax3.add_patch(Rectangle((plot_t1[5], ylims[0]), 0.5, ylims[1]-ylims[0], edgecolor='lightgray', facecolor='lightgray', linewidth=1))
    for n_dec in range(num_dec):
        ax3.plot(plot_t1, np.mean(perform_binwise[:,:,n_dec],axis=1))
    ax3.plot(plot_t1, np.ones(plot_t1.shape[0]), '--', color='black')
    ax3.plot(plot_t1, np.ones(plot_t1.shape[0])/num_cat, '--', color='gray')
    plt.ylim(ylims)
    plt.ylabel('decoder performance')
    plt.xlabel('time (sec)')
    plt.title('performance')
    #plt.legend(('deviant', 'redundant'))
    plt.legend(ax3.lines, leg_all2+ ['max', 'chance']) # ['actual'] + 
    plt.suptitle('stim type decoding, one shot train, %s' % (title_tag7))
    
    
    # plt.figure()
    # ax1 = plt.subplot(1,3,1)
    # ax1.add_patch(patches.Rectangle((0, ylims[0]), 0.5, ylims[1]-ylims[0], edgecolor='pink', facecolor='pink', linewidth=1))
    # #ax1.plot(plot_t1, y_is_ddfin[:,0])
    # pllist = []
    # for n_dec in range(num_dec):
    #     pllist.append(ax1.plot(plot_t1, test_is_ddfin[:,0,n_dec]))
    # plt.ylim(ylims)
    # plt.ylabel('deviant probability')
    # plt.xlabel('time (sec)')
    # plt.title('deviant trial')
    # ax2 = plt.subplot(1,3,2)
    # ax2.add_patch(patches.Rectangle((0, ylims[0]), 0.5, ylims[1]-ylims[0], edgecolor='lightblue', facecolor='lightblue', linewidth=1))
    # #plt.plot(plot_t1, y_is_ddfin[:,1])
    # for n_dec in range(num_dec):
    #     ax2.plot(plot_t1, test_is_ddfin[:,1,n_dec])
    # plt.ylim(ylims)
    # plt.tick_params(axis='y', labelleft=False)
    # plt.title('redundant trial')
    # plt.legend(ax2.lines, leg_all) #  + ['actual']
    # ax3 = plt.subplot(1,3,3)
    # ax3.add_patch(patches.Rectangle((0, ylims[0]), 0.5, ylims[1]-ylims[0], edgecolor='lightgray', facecolor='lightgray', linewidth=1))
    # for n_dec in range(num_dec):
    #     ax3.plot(plot_t1, np.mean(perform1_binwise[:,:,n_dec],axis=1))
    # ax3.plot(plot_t1, np.ones(plot_t1.shape[0]), '--', color='black')
    # ax3.plot(plot_t1, 0.5*np.ones(plot_t1.shape[0]), '--', color='gray')
    # plt.ylim(ylims)
    # plt.ylabel('decoder performance', labelpad=-150, rotation=270)
    # plt.tick_params(axis='y', labelleft=False, labelright=True, left=False, right=True)
    # plt.title('performance')
    # #plt.legend(('deviant', 'redundant'))
    # plt.legend(ax3.lines, leg_all+ ['ideal', 'chance']) # ['actual'] + 
    # title_tag6 = ''
    # for n_dec in range(num_dec):
    #     title_tag6 += ' %s=%.1f%% ' % (leg_all[n_dec], perform1_final[n_dec]*100) 
    # plt.suptitle('deviant probability decoding, one shot train, %s\nperf %s' % (title_tag7, title_tag6))