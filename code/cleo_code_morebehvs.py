"""
Functions to plot a bunch of relevant graphs for each plane. Can be ran through plot_individual_plane_runningscipt.py

@Zichen He 20240313
"""

# goal: integrate additional behavior prediction tasks for llms and rnns.. can have exp4 comparing these perfs
# Note: gold standard paper is https://pubmed.ncbi.nlm.nih.gov/29307558/ -- look to this for more guidance.

import constants
from utilities import clustering, arrutils
#from fishy import WorkingFish, BaseFish, VizStimVolume

import pandas as pd
import cmasher as cmr
import matplotlib as mpl
import matplotlib.patches as patches
from matplotlib import pyplot as plt
# from tqdm.auto import tqdm
import numpy as np
import seaborn as sns
from scipy.signal import find_peaks
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr, zscore
import random

from fishy import BaseFish
from datetime import datetime as dt
hzReturner = BaseFish.hzReturner

def analyze_tail(frametimes_df, stimulus_df, tail_df, tail_hz, stimulus_s = 5, strength_boundary = 0.25, min_on_s = 0.1, cont_cutoff_s = 0.05):
    """
    capture tail events happened in the current inputs. Also produce a graph for the raw trace, smoothed trace, and positive/negative smoothed trace for the bouts
        frametimes_df: the dataframe for the frames and corresponding real times
        stimulus_df: the dataframe contain all the stimulus and their onset frames
        tail_df: the dataframe of the tail movement and their corresponding frames
        stimuli_s: the second that the stimuli was on
        strength_boundary: the minimal std for a tail to be counted as on
        min_on_s: the minimal frame for a bout to be considered bouts
        cont_cutoff_s: the minimal frame of bout interval (in imaging speed rather than behavior speed) for the bout
        to be counted as being continuous
    Return:
        tail_byframe: a ndarray containing the calibrated tail angle by each frame
        tail_bout_df: a dataframe contains all the bout information, which includes
            cont_tuples: a list that contains the frames for [on_frame, off_frame] of each bout
            tail_strength: a list that contains the strength of the tail for each bout, calculated by standard deviation
            tail_angle_pos: a list that contains all the bouts and their corresponding max positive tail angle
            tail_angle_neg:a list that contains all the bouts and their corresponding max negative tail angle
            tail_duration_s: a list that contains all the bouts and their corresponding duration in seconds
            tail_stimuli: a list that contains all the stimuli for each bouts, if they are happened within the stimuli
             presentation. If not, the stimuli is labeled "spontaneous"
    """
    hz = hzReturner(frametimes_df)

    #min_on_hz = min_on_s * hz
    fig, ax = plt.subplots(3, 1, dpi = 400, figsize = (20, 6))
    tail_df = tail_df.ffill()

    ax[0].plot(tail_df.frame, list(tail_df.tail_sum), linewidth = 0.5, color = 'black')
    ax[0].set_xlim([0, np.max(tail_df.frame)])

    #calibrate to mean
    baseline = np.nanmean(tail_df.tail_sum)
    tail_df.tail_sum = np.subtract(tail_df.tail_sum, baseline)
    tail_df.tail_sum = tail_df.tail_sum.ffill()

    #collect positive and negative tail movement
    pos = np.where(tail_df.tail_sum > 0,tail_df.tail_sum, 0)
    neg = np.where(tail_df.tail_sum < 0,tail_df.tail_sum, 0)
    ax[1].plot(tail_df.frame, pos, linewidth = 0.5, color = 'maroon')
    ax[1].plot(tail_df.frame, neg, linewidth = 0.5, color = 'midnightblue')

    # group/smooth by running window  of ~100ms
    smooth_tailframe = int(0.05 * tail_hz)
    std = [np.std(tail_df.tail_sum[i - smooth_tailframe:i + smooth_tailframe]) for i in range(smooth_tailframe, len(tail_df.tail_sum) - smooth_tailframe)]
    tail_df['std'] =[0] * smooth_tailframe + std + [0] * smooth_tailframe
    ax[2].plot(tail_df.frame, tail_df['std'], linewidth = 0.5, color = 'black')
    bout_on = tail_df['std']> strength_boundary
    bout_on = [int(x) for x in bout_on]
    on_index = np.where(np.diff(bout_on) == 1)[0]
    on_index = [i + smooth_tailframe for i in on_index]
    off_index = np.where(np.diff(bout_on) == -1)[0]
    off_index = [i + smooth_tailframe for i in off_index]
    if len(on_index) != 0 and len(off_index) != 0:
        if on_index[0] > off_index[0]:
            on_index = np.concatenate([[0], on_index])
        if on_index[-1] > off_index[-1]:
            off_index = np.concatenate([off_index, [len(tail_df) - 1]])
        on_tuples = [(on, off) for on, off in zip(on_index, off_index) if off - on < len(tail_df) and off > on]
    cont_on_index = []
    cont_off_index = []
    if len(on_tuples) > 0:
        cont_on_index = [on_tuples[0][0]]
        if len(on_tuples) > 1:
            big_interval = np.array([on_tuples[i][0] - on_tuples[i - 1][1] for i in range(1, len(on_tuples))]) \
                           > (cont_cutoff_s * tail_hz)
            for i in range(0, len(big_interval)):
                if big_interval[i]:
                    cont_on_index = cont_on_index + [on_tuples[i + 1][0]]
                    cont_off_index = cont_off_index + [on_tuples[i][1]]
        cont_off_index = cont_off_index + [on_tuples[-1][1]]
    cont_tuples = [(on, off) for on, off in zip(cont_on_index, cont_off_index) if off - on > (min_on_s * tail_hz)]

    #calculate the actual frame(approx.) and image onset index/frames
    cont_tuples_imageframe = [(tail_df.iloc[tu[0]].frame, tail_df.iloc[tu[1]].frame) for tu in cont_tuples]

    ax[2].axhline(strength_boundary, linewidth = 1, linestyle = ':', color = 'purple')
    tail_strength = np.full(len(cont_tuples), np.nan)
    tail_angle = np.full(len(cont_tuples), np.nan)
    tail_angle_pos = np.full(len(cont_tuples), np.nan)
    tail_angle_posmax = np.full(len(cont_tuples), np.nan)
    tail_angle_neg = np.full(len(cont_tuples), np.nan)
    tail_angle_negmin = np.full(len(cont_tuples), np.nan)
    tail_duration_s = np.full(len(cont_tuples), np.nan)
    tail_frequency_s = np.full(len(cont_tuples), np.nan)
    tail_stimuli = ['spontaneous'] *len(cont_tuples)
    for i in range(len(cont_tuples)):
        ax[2].axvspan(cont_tuples_imageframe[i][0], cont_tuples_imageframe[i][1], color = 'pink', alpha = 0.5)
        tail_strength[i] = np.nanmean(tail_df['std'][cont_tuples[i][0]:cont_tuples[i][1]])
        tail_angle[i] = np.nanmean(tail_df.tail_sum[cont_tuples[i][0]:cont_tuples[i][1]])
        tail_angle_pos[i] = np.nanmean(pos[cont_tuples[i][0]:cont_tuples[i][1]])
        tail_angle_posmax[i] = np.max(pos[cont_tuples[i][0]:cont_tuples[i][1]])
        tail_angle_neg[i] = np.nanmean(neg[cont_tuples[i][0]:cont_tuples[i][1]])
        tail_angle_negmin[i] = np.min(neg[cont_tuples[i][0]:cont_tuples[i][1]])
        tail_duration_s[i] = np.divide((cont_tuples[i][1] - cont_tuples[i][0]), tail_hz)
        tail_of_interest = list(tail_df.iloc[cont_tuples[i][0]:cont_tuples[i][1]].tail_sum)
        mean_line = np.mean(tail_of_interest)
        crossing = np.subtract(arrutils.pretty(tail_of_interest, 3), mean_line)
        crossing = np.sign(crossing)
        crossing = np.count_nonzero(np.diff(crossing))
        tail_frequency_s[i] = np.divide(crossing / 2, tail_duration_s[i])
        if not stimulus_df.empty:
            if cont_tuples_imageframe[i][0] > stimulus_df.iloc[0]['frame'] :
                stimulus_responding = stimulus_df[stimulus_df['frame'] <= cont_tuples_imageframe[i][0]].iloc[-1]#find the nearest stimuli before and see if tail happens within the stimulus
                if stimulus_responding['frame'] + stimulus_s * hz >= cont_tuples_imageframe[i][0]:#if tail starts before the stimulus ends
                    tail_stimuli[i] = stimulus_responding['stim_name']
    #plot stimulus
    ax_stimuli = ax[1].twiny()
    ax_stimuli.set_xlim([0, np.max(tail_df.frame)])
    ax_stimuli.set_xticks([])
    ax_stimuli.set_xticklabels([])
    if not stimulus_df.empty:
        for i, stim_row in stimulus_df.iterrows():
            c = constants.allcolor_dict[stim_row['stim_name']]
            ax_stimuli.axvspan(stim_row['frame'] - 1,stim_row['frame'] - 1 + stimulus_s*hz/2, color = c[0], alpha = 0.2)
            ax_stimuli.axvspan(stim_row['frame'] - 1 + stimulus_s*hz/2 + 0.1, stim_row['frame'] - 1 + stimulus_s*hz, color = c[1], alpha = 0.2)

    #make plot prettier
    ax[0].set_xlabel('tail frames')
    ax[0].set_ylabel('raw tail angle (rad)')
    #ax[0].set_yticks([3.14, 0, -3.14])
    ax[1].set_ylabel('smooth tail angle (rad)')
    ax[1].set_xlim([0, np.max(tail_df.frame)])
    ax[1].set_xticks([])
    #ax[1].set_yticks([3.14, 0, -3.14])
    ax[2].set_ylabel('tail strength (std)')
    ax[2].set_xlabel('frames')
    #ax[2].set_yticks([0.5, 0])
    ax[2].sharex(ax[1])

    tail_bout_df = pd.DataFrame(
        {'cont_tuples_tailindex': cont_tuples, 'cont_tuples_imageframe': cont_tuples_imageframe,
         'tail_strength': tail_strength, 'tail_angle': tail_angle, 'tail_angle_pos': tail_angle_pos,
         'tail_angle_posmax': tail_angle_posmax, 'tail_angle_neg': tail_angle_neg, 'tail_angle_negmin': tail_angle_negmin,
         'tail_duration_s': tail_duration_s, 'tail_frequency_s': tail_frequency_s,'tail_stimuli': tail_stimuli})
    return tail_bout_df

def tail_angle_all(frametimes_df, stimulus_df, tail_bout_df, stimulus_s = 5):
    """
    Plot the tail response against all angles, including spontanesous response
        frametimes_df: the dataframe for the frames and corresponding real times
        stimulus_df: the dataframe contain all the stimulus and their onset frames
        tail_bout_df: the dataframe of the tail bouts movement details, which includes cont_tuples, tail_strength, tail_angle_pos, tail_angle_neg, tail_duration_s, tail_stimuli
        stimuli_s: the second that the stimuli was on
    Return:
        stimuli_presenting_responding_df: a dataframe containing all the stimuli, how many times they are presented,
         and how many times they are responded to
    """
    hz = hzReturner(frametimes_df)
    # make prettier
    def fix_ax(ax):
        """
        Make axis look prettier
        """
        ax.set_theta_offset(np.deg2rad(90))
        ax.set_theta_direction('clockwise')
        #ax.set_ylim([0, 0.15])#0.8
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    fig, ax = plt.subplots(4, 4, figsize = (10, 8), dpi = 240, gridspec_kw={'width_ratios': [1, 1, 1, 1]})

    #find out how many time each stimuli is presented and responded to
    gs = ax[0, 0].get_gridspec()
    for axes in ax[0:, 0]:
        axes.remove()
    ax_hist = fig.add_subplot(gs[0:, 0])
    all_stimuli_list = list(constants.deg_dict.keys()) + ['spontaneous']
    all_stimuli_color_list = {}
    for s in all_stimuli_list:
        if s in constants.monocular_dict.keys():
            all_stimuli_color_list[s] = constants.monocular_dict[s]
    all_stimuli_color_list['lateral_left'] = constants.monocular_dict['left']
    all_stimuli_color_list['medial_left'] = constants.monocular_dict['left']
    all_stimuli_color_list['lateral_right'] = constants.monocular_dict['right']
    all_stimuli_color_list['medial_right'] = constants.monocular_dict['right']
    all_stimuli_color_list['diverging'] = 'brown'
    all_stimuli_color_list['converging'] = 'brown'
    all_stimuli_color_list['spontaneous'] = 'brown'
    #all_stimuli_color_list = all_stimuli_color_list + [constants.monocular_dict['left']] * 2 \
    #                         + [constants.monocular_dict['right']] * 2 + ['brown'] * 3

    stimuli_presentation = {stimuli: 0 for stimuli in all_stimuli_list}
    stimuli_responding = {stimuli: 0 for stimuli in all_stimuli_list}
    stimuli_responding_duration_s = {stimuli: [] for stimuli in all_stimuli_list}
    tail_bout_on = np.array([list(tail_bout_df.cont_tuples_imageframe)[i][0] for i in range(len(tail_bout_df))])
    for stimuli in all_stimuli_list:
        stimulus_df_thisstim = stimulus_df[stimulus_df.stim_name == stimuli]
        stimuli_presentation[stimuli] = len(stimulus_df_thisstim)
        for rep in range(len(stimulus_df_thisstim)):
            stim_on = list(stimulus_df_thisstim.frame)[rep]
            stim_tuple = (stim_on, stim_on + stimulus_s * hz)
            if np.any((tail_bout_on <= stim_tuple[1])&(tail_bout_on >= stim_tuple[0] )):
                stimuli_responding[stimuli] += 1
        stimuli_responding_duration_s[stimuli] = list(tail_bout_df[tail_bout_df.tail_stimuli == stimuli].tail_duration_s)
    ax_hist.barh(all_stimuli_list + ['all'], list(stimuli_presentation.values()) + [np.nan], color = 'grey', alpha = 0.5)
    ax_hist.barh(all_stimuli_list + ['all'], list(stimuli_responding.values()) + [np.nan], color = all_stimuli_color_list.values(), alpha = 0.8)
    stimuli_presenting_responding_df = pd.DataFrame(data = {'stimuli': all_stimuli_list,
        'presenting': stimuli_presentation.values(), 'responding': stimuli_responding.values()})
    ax_hist.set_xlabel('bout/stim occurance')
    ax_hist.set_ylim([-1, len(all_stimuli_list) + 1])
    #plot scatter/violin plot for bout durations
    tail_bout_df_copy = tail_bout_df.copy()
    tail_bout_df_copy['tail_stimuli'] = ['all'] * len(tail_bout_df_copy)
    tail_bout_df_copy = pd.concat([tail_bout_df, tail_bout_df_copy])
    all_stimuli_color_list['all'] = 'black'
    gs = ax[0, 1].get_gridspec()
    for axes in ax[0:, 1]:
        axes.remove()
    ax_time = fig.add_subplot(gs[0:, 1])
    sns.violinplot(data=tail_bout_df_copy, y='tail_stimuli', x='tail_duration_s', order=all_stimuli_list + ['all'], ax=ax_time,
                   hue = 'tail_stimuli', palette=all_stimuli_color_list, alpha=0.2, linewidth = 0, log_scale = True,
                   inner_kws=dict(box_width=5, whis_width=1, color = 'silver',  alpha = 1, zorder = 0), zorder = 0)
    for i in range(0, len(all_stimuli_list)):
        y = all_stimuli_list[i]
        x = stimuli_responding_duration_s[y]
        ax_time.scatter(y= np.add([i] * len(x), np.random.uniform(-0.1, 0.1, len(x))), x= x,
                        color = all_stimuli_color_list[y], alpha = 0.5, s = 2, zorder = 1)
    x = [a for b in stimuli_responding_duration_s.values() for a in b]
    ax_time.scatter(y= np.add([len(all_stimuli_list)] * len(x), np.random.uniform(-0.1, 0.1, len(x))), x= x,
                        color = all_stimuli_color_list['all'], alpha = 0.5, s = 2, zorder = 1)
    ax_time.invert_yaxis()
    ax_time.set_ylim([-1, len(all_stimuli_list) + 1])
    ax_time.set_yticks(all_stimuli_list)
    ax_time.set_yticklabels([])
    ax_time.set_ylabel('')
    ax_time.set_xlim([0.02, 20])
    ax_time.set_xscale('log')
    ax_time.set_xticks([0.02, 0.2, 1, 2, 20])
    ax_time.set_xticklabels([0.02, 0.2,1, 2, 20])
    ax_time.axvline(0.2, linestyle = ':', color = 'grey', linewidth = 1)
    ax_time.set_xlabel('bout duration (s)')
    #separate binocular visually evoked response and spontaneous response
    bi_response_index = []
    spon_response_index = []
    for index in tail_bout_df.index:
        if tail_bout_df.tail_stimuli[index] in constants.monocular_dict.keys():
           bi_response_index = bi_response_index + [index]
        elif not tail_bout_df.tail_stimuli[index] in constants.deg_dict.keys():
            spon_response_index = spon_response_index + [index]
    bi_tail_strength = [tail_bout_df.tail_strength[index] for index in bi_response_index]
    bi_tail_angle = [tail_bout_df.tail_angle[index] for index in bi_response_index]
    bi_tail_angle_pos = [tail_bout_df.tail_angle_pos[index] for index in bi_response_index]
    bi_tail_angle_neg = [tail_bout_df.tail_angle_neg[index] for index in bi_response_index]
    bi_tail_duration_s = [tail_bout_df.tail_duration_s[index] for index in bi_response_index]
    bi_tail_stimuli = [tail_bout_df.tail_stimuli[index] for index in bi_response_index]
    bi_tail_stimuli_color = [constants.monocular_dict.get(stimuli) for stimuli in bi_tail_stimuli]

    spon_tail_strength = [tail_bout_df.tail_strength[index] for index in spon_response_index]
    spon_tail_angle = [tail_bout_df.tail_angle[index] for index in spon_response_index]
    spon_tail_angle_pos = [tail_bout_df.tail_angle_pos[index] for index in spon_response_index]
    spon_tail_angle_neg = [tail_bout_df.tail_angle_neg[index] for index in spon_response_index]
    spon_tail_duration_s = [tail_bout_df.tail_duration_s[index] for index in spon_response_index]

    #plot all binocular responses
    if len(bi_tail_strength) > 0:
        gs = ax[0, 2].get_gridspec()
        ax[0, 2].remove()
        ax_bip = fig.add_subplot(gs[0, 2], projection = 'polar')
        ax_bip.bar(bi_tail_angle_pos, bi_tail_strength, color = bi_tail_stimuli_color , width = 0.02, alpha = 0.5, zorder = 10)
        gs = ax[1, 2].get_gridspec()
        ax[1, 2].remove()
        ax_bin = fig.add_subplot(gs[1, 2], projection = 'polar')
        ax_bin.bar(bi_tail_angle_neg, bi_tail_strength, color = bi_tail_stimuli_color , width = 0.02, alpha = 0.5, zorder = 10)
        gs = ax[2, 2].get_gridspec()
        ax[2, 2].remove()
        ax_bia = fig.add_subplot(gs[2, 2], projection = 'polar')
        ax_bia.bar(bi_tail_angle, bi_tail_strength, color = bi_tail_stimuli_color , width = 0.02, alpha = 0.5, zorder = 10)
        gs = ax[3, 2].get_gridspec()
        ax[3, 2].remove()
        ax_bid = fig.add_subplot(gs[3, 2], projection = 'polar')
        ax_bid.bar([np.deg2rad(constants.deg_dict[s]) for s in constants.monocular_dict.keys()],
                   len(constants.monocular_dict.values()) * [1],
                   color = constants.monocular_dict.values() , width = 0.75, alpha = 0.3, zorder = 10)
        fix_ax(ax_bip)
        ax_bip.axvspan(np.deg2rad(180), np.deg2rad(360), color='lightgrey', alpha=0.3)
        fix_ax(ax_bin)
        ax_bin.axvspan(np.deg2rad(0), np.deg2rad(180), color='lightgrey', alpha=0.3)
        fix_ax(ax_bia)
        fix_ax(ax_bid)
        ax_bip.set_title('binocular')
        ax_bip.set_ylabel('pos mean\ntail angle')
        ax_bin.set_ylabel('neg mean\ntail angle')
        ax_bia.set_ylabel('sum mean\ntail angle')
        ax_bid.set_ylabel('stimuli\nangle')
    #plot spontaneous responses
    if len(spon_tail_strength) > 0:
        gs = ax[0, 3].get_gridspec()
        ax[0, 3].remove()
        ax_sponp = fig.add_subplot(gs[0, 3], projection = 'polar')
        ax_sponp.bar(spon_tail_angle_pos, spon_tail_strength, color = 'brown' , width = 0.02, alpha = 0.5, zorder = 10)
        gs = ax[1, 3].get_gridspec()
        ax[1, 3].remove()
        ax_sponn = fig.add_subplot(gs[1, 3], projection = 'polar')
        ax_sponn.bar(spon_tail_angle_neg, spon_tail_strength, color = 'brown' , width = 0.02, alpha = 0.5, zorder = 10)
        gs = ax[2, 3].get_gridspec()
        ax[2, 3].remove()
        ax_spona = fig.add_subplot(gs[2, 3], projection = 'polar')
        ax_spona.bar(spon_tail_angle, spon_tail_strength, color = 'brown' , width = 0.04, alpha = 0.5, zorder = 10)
        ax[3, 3].axis('off')
        fix_ax(ax_sponp)
        ax_sponp.axvspan(np.deg2rad(180), np.deg2rad(360), color='lightgrey', alpha=0.3)
        fix_ax(ax_sponn)
        ax_sponn.axvspan(np.deg2rad(0), np.deg2rad(180), color='lightgrey', alpha=0.3)
        fix_ax(ax_spona)
        ax_sponp.set_title('spontaneous')
    return stimuli_presenting_responding_df
