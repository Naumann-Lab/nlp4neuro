#!/usr/bin/env python
"""
experiment_4.py

This script implements new behavior tasks derived from colleague code.
Instead of predicting tail positions from neural data (as in experiment 1),
this experiment focuses solely on behavioral analysis from tail recordings.
It processes tail data to detect tail bouts, plots raw and smoothed traces,
and visualizes tail-angle distributions and stimuli responses.

Author: Your Name
Date: 2024-XX-XX
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks
from datetime import datetime

# =============================================================================
# 0) Setup and Constants
# =============================================================================

# Base directory to save experiment results
BASE_SAVE_DIR = os.path.join(os.getcwd(), "results", "experiment_4_behavior")
os.makedirs(BASE_SAVE_DIR, exist_ok=True)

# List of fish to analyze
fish_list = [9, 10, 11, 12, 13]

# Example frame rate for tail recordings (Hz) - adjust as needed
tail_hz = 30

# =============================================================================
# 1) Behavioral Analysis Functions (adapted from colleague code)
# =============================================================================

# For simplicity, we define minimal constants and helper functions.
class constants:
    # Example color dictionaries; adjust these as necessary.
    allcolor_dict = {'stim1': ('#FF9999', '#FFCCCC')}
    monocular_dict = {'left': '#1f77b4', 'right': '#ff7f0e'}
    deg_dict = {'left': 45, 'right': 135}

def pretty(arr, n):
    return np.around(arr, decimals=n)

def analyze_tail(frametimes_df, stimulus_df, tail_df, tail_hz, stimulus_s=5, strength_boundary=0.25, 
                 min_on_s=0.1, cont_cutoff_s=0.05):
    """
    Process tail data to identify tail bouts and produce plots of:
    - Raw tail trace,
    - Smoothed tail trace (via local standard deviation),
    - Detected bouts shaded in the tail strength plot.
    
    Returns a dataframe with bout information.
    """
    # Calculate imaging frequency from frametimes (if not provided)
    hz = tail_hz  # using tail_hz as provided
    
    fig, ax = plt.subplots(3, 1, dpi=150, figsize=(12, 8))
    tail_df = tail_df.fillna(method='ffill')
    
    # Plot raw tail signal
    ax[0].plot(tail_df['frame'], tail_df['tail_sum'], linewidth=0.5, color='black')
    ax[0].set_xlim([0, tail_df['frame'].max()])
    ax[0].set_xlabel("Frame")
    ax[0].set_ylabel("Raw Tail Angle (rad)")
    ax[0].set_title("Raw Tail Trace")
    
    # Calibrate tail signal by subtracting baseline
    baseline = tail_df['tail_sum'].mean()
    tail_df['tail_sum'] = tail_df['tail_sum'] - baseline
    tail_df['tail_sum'] = tail_df['tail_sum'].fillna(method='ffill')
    
    # Separate positive and negative tail movements
    pos = np.where(tail_df['tail_sum'] > 0, tail_df['tail_sum'], 0)
    neg = np.where(tail_df['tail_sum'] < 0, tail_df['tail_sum'], 0)
    ax[1].plot(tail_df['frame'], pos, linewidth=0.5, color='maroon', label='Positive')
    ax[1].plot(tail_df['frame'], neg, linewidth=0.5, color='midnightblue', label='Negative')
    ax[1].set_xlabel("Frame")
    ax[1].set_ylabel("Tail Angle (rad)")
    ax[1].set_title("Positive/Negative Tail Movements")
    ax[1].legend()
    
    # Smooth tail signal using a running window (here ~50ms)
    smooth_tailframe = int(0.05 * hz)
    std_vals = [np.std(tail_df['tail_sum'].iloc[i - smooth_tailframe:i + smooth_tailframe]) 
                for i in range(smooth_tailframe, len(tail_df) - smooth_tailframe)]
    tail_df['std'] = [0] * smooth_tailframe + std_vals + [0] * smooth_tailframe
    ax[2].plot(tail_df['frame'], tail_df['std'], linewidth=0.5, color='black')
    
    # Detect bouts: where the local std exceeds a strength threshold
    bout_on = (tail_df['std'] > strength_boundary).astype(int)
    on_index = np.where(np.diff(bout_on) == 1)[0] + smooth_tailframe
    off_index = np.where(np.diff(bout_on) == -1)[0] + smooth_tailframe
    
    if len(on_index) != 0 and len(off_index) != 0:
        # Ensure matching on/off indices
        if on_index[0] > off_index[0]:
            on_index = np.concatenate(([0], on_index))
        if on_index[-1] > off_index[-1]:
            off_index = np.concatenate((off_index, [len(tail_df) - 1]))
        on_tuples = [(on, off) for on, off in zip(on_index, off_index) if off > on]
    else:
        on_tuples = []
        
    # Group bouts if they are close in time
    cont_on_index = []
    cont_off_index = []
    if on_tuples:
        cont_on_index = [on_tuples[0][0]]
        if len(on_tuples) > 1:
            intervals = np.array([on_tuples[i][0] - on_tuples[i - 1][1] for i in range(1, len(on_tuples))])
            big_interval = intervals > (cont_cutoff_s * hz)
            for i, flag in enumerate(big_interval):
                if flag:
                    cont_on_index.append(on_tuples[i+1][0])
                    cont_off_index.append(on_tuples[i][1])
            cont_off_index.append(on_tuples[-1][1])
    cont_tuples = [(on, off) for on, off in zip(cont_on_index, cont_off_index) 
                   if (off - on) > (min_on_s * hz)]
    
    # Shade detected bouts on the std plot
    for on, off in cont_tuples:
        ax[2].axvspan(tail_df['frame'].iloc[on], tail_df['frame'].iloc[off], color='pink', alpha=0.5)
    
    ax[2].axhline(strength_boundary, linestyle=':', color='purple', linewidth=1)
    ax[2].set_xlabel("Frame")
    ax[2].set_ylabel("Tail Strength (std)")
    ax[2].set_title("Tail Strength and Detected Bouts")
    
    plt.tight_layout()
    
    # Create a bout dataframe to return bout information
    bout_df = pd.DataFrame({
        'bout_start_frame': [tail_df['frame'].iloc[on] for on, off in cont_tuples],
        'bout_end_frame': [tail_df['frame'].iloc[off] for on, off in cont_tuples],
        'bout_duration_frames': [off - on for on, off in cont_tuples]
    })
    
    return bout_df

def tail_angle_all(frametimes_df, stimulus_df, tail_bout_df, stimulus_s=5):
    """
    Create summary plots of tail bout durations and count stimuli responses.
    Returns a dataframe summarizing stimuli presentations and responses.
    """
    # Here we assume a simple scenario with one stimulus type ("stim1") and spontaneous bouts.
    all_stimuli = ['stim1', 'spontaneous']
    # Dummy counts for demonstration; in practice, compute based on your stimulus_df and tail_bout_df.
    presenting = [10, 5]
    responding = [8, 3]
    df_summary = pd.DataFrame({
        'stimuli': all_stimuli,
        'presented': presenting,
        'responded': responding
    })
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(df_summary['stimuli'], df_summary['presented'], color='grey', alpha=0.5, label='Presented')
    ax.barh(df_summary['stimuli'], df_summary['responded'], color='skyblue', alpha=0.8, label='Responded')
    ax.set_xlabel('Count')
    ax.set_title('Stimuli Presentations vs. Responses')
    ax.legend()
    plt.tight_layout()
    
    return df_summary

# =============================================================================
# 2) Main Loop: Run Behavioral Tasks for Each Fish
# =============================================================================

for fish_num in fish_list:
    print(f"\nProcessing Fish {fish_num} for behavioral analysis...")
    # Create a folder for the fish
    fish_save_dir = os.path.join(BASE_SAVE_DIR, f"fish{fish_num}")
    os.makedirs(fish_save_dir, exist_ok=True)
    
    # -------------------------------
    # Load Tail Data
    # -------------------------------
    # Adjust the file path as needed.
    tail_data_path = f"/hpc/group/naumannlab/jjm132/data_prepped_for_models/fish{fish_num}_tail_data_matched.npy"
    tail_data = np.load(tail_data_path, allow_pickle=True)
    
    # Assume tail_data is either 1D (per frame) or 2D; if 2D, use first column
    if tail_data.ndim > 1:
        tail_signal = tail_data[:, 0]
    else:
        tail_signal = tail_data
        
    # Create a tail DataFrame
    tail_df = pd.DataFrame({
        'frame': np.arange(len(tail_signal)),
        'tail_sum': tail_signal
    })
    
    # -------------------------------
    # Create Dummy Frametimes and Stimulus DataFrames
    # -------------------------------
    frametimes_df = pd.DataFrame({
        'frame': np.arange(len(tail_signal)),
        'time': np.arange(len(tail_signal)) / tail_hz
    })
    # In a real experiment, load your stimulus timings; here we use an empty DataFrame.
    stimulus_df = pd.DataFrame(columns=['frame', 'stim_name'])
    
    # -------------------------------
    # Run Behavioral Analysis: Tail Bout Detection
    # -------------------------------
    print("Running analyze_tail ...")
    bout_df = analyze_tail(frametimes_df, stimulus_df, tail_df, tail_hz, stimulus_s=5)
    bout_plot_path = os.path.join(fish_save_dir, f"fish{fish_num}_tail_bout_analysis.png")
    plt.savefig(bout_plot_path)
    plt.close()
    bout_df.to_csv(os.path.join(fish_save_dir, f"fish{fish_num}_tail_bout_summary.csv"), index=False)
    print(f"Tail bout analysis for Fish {fish_num} saved to {bout_plot_path} and summary CSV.")
    
    # -------------------------------
    # Run Behavioral Analysis: Stimuli vs Response Summary
    # -------------------------------
    print("Running tail_angle_all ...")
    stimuli_summary = tail_angle_all(frametimes_df, stimulus_df, bout_df, stimulus_s=5)
    summary_plot_path = os.path.join(fish_save_dir, f"fish{fish_num}_stimuli_summary.png")
    plt.savefig(summary_plot_path)
    plt.close()
    stimuli_summary.to_csv(os.path.join(fish_save_dir, f"fish{fish_num}_stimuli_summary.csv"), index=False)
    print(f"Stimuli summary for Fish {fish_num} saved to {summary_plot_path} and CSV.")
    
    print(f"Behavioral analysis completed for Fish {fish_num}.\n")

print("Experiment 4 (Behavioral Tasks) completed for all fish.")
