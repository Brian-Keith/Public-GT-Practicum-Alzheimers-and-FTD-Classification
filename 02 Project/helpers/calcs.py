'''Calculations for the project.

This module contains some helper functions used for calculations in the project.

Copyright 2024, Brian Keith
All Rights Reserved
'''
from scipy.signal import welch
import numpy as np
import pandas as pd
from timeit import default_timer
from .utils import format_timing, printmd
from IPython.display import display

FREQ_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 25),
    'gamma': (25, 45),
}

def calc_rbp(
    segment: np.ndarray,
    sensor: str = 'Fp1',
    total_band: set = (0.5, 45),
    sample_freq: int = 500,
    verbose: bool = False,
    ) -> dict:
    '''Calculate relative band power (RBP) for a given EEG segment.
    
    Args:
        segment (array, required): segment of EEG readings in uV (microvolts)
        sensor (str, optional): Sensor name. Defaults to 'Fp1'.
        total_band (tuple, optional): Lower and upper frequencies of the total
            band. Defaults to (0.5, 45).
        sample_freq (int, optional): Sampling frequency in Hz. Defaults to 500.
        verbose (bool, optional): Whether or not to print information. Defaults
            to False.
    
    Returns:
        dict: Dictionaries containing the PSDs and RBPs
            for each frequency band. Keys are formatted as
            follows: {sensor}_{band}_{psd/rbp}. For example, the key for the
            PSD of the alpha band for sensor Fp1 would be 'Fp1_alpha_psd'.
    '''
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html
    seg_freqs, seg_psd = welch(segment, fs=sample_freq)

    #calculate PSD in a specific band
    def band_psd(seg_freqs, seg_psd, band):
        idx_band = np.logical_and(seg_freqs >= band[0], seg_freqs <= band[1])
        return np.sum(seg_psd[idx_band])

    #total PSD
    total_power = band_psd(seg_freqs, seg_psd, total_band)
    #psd for each band
    powers = {}
    for k,v in FREQ_BANDS.items():
        if verbose:
            print(f'Calculating {k} band power for {sensor} sensor')
        tmp_power = band_psd(seg_freqs, seg_psd, v)
        powers[sensor + '_' + k + '_psd'] = tmp_power
        try:
            powers[sensor +'_'+ k + '_rbp'] = tmp_power / total_power
        except ZeroDivisionError:
            print(f'ZeroDivisionError: {sensor} {k} band power is 0.0')
            powers[sensor +'_'+ k + '_rbp'] = 0.0
        
        if verbose:
            print(f'{k} band power: {tmp_power:.2f}')
            print(f'{k} band RBP: {powers[sensor + "_" + k + "_rbp"]:.2f}')

    return powers

def create_epochs(
    df: pd.DataFrame,
    sensor_names: list,
    seg_dur: int = 4000, #ms
    overlap: float = 0.5, #percent overlap btwn segments
    debug: bool = False,
    verbose: bool = False,
    disp_summary: bool = True
    ) -> pd.DataFrame:
    '''Create epochs from a DataFrame of EEG data.
    
    Args:
        df (DataFrame, required): DataFrame containing EEG data.
        seg_dur (int, optional): Duration of each segment in ms. Defaults to 4000.
        overlap (float, optional): Percent overlap between segments. Defaults to 0.5.
        debug (bool, optional): Whether or not to print debug information. Defaults to False.
        verbose (bool, optional): Whether or not to print verbose information. Defaults to False.
    
    Returns:
        list: List of DataFrames, each containing an epoch.
    '''
    
    step_size_ms = int(seg_dur * (1 - overlap))  #step size to account for overlap
    dfs = []
    s_time = default_timer()
    for c, p_id in enumerate(df['participant_id'].unique()):
        if c > 1 and debug:
            break
        
        p_time = default_timer()
        
        tmp = df[df['participant_id'] == p_id]
        max_time = tmp['time_ms'].max()
        rows = []
        for seg_start in range(0, max_time + step_size_ms, step_size_ms):
            seg_end = seg_start + seg_dur
            seg = tmp[(tmp['time_ms'] >= seg_start) & (tmp['time_ms'] < seg_end)]
            
            if not seg.empty:
                cols = []
                for sens in sensor_names:
                    seg_array = seg[sens].to_numpy()
                    
                    #calc rbp for each sensor
                    tmp_power = calc_rbp(seg_array,sens, verbose=False)
                    tmp_power = pd.DataFrame(tmp_power, index = [0])
                    
                    # track the sensor
                    cols.append(tmp_power)
                
                tmp_row = pd.concat(cols, axis=1)
                tmp_row.insert(0, 'participant_id', p_id)
                tmp_row.insert(1, 'seg_start', seg_start)
                tmp_row.insert(2, 'seg_end', seg_end)
                rows.append(tmp_row)
                
            elif seg_start <= max_time:
                #only print if there actually should be data
                print(f'P {p_id} has no data between {seg_start} and {seg_end} ms')
            else:
                pass
        
        if verbose: 
                print(f'Data for {p_id} processed. | Total Segments: {len(rows):,} | Time: {format_timing(default_timer() - p_time)} | Elapsed: {format_timing(default_timer() - s_time)}')
        
        dfs.append(pd.concat(rows))

    epoch_data = pd.concat(dfs, ignore_index=True)
    
    if disp_summary:
        printmd('### <u>Epoch Data Sample Preview:</u>')
        print(f'Number of participants: {len(epoch_data.participant_id.unique())}')
        print(f'Total of segments: {len(epoch_data):,} | Avg. Segments per Participant: {len(epoch_data)/len(epoch_data.participant_id.unique()):.2f}')
        print('Total Processing Time:', format_timing(default_timer() - s_time))
        display(epoch_data)
    
    return epoch_data