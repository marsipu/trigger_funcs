import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import time


def get_trigger_only_meg(sub):
    raw = sub.load_raw()

    if raw.info['nchan'] > 130:
        trig_ch = ['EEG 029']
    elif raw.info['nchan'] == 1:
        trig_ch = None
    else:
        trig_ch = ['EEG 001']

    pd_data = raw.to_data_frame(picks=['EEG 001'])
    eeg_series = pd_data['EEG 001']

    rolling_left2000 = eeg_series.rolling(2000, min_periods=1).mean()
    rolling_right2000 = eeg_series.iloc[::-1].rolling(2000, min_periods=1).mean()
    rolling_diff2000 = rolling_left2000 - rolling_right2000

    rolling_left100 = eeg_series.rolling(100, min_periods=1).mean()
    rolling_right100 = eeg_series.iloc[::-1].rolling(100, min_periods=1).mean()
    rolling_diff100 = rolling_left100 - rolling_right100

    rolling_leftstd100 = eeg_series.rolling(100, min_periods=1).std()
    rolling_rightstd100 = eeg_series.iloc[::-1].rolling(100, min_periods=1).std()
    rolling_diffstd100 = rolling_leftstd100 - rolling_rightstd100

    events = np.hstack((np.expand_dims(trig_peaks, axis=1), np.full((len(trig_peaks), 2), 0)))

    for n, event in enumerate(events):
        if trig_diff[event[0]] < 0:
            events[n, 2] = 1
        elif trig_diff[event[0]] > 0:
            events[n, 2] = 2

    print(f'{len(events)} events found for {sub.name}')

    sub.save_events(events)


def plot_part_trigger(sub):
    raw = sub.load_raw()

    # raw = raw.filter(0, 20, n_jobs=-1)

    pd_data = raw.to_data_frame(picks=['EEG 001'])
    eeg_series = pd_data['EEG 001']

    rolling_left2000 = eeg_series.rolling(2000, min_periods=1).mean()
    rolling_right2000 = eeg_series.iloc[::-1].rolling(2000, min_periods=1).mean()
    rolling_diff2000 = rolling_left2000 - rolling_right2000
    rd2000_peaks = find_peaks(rolling_diff2000)

    rolling_left100 = eeg_series.rolling(100, min_periods=1).mean()
    rolling_right100 = eeg_series.iloc[::-1].rolling(100, min_periods=1).mean()
    rolling_diff100 = rolling_left100 - rolling_right100
    rd100_peaks = find_peaks(rolling_diff100)

    rolling_leftstd200 = eeg_series.rolling(200, min_periods=1).std()
    rolling_rightstd200 = eeg_series.iloc[::-1].rolling(200, min_periods=1).std()
    rolling_diffstd200 = rolling_leftstd200 - rolling_rightstd200
    rdstd200_peaks = find_peaks()

    rolling_leftstd100 = eeg_series.rolling(100, min_periods=1).std()
    rolling_rightstd100 = eeg_series.iloc[::-1].rolling(100, min_periods=1).std()
    rolling_diffstd100 = rolling_leftstd100 - rolling_rightstd100

    plt.figure()
    plt.plot(eeg_series[160000:200000], label='data')
    plt.plot(rolling_diff2000[160000:200000], label='RollingDiff2000')
    plt.plot(rolling_diff100[160000:200000], label='RollingDiff100')
    plt.plot(rolling_diffstd200[160000:200000], label='RollingDiffStd200')
    plt.plot(rolling_diffstd100[160000:200000], label='RollingDiffStd100')
    plt.legend()
    plt.show()


def only_eeg(sub):
    raw_filtered = sub.load_filtered()
    if raw_filtered.info['nchan'] > 130:
        trig_ch = ['EEG 029']
    elif raw_filtered.info['nchan'] == 1:
        trig_ch = None
    else:
        trig_ch = ['EEG 001']
    raw_filtered = raw_filtered.pick(trig_ch)

    sub.save_filtered(raw_filtered)
