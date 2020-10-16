import mne
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import time
import math

from mne_pipeline_hd.basic_functions.operations import find_6ch_binary_events, find_events
from mne_pipeline_hd.pipeline_functions.decorators import small_func


def _get_trig_ch(raw):
    if raw.info['nchan'] > 160:
        trig_ch = 'EEG 064'
    elif raw.info['nchan'] > 130 and raw.info['nchan'] < 160:
        trig_ch = 'EEG 029'
    elif raw.info['nchan'] == 1:
        trig_ch = '29/Weight'
    else:
        trig_ch = 'EEG 001'

    return trig_ch


def _get_load_cell_trigger(raw):
    trig_ch = _get_trig_ch(raw)
    pd_data = raw.to_data_frame(picks=trig_ch)
    eeg_series = pd_data[trig_ch]

    # Difference of Rolling Mean on both sides of each value, window=2000
    rolling_left1000 = eeg_series.rolling(1000, min_periods=1).mean()
    rolling_right1000 = eeg_series.iloc[::-1].rolling(1000, min_periods=1).mean()
    rolling_diff1000 = rolling_left1000 - rolling_right1000

    # Difference of Rolling Mean on both sides of each value, window=100
    rolling_left100 = eeg_series.rolling(100, min_periods=1).mean()
    rolling_right100 = eeg_series.iloc[::-1].rolling(100, min_periods=1).mean()
    rolling_diff100 = rolling_left100 - rolling_right100

    # Difference of Rolling Standard-Deviation on both sides of each value, window=200
    rolling_leftstd200 = eeg_series.rolling(200, min_periods=1).std()
    rolling_rightstd200 = eeg_series.iloc[::-1].rolling(200, min_periods=1).std()
    rolling_diffstd200 = rolling_leftstd200 - rolling_rightstd200

    # Difference of Rolling Standard-Deviation on both sides of each value, window=100
    rolling_leftstd100 = eeg_series.rolling(100, min_periods=1).std()
    rolling_rightstd100 = eeg_series.iloc[::-1].rolling(100, min_periods=1).std()
    rolling_diffstd100 = rolling_leftstd100 - rolling_rightstd100

    std2000 = np.std(rolling_diff1000)
    stdstd100 = np.std(rolling_diffstd100) * 2

    rd1000_peaks, rd1000_props = find_peaks(abs(rolling_diff1000), height=std2000, distance=3000)
    rd100_peaks, rd100_props = find_peaks(abs(rolling_diff100), distance=100)
    rdstd200_peaks, rdstd200_props = find_peaks(abs(rolling_diffstd200), distance=100)
    rdstd100_peaks, rdstd100_props = find_peaks(abs(rolling_diffstd100), height=stdstd100, distance=100)

    return pd_data, rolling_diff1000, rolling_diff100, rolling_diffstd200, rolling_diffstd100, \
           rd1000_peaks, rd100_peaks, rdstd200_peaks, rdstd100_peaks, std2000, stdstd100


@small_func
def cross_correlation(x, y):
    # nx = (len(x) + len(y)) // 2
    # if nx != len(y):
    #     raise ValueError('x and y must be equal length')
    correls = np.correlate(x, y, mode="same")
    # Normed correlation values
    correls /= np.sqrt(np.dot(x, x) * np.dot(y, y))
    maxlags = int(len(correls) / 2)
    lags = np.arange(-maxlags, maxlags + 1)

    max_lag = lags[np.argmax(np.abs(correls))]
    max_val = correls[np.argmax(np.abs(correls))]

    return lags, correls, max_lag, max_val


def _get_load_cell_trigger_model(sub, min_duration, shortest_event, adjust_timeline_by_msec):
    raw = sub.load_raw()

    trig_ch = _get_trig_ch(raw)
    eeg_raw = raw.copy().pick(trig_ch)
    eeg_series = eeg_raw.to_data_frame()[trig_ch]

    model_signal_down = np.append(np.full(1000, 1), np.full(1000, 0))
    model_signal_up = np.append(np.full(1000, 0), np.full(1000, 1))

    find_6ch_binary_events(sub, min_duration, shortest_event, adjust_timeline_by_msec)
    events = sub.load_events()

    event_id = {'Matlab-Start': 32}
    eeg_epochs = mne.Epochs(eeg_raw, events, event_id=event_id, tmin=0, tmax=7, baseline=None)
    epochs_data = eeg_epochs.get_data()

    fig, axes = plt.subplots(3, 1)
    for ep in epochs_data:
        ep_data = ep[0]

        lag_down, correl_down, max_lag_down, max_val_down = cross_correlation(ep_data, model_signal_down)
        lag_up, correl_up, max_lag_up, max_val_up = cross_correlation(ep_data, model_signal_up)

        axes[0].plot(lag_down, correl_down)
        axes[0].plot(lag_up, correl_up)
        axes[0].plot(max_lag_down, max_val_up, 'x')
        axes[0].plot(max_lag_up, max_val_up, 'x')

        # Amplify eeg-data
        amp_factor = 10**abs(round(math.log(np.mean(abs(ep_data)), 10) * -1) + 1)
        axes[1].plot(ep_data[max_lag_down - 500:max_lag_down + 500] * amp_factor)
        axes[1].plot(model_signal_down[500:-500])
        axes[2].plot(ep_data[max_lag_up - 500:max_lag_up + 500] * amp_factor)
        axes[2].plot(model_signal_up[500:-500])

    axes[0].set_title('Correlation Values TriggerData x ModelData')
    axes[1].set_title('Trigger + Model-Data Down')
    axes[2].set_title('Trigger + Model-Data Up')

    fig.show()


def get_load_cell_events(sub, min_duration, shortest_event, adjust_timeline_by_msec):
    raw = sub.load_raw()

    pd_data, rolling_diff1000, rolling_diff100, rolling_diffstd200, rolling_diffstd100, \
    rd1000_peaks, rd100_peaks, rdstd200_peaks, rdstd100_peaks, std2000, stdstd100 = _get_load_cell_trigger(raw)

    find_6ch_binary_events(sub, min_duration, shortest_event, adjust_timeline_by_msec)
    events = sub.load_events()

    for pk in rd1000_peaks:

        # get the two closest peaks of rdstd100 to the rd2000-peak with different signs
        dist100 = rdstd100_peaks - pk
        close_dist100 = abs(dist100).argsort()
        if np.sign(dist100[close_dist100[0]]) != np.sign(dist100[close_dist100[1]]):
            close2_dist100 = close_dist100[:2]
        else:
            close2_dist100 = [close_dist100[0]]
            # count = 1
            # while np.sign(dist100[close_dist100[0]]) == np.sign(dist100[close_dist100[count]]):
            #     count += 1
            # close2_dist100 = np.append(close_dist100[0], close_dist100[count])

        if rolling_diff1000[pk] > 0:
            events = np.append(events, [[pk, 0, 2]], axis=0)
            for std_pkx in close2_dist100:
                std_pk = rdstd100_peaks[std_pkx]
                if dist100[std_pkx] < 0:
                    events = np.append(events, [[std_pk, 0, 1]], axis=0)
                else:
                    events = np.append(events, [[std_pk, 0, 3]], axis=0)
        else:
            events = np.append(events, [[pk, 0, 5]], axis=0)
            for std_pkx in close2_dist100:
                std_pk = rdstd100_peaks[std_pkx]
                if dist100[std_pkx] < 0:
                    events = np.append(events, [[std_pk, 0, 4]], axis=0)
                else:
                    events = np.append(events, [[std_pk, 0, 6]], axis=0)

    print(f'{len(events)} events found for {sub.name}')

    # Adjust for first sample
    events[:, 0] += raw.first_samp

    # sort events
    events = events[events[:, 0].argsort()]
    # Todo: Somehow(in pltest1_256_au), there are uniques left
    while (len(events[:, 0]) != len(np.unique(events[:, 0]))):
        # Remove duplicates
        uniques, inverse, counts = np.unique(events[:, 0], return_inverse=True, return_counts=True)
        duplicates = uniques[np.nonzero(counts != 1)]

        for dpl in duplicates:
            events = np.delete(events, np.nonzero(events[:, 0] == dpl)[0][0], axis=0)

    sub.save_events(events)


def plot_load_cell_epochs(sub):
    raw = sub.load_raw()
    trig_ch = _get_trig_ch(raw)
    eeg_raw = raw.copy().pick(trig_ch)
    events = sub.load_events()

    event_id = {'Down': 2, 'Up': 5}
    eeg_epochs = mne.Epochs(eeg_raw, events, event_id=event_id, tmin=-2, tmax=2, baseline=None)
    eeg_epochs.plot(title=sub.name, event_id=event_id)

    data = eeg_epochs.get_data()
    plt.figure()
    for ep in data:
        plt.plot(range(-2000, 2001), ep[0])
        plt.plot(0, data[2001], 'x')

    plt.title('All Epochs together')
    plt.show()


def plot_part_trigger(sub):
    raw = sub.load_raw()

    # raw = raw.filter(0, 20, n_jobs=-1)
    trig_ch = _get_trig_ch(raw)

    pd_data, rolling_diff1000, rolling_diff100, rolling_diffstd200, rolling_diffstd100, \
    rd1000_peaks, rd100_peaks, rdstd200_peaks, rdstd100_peaks, std2000, stdstd100 = _get_load_cell_trigger(raw)

    tmin = 80000
    tmax = 120000

    plt.figure()
    plt.plot(pd_data[trig_ch][tmin:tmax], label='data')
    plt.plot(pd_data[trig_ch][tmin:tmax], 'ok', label='data_dots')
    plt.plot(rolling_diff1000[tmin:tmax], label='RollingDiff2000')
    plt.plot(rolling_diff100[tmin:tmax], label='RollingDiff100')
    plt.plot(rolling_diffstd200[tmin:tmax], label='RollingDiffStd200')
    plt.plot(rolling_diffstd100[tmin:tmax], label='RollingDiffStd100')
    plt.plot([p for p in rd1000_peaks if tmin < p < tmax],
             [rolling_diff1000[p] for p in rd1000_peaks if tmin < p < tmax], 'x', label='RollingDiff2000-Peaks')
    plt.plot(range(tmin, tmax), np.full(tmax - tmin, std2000), label='RollingDiff2000-Std')
    plt.plot(range(tmin, tmax), np.full(tmax - tmin, stdstd100), label='RollingDiffStd100-Std')
    plt.plot([p for p in rd100_peaks if tmin < p < tmax],
             [rolling_diff100[p] for p in rd100_peaks if tmin < p < tmax], 'x', label='RollingDiff100-Peaks')
    plt.plot([p for p in rdstd200_peaks if tmin < p < tmax],
             [rolling_diffstd200[p] for p in rdstd200_peaks if tmin < p < tmax], 'x',
             label='RollingDiffStd200-Peaks')
    plt.plot([p for p in rdstd100_peaks if tmin < p < tmax],
             [rolling_diffstd100[p] for p in rdstd100_peaks if tmin < p < tmax], 'x',
             label='RollingDiffStd100-Peaks')
    plt.legend()
    plt.show()


def plot_load_cell_trigger_raw(sub):
    raw = sub.load_raw()
    trig_ch = _get_trig_ch(raw)
    eeg_raw = raw.pick(trig_ch)
    try:
        events = sub.load_events()
    except FileNotFoundError:
        get_load_cell_events(sub)
        events = sub.load_events()
    eeg_raw.plot(events, duration=30, scalings='auto', title=sub.name)
