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

    trig_data = raw.get_data(picks=trig_ch)

    trig_diff = np.diff(trig_data)[0]
    trig_peaks, _ = find_peaks(abs(trig_diff), distance=2000)

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
    eeg = raw.pick_types(meg=False, eeg=True)
    
    # Filter Data
    # eeg_filtered = eeg.filter(0, 10, n_jobs=-1)
    # data = eeg_filtered.get_data()[0]

    data = eeg.get_data()[0]

    diff1 = np.diff(data)
    diff10 = data[10:] - data[:-10]
    diff100 = data[100:] - data[:-100]
    diff200 = data[200:] - data[:-200]
    meandiff = np.array([], dtype=np.float)

    print('Calculating Mean-Diff...')
    start_time = time.time()
    for idx, item in enumerate(data[2000:-2000]):
        meandiff = np.append(meandiff, np.mean(data[idx-2000:idx]) - np.mean(data[idx:idx+2000]))
    print(f'Calculating Mean-Diff took {round(time.time() - start_time, 2)} s')

    plt.figure()
    plt.plot(data[160000:200000], label='data')
    plt.plot(diff1[160000:200000], label='diff1')
    plt.plot(diff10[160000:200000], label='diff10')
    plt.plot(diff100[160000:200000], label='diff100')
    plt.plot(diff200[160000:200000], label='diff200')
    plt.plot(meandiff[160000:200000], label='meandiff')
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
