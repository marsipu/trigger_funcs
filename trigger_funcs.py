import math
import sys
from os.path import join

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import scipy
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QComboBox, QDialog, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QSpinBox, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from mne.io import RawArray
from scipy.signal import find_peaks, savgol_filter
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from mne_pipeline_hd.pipeline.loading import MEEG, Group
from mne_pipeline_hd.functions.operations import find_6ch_binary_events

def _get_load_cell_trigger(raw, trig_channel):
    eeg_raw = raw.copy().pick(trig_channel)
    # eeg_raw = eeg_raw.filter(0, 20, n_jobs=-1)
    eeg_series = eeg_raw.to_data_frame()[trig_channel]

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

    std1000 = np.std(rolling_diff1000)
    stdstd100 = np.std(rolling_diffstd100) * 2

    rd1000_peaks, rd1000_props = find_peaks(abs(rolling_diff1000), height=std1000, distance=2500)
    rd100_peaks, rd100_props = find_peaks(abs(rolling_diff100), distance=100)
    rdstd200_peaks, rdstd200_props = find_peaks(abs(rolling_diffstd200), distance=100)
    rdstd100_peaks, rdstd100_props = find_peaks(abs(rolling_diffstd100), height=stdstd100, distance=100)

    return (eeg_series, rolling_diff1000, rolling_diff100, rolling_diffstd200, rolling_diffstd100,
            rd1000_peaks, rd100_peaks, rdstd200_peaks, rdstd100_peaks, std1000, stdstd100)


def get_load_cell_events(meeg, min_duration, shortest_event, adjust_timeline_by_msec, trig_channel):
    raw = meeg.load_raw()

    (eeg_series, rolling_diff1000, rolling_diff100, rolling_diffstd200, rolling_diffstd100,
     rd1000_peaks, rd100_peaks, rdstd200_peaks, rdstd100_peaks, std2000, stdstd100) = _get_load_cell_trigger(raw, trig_channel)

    find_6ch_binary_events(meeg, min_duration, shortest_event, adjust_timeline_by_msec)
    events = meeg.load_events()

    for pk in rd1000_peaks:
        sp = np.asarray(eeg_series[pk - 500:pk + 500])
        rd100 = np.asarray(rolling_diff100[pk - 500:pk + 500])
        # Correct Offset so the first 250 ms are around zero
        spoff = sp - np.mean(sp[:250])

        diff_times = events[:, 0] - (pk + raw.first_samp)
        neg_diff_times = diff_times[np.nonzero(diff_times < 0)]
        if len(neg_diff_times) > 0:
            previous_id = int(events[np.nonzero(diff_times == np.max(neg_diff_times)), 2])
        else:
            previous_id = 33

        if previous_id == 33 or previous_id == 1:
            try:
                # Get last index under some threshold for rd100
                rd100lastidx = np.nonzero(rd100[:500] < np.std(rd100[:250]) * 2)[0][-1]
                # Get first value under some threshold for spoff
                trig_idx = 500 - (np.nonzero(spoff[rd100lastidx:500] < np.std(spoff[:250]) * -3)[0][0] + rd100lastidx)
                trig_time = pk - trig_idx + raw.first_samp
                events = np.append(events, [[trig_time, 0, 5]], axis=0)
            except IndexError:
                events = np.append(events, [[pk + raw.first_samp, 0, 5]], axis=0)
        elif previous_id == 2:
            try:
                # Get last index above some threshold for rd100
                rd100lastidx = np.nonzero(rd100[:500] > np.std(rd100[:250]) * -2)[0][-1]
                # Get first value above some threshold for spoff
                trig_idx = 500 - (np.nonzero(spoff[rd100lastidx:500] > np.std(spoff[:250]) * 3)[0][0] + rd100lastidx)
                trig_time = pk - trig_idx + raw.first_samp
                events = np.append(events, [[trig_time, 0, 6]], axis=0)
            except IndexError:
                events = np.append(events, [[pk + raw.first_samp, 0, 6]], axis=0)

    print(f'{len(events)} events found for {meeg.name}')

    # sort events
    events = events[events[:, 0].argsort()]
    # Todo: Somehow(in pltest1_256_au), there are uniques left
    while len(events[:, 0]) != len(np.unique(events[:, 0])):
        # Remove duplicates
        uniques, inverse, counts = np.unique(events[:, 0], return_inverse=True, return_counts=True)
        duplicates = uniques[np.nonzero(counts != 1)]

        for dpl in duplicates:
            events = np.delete(events, np.nonzero(events[:, 0] == dpl)[0][0], axis=0)

    print(f'Found {len(np.nonzero(events[:, 2] == 5)[0])} Events for Down')
    print(f'Found {len(np.nonzero(events[:, 2] == 6)[0])} Events for Up')

    meeg.save_events(events)


def get_load_cell_events_regression(meeg, min_duration, shortest_event, adjust_timeline_by_msec,
                                    diff_window, min_ev_distance, max_ev_distance, regression_degree,
                                    regression_range, trig_channel, n_jobs):
    # Load Raw and extract the load-cell-trigger-channel
    raw = meeg.load_raw()
    eeg_raw = raw.copy().pick(trig_channel)
    eeg_series = eeg_raw.to_data_frame()[trig_channel]

    # Difference of Rolling Mean on both sides of each value
    rolling_left = eeg_series.rolling(diff_window, min_periods=1).mean()
    rolling_right = eeg_series.iloc[::-1].rolling(diff_window, min_periods=1).mean()
    rolling_diff = rolling_left - rolling_right

    # Find peaks of the Rolling-Difference
    rd_peaks, _ = find_peaks(abs(rolling_diff), height=np.std(rolling_diff), distance=min_ev_distance)

    # Find the other events encoded by the binary channel
    find_6ch_binary_events(meeg, min_duration, shortest_event, adjust_timeline_by_msec)
    events = meeg.load_events()

    events_meta = dict()
    reg_min, reg_max = regression_range

    # Iterate through the peaks found in the rolling difference
    for ev_idx, pk in enumerate(rd_peaks):
        # +1 -> Account for leftwards shift in indexing
        data = np.asarray(eeg_series[int(pk - reg_max):int(pk + reg_max + 1)])

        if ev_idx != len(rd_peaks) - 1:
            sys.stderr.write(f'\rProgress: {int((ev_idx + 1) / len(rd_peaks) * 100)} %')
        else:
            sys.stderr.write(f'\rProgress: {int((ev_idx + 1) / len(rd_peaks) * 100)} %\n')

        # Get closest peak to determine down or up
        if ev_idx == 0:
            if rd_peaks[1] - pk < max_ev_distance:
                # Get first trigger if up follows in under min_ev_distance
                direction = 'down'
                event_id = 5
            else:
                continue
        elif ev_idx == len(rd_peaks) - 1:
            if pk - rd_peaks[ev_idx - 1] < max_ev_distance:
                # Get last trigger if down was before under min_ev_distance
                direction == 'up'
                event_id = 6
            else:
                continue
        elif rd_peaks[ev_idx + 1] - pk < max_ev_distance:
            direction = 'down'
            event_id = 5
        elif pk - rd_peaks[ev_idx - 1] < max_ev_distance:
            direction = 'up'
            event_id = 6
        else:
            continue

        # Containers to store the results ot the linear-regressions of multiple ranges
        events_meta[ev_idx] = dict()
        scores = list()
        models = dict()

        # Find the best range for a regression fit (going symmetrically from the rolling-difference-peak)
        for n in range(reg_min, reg_max):
            # Create the x-array with appropriate shape and transformed
            x = PolynomialFeatures(degree=regression_degree, include_bias=False) \
                .fit_transform(np.arange(n * 2 + 1).reshape(-1, 1))
            y = data[reg_max - n:reg_max + n + 1]
            model = LinearRegression(n_jobs=n_jobs).fit(x, y)
            scores.append(model.score(x, y))
            models[n] = model

        # Get best model with best_index (which is also the best symmetrical range from the peak)
        best_score = max(scores)
        best_idx = scores.index(max(scores)) + reg_min  # Compensate for reg_min because of range(reg_min, ...) above
        best_model = models[best_idx]
        poly_x = PolynomialFeatures(degree=regression_degree, include_bias=False) \
            .fit_transform(np.arange(best_idx * 2).reshape(-1, 1))
        best_y = best_model.predict(poly_x)
        best_coef = best_model.coef_

        first_time = pk - best_idx + raw.first_samp
        last_time = pk + best_idx + raw.first_samp

        # Store information about event in meta-dict
        events_meta[ev_idx]['best_score'] = best_score
        events_meta[ev_idx]['first_time'] = first_time
        events_meta[ev_idx]['last_time'] = last_time
        events_meta[ev_idx]['best_y'] = best_y
        events_meta[ev_idx]['best_coef'] = best_coef
        events_meta[ev_idx]['direction'] = direction

        # add to events
        events = np.append(events, [[first_time, 0, event_id]], axis=0)

    # sort events by time (first column)
    events = events[events[:, 0].argsort()]

    # Remove duplicates
    while len(events[:, 0]) != len(np.unique(events[:, 0])):
        uniques, inverse, counts = np.unique(events[:, 0], return_inverse=True, return_counts=True)
        duplicates = uniques[np.nonzero(counts != 1)]

        for dpl in duplicates:
            events = np.delete(events, np.nonzero(events[:, 0] == dpl)[0][0], axis=0)
            print(f'Removed duplicate at {dpl}')

    print(f'Found {len(np.nonzero(events[:, 2] == 5)[0])} Events for Down')
    print(f'Found {len(np.nonzero(events[:, 2] == 6)[0])} Events for Up')

    # Save events
    meeg.save_events(events)

    # Save event-meta
    meeg.save_json('load_events_meta', events_meta)

    # Save Trigger-Raw with correlation-signal for plotting
    reg_signal = np.asarray([])
    for idx, ev_idx in enumerate(events_meta):
        first_time = events_meta[ev_idx]['first_time'] - eeg_raw.first_samp
        best_y = events_meta[ev_idx]['best_y']

        if idx == 0:
            # Fill the time before the first event
            reg_signal = np.concatenate([reg_signal, np.full(first_time, best_y[0]), best_y])
        else:
            # Get previous index even when it is missing
            n_minus = 1
            previous_idx = None
            while True:
                try:
                    events_meta[ev_idx - n_minus]
                except KeyError:
                    n_minus += 1
                    if ev_idx - n_minus < 0:
                        break
                else:
                    previous_idx = ev_idx - n_minus
                    break
            if idx == len(events_meta) - 1:
                # Fill the time before and after the last event
                first_fill_time = first_time - (events_meta[previous_idx]['last_time'] - eeg_raw.first_samp)
                last_fill_time = eeg_raw.n_times - (events_meta[ev_idx]['last_time'] - eeg_raw.first_samp)
                reg_signal = np.concatenate([reg_signal,
                                             np.full(first_fill_time, best_y[0]),
                                             best_y,
                                             np.full(last_fill_time, best_y[-1])])
            else:
                # Fill the time between events
                fill_time = first_time - (events_meta[previous_idx]['last_time'] - eeg_raw.first_samp)
                reg_signal = np.concatenate([reg_signal, np.full(fill_time, best_y[0]), best_y])

    # Fit scalings back to eeg_raw
    reg_signal /= 1e6
    eeg_signal = eeg_raw.get_data()[0]
    reg_info = mne.create_info(ch_names=['reg_signal', 'lc_signal'],
                               ch_types=['eeg', 'eeg'], sfreq=eeg_raw.info['sfreq'])
    reg_raw = mne.io.RawArray([reg_signal, eeg_signal], reg_info)
    reg_raw_path = join(meeg.save_dir, f'{meeg.name}_{meeg.p_preset}_loadcell-regression-raw.fif')
    reg_raw.save(reg_raw_path, overwrite=True)
    meeg.save_file_params(reg_raw_path)


def get_load_cell_events_regression_baseline(meeg, min_duration, shortest_event, adjust_timeline_by_msec,
                                             diff_window, min_ev_distance, max_ev_distance, len_baseline,
                                             baseline_limit, regression_degree, trig_channel, n_jobs):
    from string import ascii_lowercase

    # Load Raw and extract the load-cell-trigger-channel
    raw = meeg.load_raw()
    eeg_raw = raw.copy().pick(trig_channel)
    eeg_series = eeg_raw.to_data_frame()[trig_channel]

    # Difference of Rolling Mean on both sides of each value
    rolling_left = eeg_series.rolling(diff_window, min_periods=1).mean()
    rolling_right = eeg_series.iloc[::-1].rolling(diff_window, min_periods=1).mean()
    rolling_diff = rolling_left - rolling_right

    # Find peaks of the Rolling-Difference
    rd_peaks, _ = find_peaks(abs(rolling_diff), height=np.std(rolling_diff), distance=min_ev_distance)

    try:
        # Find the other events encoded by the binary channel
        find_6ch_binary_events(meeg, min_duration, shortest_event, adjust_timeline_by_msec)
        events = meeg.load_events()
    except ValueError:
        events = np.asarray([[0, 0, 0]])

    events_meta_dict = dict()
    events_meta_pd = pd.DataFrame([])

    # Iterate through the peaks found in the rolling difference
    for ev_idx, pk in enumerate(rd_peaks):
        if ev_idx != len(rd_peaks) - 1:
            sys.stderr.write(f'\rProgress: {int((ev_idx + 1) / len(rd_peaks) * 100)} %')
        else:
            sys.stderr.write(f'\rProgress: {int((ev_idx + 1) / len(rd_peaks) * 100)} %\n')

        # Get closest peak to determine down or up

        # Get first trigger if up follows in under min_ev_distance
        if ev_idx == 0:
            if rd_peaks[1] - pk < max_ev_distance:
                direction = 'down'
            else:
                continue

        # Get last trigger if down was before under min_ev_distance
        elif ev_idx == len(rd_peaks) - 1:
            if pk - rd_peaks[ev_idx - 1] < max_ev_distance:
                direction = 'up'
            else:
                continue

        # Get other peaks
        elif rd_peaks[ev_idx + 1] - pk < max_ev_distance:
            direction = 'down'

        elif pk - rd_peaks[ev_idx - 1] < max_ev_distance:
            direction = 'up'

        else:
            continue

        # Get Trigger-Time by finding the first samples going from peak crossing the baseline
        # (from baseline_limit with length=len_baseline)
        pre_baseline_mean = np.asarray(eeg_series[pk - (len_baseline + baseline_limit):pk - baseline_limit + 1]).mean()
        post_baseline_mean = np.asarray(eeg_series[pk + baseline_limit:pk + baseline_limit + len_baseline + 1]).mean()
        pre_peak_data = np.flip(np.asarray(eeg_series[pk - min_ev_distance:pk + 1]))
        post_peak_data = np.asarray(eeg_series[pk:pk + min_ev_distance + 1])
        if pre_baseline_mean > post_baseline_mean:
            first_idx = pk - (pre_peak_data > pre_baseline_mean).argmax()
            last_idx = pk + (post_peak_data < post_baseline_mean).argmax()
        else:
            first_idx = pk - (pre_peak_data < pre_baseline_mean).argmax()
            last_idx = pk + (post_peak_data > post_baseline_mean).argmax()

        # Do the regression for the data spanned by first-time and last-time
        y = eeg_series[first_idx:last_idx]
        x = PolynomialFeatures(degree=regression_degree, include_bias=False) \
            .fit_transform(np.arange(len(y)).reshape(-1, 1))
        model = LinearRegression(n_jobs=n_jobs).fit(x, y)

        score = model.score(x, y)
        y_pred = model.predict(x)
        coef = model.coef_

        peak_time = pk + raw.first_samp
        first_time = first_idx + raw.first_samp
        last_time = last_idx + raw.first_samp

        # Store information about event in meta-dict
        events_meta_dict[ev_idx] = dict()
        events_meta_dict[ev_idx]['best_score'] = score
        events_meta_dict[ev_idx]['first_time'] = first_time
        events_meta_dict[ev_idx]['peak_time'] = peak_time
        events_meta_dict[ev_idx]['last_time'] = last_time
        events_meta_dict[ev_idx]['y_pred'] = y_pred
        events_meta_dict[ev_idx]['coef'] = coef
        events_meta_dict[ev_idx]['direction'] = direction

        # Event-ID-naming:
        #   4 = Down-First
        #   5 = Down-Middle
        #   6 = Down-Last
        #
        #   7 = Up-First
        #   8 = Up-Middle
        #   9 = Up-Last

        if direction == 'down':
            event_id_first = 4
            event_id_middle = 5
            event_id_last = 6
        else:
            event_id_first = 7
            event_id_middle = 8
            event_id_last = 9

        for timep, evid in zip([first_time, peak_time, last_time],
                               [event_id_first, event_id_middle, event_id_last]):
            meta_dict = {'time': timep, 'score': score, 'direction': direction, 'id': evid,
                         'coef': coef[0]}
            # coef_dict = {k: v for k, v in zip(list(ascii_lowercase)[:len(coef)], coef)}
            # meta_dict.update(coef_dict)
            meta_series = pd.Series(meta_dict)
            events_meta_pd = events_meta_pd.append(meta_series, ignore_index=True)

        # add to events
        events = np.append(events, [[first_time, 0, event_id_first]], axis=0)
        events = np.append(events, [[peak_time, 0, event_id_middle]], axis=0)
        events = np.append(events, [[last_time, 0, event_id_last]], axis=0)

    # sort events by time (first column)
    events = events[events[:, 0].argsort()]

    # Remove duplicates
    while len(events[:, 0]) != len(np.unique(events[:, 0])):
        uniques, inverse, counts = np.unique(events[:, 0], return_inverse=True, return_counts=True)
        duplicates = uniques[np.nonzero(counts != 1)]

        for dpl in duplicates:
            events = np.delete(events, np.nonzero(events[:, 0] == dpl)[0][0], axis=0)
            print(f'Removed duplicate at {dpl}')

    print(f'Found {len(np.nonzero(events[:, 2] == 4)[0])} Events for Down-First')
    print(f'Found {len(np.nonzero(events[:, 2] == 5)[0])} Events for Down-Middle')
    print(f'Found {len(np.nonzero(events[:, 2] == 6)[0])} Events for Down-Last')
    print(f'Found {len(np.nonzero(events[:, 2] == 7)[0])} Events for Up-First')
    print(f'Found {len(np.nonzero(events[:, 2] == 8)[0])} Events for Up-Middle')
    print(f'Found {len(np.nonzero(events[:, 2] == 9)[0])} Events for Up-Last')

    # Save events
    meeg.save_events(events)

    # Save events-meta dictionary
    meeg.save_json('load_events_meta', events_meta_dict)

    # Save events-meta DataFrame
    file_name = 'load_events_meta'
    file_path = join(meeg.save_dir, f'{meeg.name}_{meeg.p_preset}_{file_name}.csv')
    events_meta_pd['time'] = events_meta_pd['time'].astype(int)
    events_meta_pd.to_csv(file_path)

    # Save Trigger-Raw with correlation-signal for plotting
    reg_signal = np.asarray([])
    for idx, ev_idx in enumerate(events_meta_dict):
        first_time = events_meta_dict[ev_idx]['first_time'] - eeg_raw.first_samp
        best_y = events_meta_dict[ev_idx]['y_pred']

        if idx == 0:
            # Fill the time before the first event
            reg_signal = np.concatenate([reg_signal, np.full(first_time, best_y[0]), best_y])
        else:
            # Get previous index even when it is missing
            n_minus = 1
            previous_idx = None
            while True:
                try:
                    events_meta_dict[ev_idx - n_minus]
                except KeyError:
                    n_minus += 1
                    if ev_idx - n_minus < 0:
                        break
                else:
                    previous_idx = ev_idx - n_minus
                    break
            if idx == len(events_meta_dict) - 1:
                # Fill the time before and after the last event
                first_fill_time = first_time - (events_meta_dict[previous_idx]['last_time'] - eeg_raw.first_samp)
                last_fill_time = eeg_raw.n_times - (events_meta_dict[ev_idx]['last_time'] - eeg_raw.first_samp)
                reg_signal = np.concatenate([reg_signal,
                                             np.full(first_fill_time, best_y[0]),
                                             best_y,
                                             np.full(last_fill_time, best_y[-1])])
            else:
                # Fill the time between events
                fill_time = first_time - (events_meta_dict[previous_idx]['last_time'] - eeg_raw.first_samp)
                reg_signal = np.concatenate([reg_signal, np.full(fill_time, best_y[0]), best_y])

    # Fit scalings back to eeg_raw
    reg_signal /= 1e6
    eeg_signal = eeg_raw.get_data()[0]
    reg_info = mne.create_info(ch_names=['reg_signal', 'lc_signal'],
                               ch_types=['eeg', 'eeg'], sfreq=eeg_raw.info['sfreq'])
    reg_raw = mne.io.RawArray([reg_signal, eeg_signal], reg_info)
    reg_raw_path = join(meeg.save_dir, f'{meeg.name}_{meeg.p_preset}_loadcell-regression-raw.fif')
    reg_raw.save(reg_raw_path, overwrite=True)


def get_ratings(meeg, target_event_id):
    events = meeg.load_events()

    file_name = 'ratings_meta'
    file_path = join(meeg.save_dir, f'{meeg.name}_{meeg.p_preset}_{file_name}.csv')
    rating_meta_pd = pd.DataFrame([], columns=['time', 'id', 'rating'], dtype=int)

    # Get Ratings from Triggers 10-19
    pre_ratings = np.copy(events[np.nonzero(np.logical_and(10 <= events[:, 2], events[:, 2] <= 19))])
    first_idx = np.nonzero(np.diff(pre_ratings[:, 0], axis=0) < 200)[0]
    last_idx = first_idx + 1
    ratings = pre_ratings[first_idx]
    ratings[:, 2] = (ratings[:, 2] - 10) * 10 + pre_ratings[last_idx][:, 2] - 10

    # Get time sample from target_event_id
    target_events = events[np.nonzero(events[:, 2] == target_event_id)]
    for rating in ratings:
        # Get time from previous target_event_id
        try:
            rating_time = target_events[np.nonzero(target_events[:, 0] - rating[0] < 0)][-1][0]
        except IndexError:
            pass
        else:
            # Make sure there are no duplicates (because of missing events)
            if rating_time not in list(rating_meta_pd['time']):
                rating_value = rating[2]
                rating_dict = {'time': rating_time, 'id': target_event_id, 'rating': rating_value}
                meta_series = pd.Series(rating_dict)
                rating_meta_pd = rating_meta_pd.append(meta_series, ignore_index=True)

    rating_meta_pd.to_csv(file_path)


def plot_group_ratings(group, show_plots):
    ratings = list()
    for meeg_name in group.group_list:
        meeg = MEEG(meeg_name, group.mw)
        file_name = 'ratings_meta'
        file_path = join(meeg.save_dir, f'{meeg.name}_{meeg.p_preset}_{file_name}.csv')
        rating_meta_pd = pd.read_csv(file_path, index_col=0)

        ratings.append(list(rating_meta_pd.loc[:, 'rating']))

    fig, ax = plt.subplots()
    ax.boxplot(ratings)
    ax.set_title(f'Ratings for {group.name} for {len(group.group_list)} subjects')

    if show_plots:
        fig.show()

    group.plot_save('group_ratings', matplotlib_figure=fig)


def plot_group_ratings_compared(mw, show_plots):
    all_ratings = dict()
    for group_name in mw.pr.sel_groups:
        group = Group(group_name, mw)
        ratings = list()
        for meeg_name in group.group_list:
            meeg = MEEG(meeg_name, group.mw)
            file_name = 'ratings_meta'
            file_path = join(meeg.save_dir, f'{meeg.name}_{meeg.p_preset}_{file_name}.csv')
            rating_meta_pd = pd.read_csv(file_path, index_col=0)

            ratings.append(np.mean(list(rating_meta_pd.loc[:, 'rating'])))
        all_ratings[group_name] = ratings

    fig, ax = plt.subplots()


def plot_group_lc_coef(group, show_plots):
    ratings = list()
    for meeg_name in group.group_list:
        meeg = MEEG(meeg_name, group.mw)
        file_name = 'load_events_meta'
        file_path = join(meeg.save_dir, f'{meeg.name}_{meeg.p_preset}_{file_name}.csv')
        rating_meta_pd = pd.read_csv(file_path, index_col=0)

        ratings.append(list(rating_meta_pd.loc[:, 'coef']))

    fig, ax = plt.subplots()
    ax.boxplot(ratings)
    ax.set_title(f'Load-Cell-Coef for {group.name} for {len(group.group_list)} subjects')

    if show_plots:
        fig.show()

    group.plot_save('group_ratings', matplotlib_figure=fig)


def _add_events_meta(epochs, meta_pd):
    """Make sure, that meat-data is assigned to correct epoch
    (requires parameter "time" and "id" to be included in meta_pd)
    """
    meta_pd_filtered = meta_pd.loc[meta_pd['id'].isin(epochs.event_id.values()) &
                          meta_pd['time'].isin(epochs.events[:, 0])]

    metatimes = [int(t) for t in meta_pd_filtered['time']]

    # Add missing values
    for miss_ix in np.nonzero(np.isin(epochs.events[:, 0], metatimes, invert=True))[0]:
        miss_time, miss_id = epochs.events[miss_ix, [0, 2]]
        meta_pd_filtered = meta_pd_filtered.append(pd.Series({'time': miss_time, 'id': miss_id}),
                                                   ignore_index=True)

    meta_pd_filtered = meta_pd_filtered.sort_values('time', ascending=True, ignore_index=True)

    # Integrate into existing metadata
    if isinstance(epochs.metadata, pd.DataFrame):
        meta_pd_filtered = pd.merge(epochs.metadata, meta_pd_filtered,
                                    how='inner', on=['time', 'id'])

    if len(meta_pd_filtered) > 0:
        epochs.metadata = meta_pd_filtered
    else:
        raise RuntimeWarning('No metadata fits to this epochs!')


def add_lc_events_meta(meeg):
    epochs = meeg.load_epochs()
    file_name = 'load_events_meta'
    file_path = join(meeg.save_dir, f'{meeg.name}_{meeg.p_preset}_{file_name}.csv')
    meta_pd = pd.read_csv(file_path, index_col=0)

    _add_events_meta(epochs, meta_pd)
    meeg.save_epochs(epochs)


def add_ratings_meta(meeg):
    epochs = meeg.load_epochs()
    file_name = 'ratings_meta'
    file_path = join(meeg.save_dir, f'{meeg.name}_{meeg.p_preset}_{file_name}.csv')
    ratings_pd = pd.read_csv(file_path, index_col=0)

    _add_events_meta(epochs, ratings_pd)
    meeg.save_epochs(epochs)


def select_events_meta(meeg, meta_queries):
    epochs = meeg.load_epochs()
    try:
        evokeds = meeg.load_evokeds()
    except FileNotFoundError:
        evokeds = list()

    for name, mq in meta_queries.items():
        evoked = epochs[mq].average()
        evoked.comment = name
        # Add name to sel_trials to allow later processing in functions relying on sel_trials
        meeg.sel_trials.append(name)
        evokeds.append(evoked)

    meeg.save_evokeds(evokeds)


def remove_metadata(meeg):
    epochs = meeg.load_epochs()
    epochs.metadata = None
    meeg.save_epochs(epochs)


def select_ratings(meeg):
    epochs = meeg.load_epochs()
    try:
        evokeds = meeg.load_evokeds()
    except FileNotFoundError:
        evokeds = list()

    ratings_mean = np.mean(epochs.metadata['rating'])
    evoked_lr = epochs[f'rating < {ratings_mean}'].average()
    evoked_lr.comment = 'Lower Ratings'
    evoked_hr = epochs[f'rating > {ratings_mean}'].average()
    evoked_hr.comment = 'Higher Ratings'
    meeg.sel_trials += ['Lower Ratings', 'Higher Ratings']
    evokeds += [evoked_lr, evoked_hr]

    meeg.save_evokeds(evokeds)


def plot_ratings_comparision(group, show_plots):
    evokeds_lr = list()
    evokeds_hr = list()

    for meeg_name in group.group_list:
        meeg = MEEG(meeg_name, group.mw)
        epochs = meeg.load_epochs()

        try:
            ratings_mean = np.mean(epochs.metadata['rating'])
        except (KeyError, TypeError):
            print(f'{meeg_name} could not be included due to reasons')
        else:
            evoked_lr = epochs[f'rating < {ratings_mean}'].average()
            evoked_lr.comment = 'Lower Ratings'
            evoked_hr = epochs[f'rating > {ratings_mean}'].average()
            evoked_hr.comment = 'Higher Ratings'

            evokeds_lr.append(evoked_lr)
            evokeds_hr.append(evoked_hr)

    ga_lr = mne.grand_average(evokeds_lr)
    ga_lr.comment = 'Lower Ratings'
    ga_hr = mne.grand_average(evokeds_hr)
    ga_hr.comment = 'Higher Ratings'

    fig = mne.viz.plot_compare_evokeds([ga_lr, ga_hr], title=group.name, show=show_plots)
    group.plot_save('compare_ratings', matplotlib_figure=fig)


def plot_coef_comparision(group, show_plots):
    evokeds_lr = list()
    evokeds_hr = list()

    for meeg_name in group.group_list:
        meeg = MEEG(meeg_name, group.mw)
        epochs = meeg.load_epochs()

        try:
            coef_mean = np.mean(epochs.metadata['coef'])
        except (KeyError, TypeError):
            print(f'{meeg_name} could not be included due to reasons')
        else:
            evoked_lr = epochs[f'coef < {coef_mean}'].average()
            evoked_lr.comment = 'Lower Coef'
            evoked_hr = epochs[f'coef > {coef_mean}'].average()
            evoked_hr.comment = 'Higher Coef'

            evokeds_lr.append(evoked_lr)
            evokeds_hr.append(evoked_hr)

    ga_lr = mne.grand_average(evokeds_lr)
    ga_lr.comment = 'Lower Coef'
    ga_hr = mne.grand_average(evokeds_hr)
    ga_hr.comment = 'Higher Coef'

    fig = mne.viz.plot_compare_evokeds([ga_lr, ga_hr], title=group.name, show=show_plots)
    group.plot_save('compare_coef', matplotlib_figure=fig)


def select_load_cell(meeg):
    epochs = meeg.load_epochs()
    try:
        evokeds = meeg.load_evokeds()
    except FileNotFoundError:
        evokeds = list()

    coef_mean = np.mean(epochs.metadata['coef'])
    evoked_lr = epochs[f'rating < {coef_mean}'].average()
    evoked_lr.comment = 'Lower Ratings'
    evoked_hr = epochs[f'rating < {coef_mean}'].average()
    evoked_hr.comment = 'Higher Ratings'
    meeg.sel_trials += ['Lower Ratings', 'Higher Ratings']
    evokeds += [evoked_lr, evoked_hr]

    meeg.save_evokeds(evokeds)


def plot_lc_reg_raw(meeg, show_plots):
    reg_raw_path = join(meeg.save_dir, f'{meeg.name}_{meeg.p_preset}_loadcell-regression-raw.fif')
    reg_raw = mne.io.read_raw_fif(reg_raw_path, preload=True)

    raw = meeg.load_raw()
    events = meeg.load_events()
    events[:, 0] -= raw.first_samp

    reg_raw.plot(events, title=f'{meeg.name}_{meeg.p_preset}-Regression Fit for Load-Cell-Data',
                 butterfly=False, show_first_samp=True, duration=60, show=show_plots)


def plot_lc_reg_ave(meeg, trig_plt_time, show_plots):
    reg_raw_path = join(meeg.save_dir, f'{meeg.name}_{meeg.p_preset}_loadcell-regression-raw.fif')
    reg_raw = mne.io.read_raw_fif(reg_raw_path, preload=True)

    raw = meeg.load_raw()
    events = meeg.load_events()
    events[:, 0] -= raw.first_samp

    trig_plt_tmin, trig_plt_tmax = trig_plt_time

    # if 'last' in trial:
    #     baseline = (0.05, trig_plt_tmax)
    # else:
    baseline = (trig_plt_tmin, -0.05)

    lc_epo_down = mne.Epochs(reg_raw, events, {'Down': 5},
                             tmin=trig_plt_tmin, tmax=trig_plt_tmax, baseline=baseline, picks='lc_signal')
    lc_epo_up = mne.Epochs(reg_raw, events, {'Up': 6},
                           tmin=trig_plt_tmin, tmax=trig_plt_tmax, baseline=baseline, picks='lc_signal')
    reg_epo_down = mne.Epochs(reg_raw, events, {'Down': 5},
                              tmin=trig_plt_tmin, tmax=trig_plt_tmax, baseline=baseline, picks='reg_signal')
    reg_epo_up = mne.Epochs(reg_raw, events, {'Up': 6},
                            tmin=trig_plt_tmin, tmax=trig_plt_tmax, baseline=baseline, picks='reg_signal')

    fig, ax = plt.subplots(2, 6, figsize=(10, 10))

    lc_data_down = lc_epo_down.get_data()
    for ep in lc_data_down:
        ax[0, 0].plot(lc_epo_down.times, ep[0])
        ax[0, 0].plot(0, ep[0][201], 'rx')
    ax[0, 0].set_title('Load-Cell-Signal (Down)')

    lc_data_up = lc_epo_up.get_data()
    for ep in lc_data_up:
        ax[0, 1].plot(lc_epo_up.times, ep[0])
        ax[0, 1].plot(0, ep[0][201], 'rx')
    ax[0, 1].set_title('Load-Cell-Signal (Up)')

    reg_data_down = reg_epo_down.get_data()
    for ep in reg_data_down:
        ax[1, 0].plot(reg_epo_down.times, ep[0])
        ax[1, 0].plot(0, ep[0][201], 'rx')
    ax[1, 0].set_title('Regression-Signal (Down)')

    reg_data_up = reg_epo_up.get_data()
    for ep in reg_data_up:
        ax[1, 1].plot(reg_epo_up.times, ep[0])
        ax[1, 1].plot(0, ep[0][201], 'rx')
    ax[1, 1].set_title('Regression-Signal (Up)')

    fig.suptitle(meeg.name)

    meeg.plot_save('trigger-regression', matplotlib_figure=fig)
    if show_plots:
        fig.show()


def plot_lc_latencies(group, show_plots):
    fig, ax = plt.subplots(figsize=(16, 10))
    box_plots = list()
    box_labels = list()

    for meeg_name in group.group_list:
        meeg = MEEG(meeg_name, group.mw)
        events_meta = meeg.load_json('load_events_meta')
        diffs = [events_meta['Load'][idx]['first_time'] - events_meta['Touch'][idx]['first_time'] for idx in
                 events_meta['Load']]
        box_plots.append(diffs)
        box_labels.append(meeg_name)
    ax.boxplot(box_plots)
    ax.set_xticklabels(box_labels, rotation=45, fontsize=6)
    fig.suptitle(group.name)

    if show_plots:
        fig.show()

    group.plot_save('Touch-Load-Latencies')


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


def _get_load_cell_trigger_model(meeg, min_duration, shortest_event, adjust_timeline_by_msec, trig_channel):
    raw = meeg.load_raw()

    eeg_raw = raw.copy().pick(trig_channel)

    model_signal_down = np.append(np.full(1000, 1), np.full(1000, 0))
    model_signal_up = np.append(np.full(1000, 0), np.full(1000, 1))

    find_6ch_binary_events(meeg, min_duration, shortest_event, adjust_timeline_by_msec)
    events = meeg.load_events()

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
        amp_factor = 10 ** abs(round(math.log(np.mean(abs(ep_data)), 10) * -1) + 1)
        axes[1].plot(ep_data[max_lag_down - 500:max_lag_down + 500] * amp_factor)
        axes[1].plot(model_signal_down[500:-500])
        axes[2].plot(ep_data[max_lag_up - 500:max_lag_up + 500] * amp_factor)
        axes[2].plot(model_signal_up[500:-500])

    axes[0].set_title('Correlation Values TriggerData x ModelData')
    axes[1].set_title('Trigger + Model-Data Down')
    axes[2].set_title('Trigger + Model-Data Up')

    fig.show()


def _get_load_cell_epochs(meeg, trig_plt_time, baseline_limit, trig_channel, apply_savgol=False,):
    raw = meeg.load_raw()
    eeg_raw = raw.copy().pick(trig_channel)
    events = meeg.load_events()

    event_id = meeg.event_id
    trig_plt_tmin, trig_plt_tmax = trig_plt_time

    epochs_dict = dict()
    times = None

    for idx, trial in enumerate(meeg.sel_trials):
        selected_ev_id = {key: value for key, value in event_id.items() if key == trial}
        # if 'Last' in trial:
        #     baseline = (round(baseline_limit/1000, 3), trig_plt_tmax)
        # else:
        #     baseline = (trig_plt_tmin, -round(baseline_limit / 1000, 3))

        eeg_epochs = mne.Epochs(eeg_raw, events, event_id=selected_ev_id,
                                tmin=trig_plt_tmin, tmax=trig_plt_tmax, baseline=None)
        times = eeg_epochs.times
        data = eeg_epochs.get_data()
        baseline_data = list()
        for ep in data:
            epd = ep[0]
            half_idx = int(len(epd)/2) + 1
            if 'Last' in trial:
                epd -= np.mean(epd[half_idx + baseline_limit:])
            else:
                epd -= np.mean(epd[:half_idx-baseline_limit])

            if np.mean(epd[half_idx + baseline_limit:]) < 0 and 'Down' in trial:
                epd *= -1
            elif np.mean(epd[half_idx + baseline_limit:]) > 0 and 'Up' in trial:
                epd *= -1

            if apply_savgol:
                epd = savgol_filter(epd, 201, 5)

            baseline_data.append(epd)

        epochs_dict[trial] = baseline_data

    return epochs_dict, times


def plot_load_cell_ave(meeg, trig_plt_time, baseline_limit, show_plots, apply_savgol, trig_channel):

    epochs_dict, times = _get_load_cell_epochs(meeg, trig_plt_time, baseline_limit, trig_channel, apply_savgol)
    fig, ax = plt.subplots(1, len(meeg.sel_trials), figsize=(5*len(meeg.sel_trials), 8),
                           sharey=True)
    if not isinstance(ax, np.ndarray):
        ax = [ax]
    for idx, trial in enumerate(epochs_dict):
        for epd in epochs_dict[trial]:
            ax[idx].plot(times, epd, color='blue')
            half_idx = int(len(epd) / 2) + 1
            ax[idx].plot(0, epd[half_idx], 'xr')

            ax[idx].set_title(trial)
            ax[idx].set_xlabel('Time [s]')
            if idx == 0:
                ax[idx].set_ylabel('Weight (Relative)')

        fig.suptitle(meeg.name)
        meeg.plot_save('trigger_epochs', matplotlib_figure=fig)

        if show_plots:
            fig.show()


def plot_load_cell_group_ave(mw, trig_plt_time, baseline_limit, show_plots, apply_savgol):
    fig, ax = plt.subplots(len(mw.pr.sel_groups), 1, sharey=False, sharex=True)
    if not isinstance(ax, np.ndarray):
        ax = [ax]

    cmap = plt.cm.get_cmap('twilight', len(mw.pr.all_groups[mw.pr.sel_groups[0]]) + 1)
    for idx, group_name in enumerate(mw.pr.sel_groups):
        group = Group(group_name, mw)
        for color_idx, meeg_name in enumerate(group.group_list):
            meeg = MEEG(meeg_name, group.mw)
            epochs_dict, times = _get_load_cell_epochs(meeg, trig_plt_time, baseline_limit, apply_savgol)
            color = cmap(color_idx)
            for epd in epochs_dict['Down-First']:
                ax[idx].plot(times, epd, color=color, alpha=0.2)
                half_idx = int(len(epd) / 2) + 1
                ax[idx].plot(0, epd[half_idx], 'xr')

                ax[idx].set_title(group_name)
                ax[idx].set_ylabel('Weight')
                if idx == len(mw.pr.sel_groups) - 1:
                    ax[idx].set_xlabel('Time [s]')

    plt.subplots_adjust(hspace=0.2)
    fig.suptitle('Load-Cell Data')
    Group('all', mw).plot_save('lc_trigger_all', matplotlib_figure=fig)

    if show_plots:
        fig.show()


def plot_part_trigger(meeg, trig_channel, show_plots):
    raw = meeg.load_raw()

    (pd_data, rolling_diff1000, rolling_diff100, rolling_diffstd200, rolling_diffstd100,
     rd1000_peaks, rd100_peaks, rdstd200_peaks, rdstd100_peaks, std1000, stdstd100) = _get_load_cell_trigger(raw, trig_channel)

    tmin = 180000
    tmax = 220000

    plt.figure()
    plt.plot(pd_data[tmin:tmax], label='data')
    plt.plot(pd_data[tmin:tmax], 'ok', label='data_dots')
    plt.plot(rolling_diff1000[tmin:tmax], label='RollingDiff1000')
    plt.plot(rolling_diff100[tmin:tmax], label='RollingDiff100')
    plt.plot(rolling_diffstd200[tmin:tmax], label='RollingDiffStd200')
    plt.plot(rolling_diffstd100[tmin:tmax], label='RollingDiffStd100')
    plt.plot([p for p in rd1000_peaks if tmin < p < tmax],
             [rolling_diff1000[p] for p in rd1000_peaks if tmin < p < tmax], 'x', label='RollingDiff000-Peaks')
    plt.plot(range(tmin, tmax), np.full(tmax - tmin, std1000), label='RollingDiff2000-Std')
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
    if show_plots:
        plt.show()


def plot_load_cell_trigger_raw(meeg, min_duration, shortest_event, adjust_timeline_by_msec, trig_channel, show_plots):
    raw = meeg.load_raw()
    eeg_raw = raw.copy().pick(trig_channel)
    try:
        events = meeg.load_events()
    except FileNotFoundError:
        get_load_cell_events(meeg, min_duration, shortest_event, adjust_timeline_by_msec)
        events = meeg.load_events()
    events = events[np.nonzero(np.isin(events[:, 2], list(meeg.event_id.values())))]
    eeg_raw.plot(events, event_id=meeg.event_id, duration=90, scalings='auto', title=meeg.name, show=show_plots)


def plot_ica_trigger(meeg, trig_channel, show_plots):
    raw = meeg.load_raw()
    eeg_raw = raw.copy().pick(trig_channel)
    raw_filtered = meeg.load_filtered()
    raw_filtered = raw_filtered.copy().pick_types(meg=True, eeg=False, eog=False, stim=False, exclude=meeg.bad_channels)
    ica = meeg.load_ica()
    events = meeg.load_events()
    ica_sources = ica.get_sources(raw_filtered)
    try:
        ica_sources = ica_sources.add_channels([eeg_raw], force_update_info=True)
    except AssertionError:
        pass

    ica_sources.plot(events, n_channels=26, event_id=meeg.event_id, duration=30, scalings='auto', show=show_plots)


def plot_evokeds_half(meeg, show_plots):
    epochs = meeg.load_epochs()

    for trial in meeg.event_id:
        trial_epochs = epochs[trial]
        h1_evoked = trial_epochs[:int(len(epochs[trial]) / 2)].average()
        h2_evoked = trial_epochs[int(len(epochs[trial]) / 2):].average()

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        h1_axes = [axes[0, 0], axes[1, 0]]
        h1_evoked.plot(axes=h1_axes)
        axes[0, 0].set_title(f'{meeg.name}-{trial} First Half')

        h2_axes = [axes[0, 1], axes[1, 1]]
        h2_evoked.plot(axes=h2_axes)
        axes[0, 1].set_title(f'{meeg.name}-{trial} Second Half')

        meeg.plot_save('evokeds', subfolder='h1h2', trial=trial, matplotlib_figure=fig)
        if show_plots:
            fig.show()


def change_trig_channel_type(meeg, trig_channel):
    raw = meeg.load_raw()

    if trig_channel in raw.ch_names:
        print(f'Changing {trig_channel} to stim')
        raw.set_channel_types({trig_channel: 'stim'})

        if trig_channel in raw.ch_names:
            raw.rename_channels({trig_channel: 'LoadCell'})
            print(f'{meeg.name}: Rename Trigger-Channel')

        if trig_channel in meeg.bad_channels:
            meeg.bad_channels.remove(trig_channel)
            print(f'{meeg.name}: Removed Trigger-Channel from bad_channels')

        if trig_channel in raw.info['bads']:
            raw.info['bads'].remove(trig_channel)
            print(f'{meeg.name}: Removed Trigger-Channel from info["bads"]')

        meeg.save_raw(raw)


def get_dig_eegs(meeg, n_eeg_channels, eeg_dig_first=True):
    """
    Function to get EEG-Montage from digitized EEG-Electrodes
    (without them having be labeled as EEG during Digitization)

    Notes
    -----
    By Laura Doll, adapted by Martin Schulz
    """
    raw = meeg.load_raw()

    if 3 not in set([int(d['kind']) for d in raw.info['dig']]):
        ch_pos = dict()
        hsp = None
        all_extra_points = [dp for dp in raw.info['dig'] if int(dp['kind']) == 4]
        if eeg_dig_first:
            eeg_points = all_extra_points[:n_eeg_channels]
        else:
            eeg_points = all_extra_points[-n_eeg_channels:]
        for dp in eeg_points:
            ch_pos[f'EEG {dp["ident"]:03}'] = dp['r']

        hsp_points = [dp['r'] for dp in all_extra_points]

        if len(hsp_points) > 0:
            hsp = np.asarray(hsp_points)

        lpa = [dp['r'] for dp in raw.info['dig'] if int(dp['kind']) == 1 and dp['ident'] == 1][0]
        nasion = [dp['r'] for dp in raw.info['dig'] if int(dp['kind']) == 1 and dp['ident'] == 2][0]
        rpa = [dp['r'] for dp in raw.info['dig'] if int(dp['kind']) == 1 and dp['ident'] == 3][0]

        hpi = np.asarray([dp['r'] for dp in raw.info['dig'] if int(dp['kind']) == 2])

        montage = mne.channels.make_dig_montage(ch_pos, nasion, lpa, rpa, hsp, hpi)

        print(f'Added {n_eeg_channels} EEG-Channels to montage, '
              f'{len(all_extra_points) - n_eeg_channels} Head-Shape-Points remaining')

        raw.set_montage(montage, on_missing='raise')
    else:
        print('EEG channels already added here')

    meeg.save_raw(raw)


def plot_evokeds_pltest_overview(group):
    ltc_dict = dict()
    for file in group.group_list:
        meeg = MEEG(file, group.mw)
        ltcs = meeg.load_ltc()

        for trial in ltcs:
            if trial not in ltc_dict:
                ltc_dict[trial] = dict()
            for label in ltcs[trial]:
                if label not in ltc_dict[trial]:
                    ltc_dict[trial][label] = dict()
                ltc_dict[trial][label][file] = ltcs[trial][label]

    for trial in ltc_dict:
        for label in ltc_dict[trial]:
            fig = plt.figure()
            for file in ltc_dict[trial][label]:
                plt.plot(ltc_dict[trial][label][file][1], ltc_dict[trial][label][file][0], label=file)
            plt.title(f'{trial}-{label}')
            plt.legend()
            plt.xlabel('Time in s')
            plt.ylabel('Source amplitude')
            group.plot_save('pltest_ltc_overview', subfolder=label, trial=trial, matplotlib_figure=fig)


class ManualTriggerGui(QDialog):
    def __init__(self, mw):
        super().__init__(mw)
        self.mw = mw

        self.meeg = None
        self.events = None
        self.raw = None
        self.ev_idx = 0
        self.x_data = np.arange(-1000, 1000)
        self.line = None

        self.init_ui()
        self.open()
        self.showFullScreen()

    def init_ui(self):
        layout = QVBoxLayout()

        cmbx = QComboBox()
        cmbx.addItems(self.mw.pr.all_meeg)
        cmbx.currentTextChanged.connect(self.select_meeg)
        layout.addWidget(cmbx)

        self.idx_label = QLabel()
        self.idx_label.setText(str(self.ev_idx))
        self.idx_label.setFont(QFont('AnyStyle', 20))
        self.idx_label.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        layout.addWidget(self.idx_label, alignment=Qt.AlignHCenter)

        self.fig = Figure()
        # Add a subplot (1x1, index 1)
        self.axes = self.fig.add_subplot(1, 1, 1)
        self.canvas = FigureCanvasQTAgg(self.fig)
        layout.addWidget(self.canvas)

        bt_layout = QHBoxLayout()
        self.prev_bt = QPushButton('Previous')
        self.prev_bt.clicked.connect(self.previous)
        bt_layout.addWidget(self.prev_bt)

        self.minus_bt = QPushButton('-')
        self.minus_bt.clicked.connect(self.minus)
        bt_layout.addWidget(self.minus_bt)

        spinbx = QSpinBox()
        spinbx.setMinimum(1)
        spinbx.setMaximum(1000)
        spinbx.valueChanged.connect(self.set_steps)
        bt_layout.addWidget(spinbx)

        self.plus_bt = QPushButton('+')
        self.plus_bt.clicked.connect(self.plus)
        bt_layout.addWidget(self.plus_bt)

        self.next_bt = QPushButton('Next')
        self.next_bt.clicked.connect(self.next)
        bt_layout.addWidget(self.next_bt)
        layout.addLayout(bt_layout)

        cancel_bt = QPushButton('Cancel')
        cancel_bt.clicked.connect(self.cancel)
        layout.addWidget(cancel_bt)

        close_bt = QPushButton('Close')
        close_bt.clicked.connect(self.close)
        layout.addWidget(close_bt)

        self.prev_bt.setEnabled(False)
        self.minus_bt.setEnabled(False)
        self.plus_bt.setEnabled(False)
        self.next_bt.setEnabled(False)

        self.setLayout(layout)

    def select_meeg(self, name):
        # Save events of predecessor
        if self.meeg and self.events:
            self.meeg.save_events(self.events)
        else:
            self.prev_bt.setEnabled(True)
            self.minus_bt.setEnabled(True)
            self.plus_bt.setEnabled(True)
            self.next_bt.setEnabled(True)

        self.meeg = MEEG(name, self.mw)
        self.events = self.meeg.load_events()
        # Pick only events with id 5 (Down)
        self.events = self.events[np.nonzero(self.events[:, 2] == 5)]
        # Start at first event
        self.ev_idx = 0
        self.raw = self.meeg.load_raw()
        self.trig_ch = self.meeg.pa['trig_channel']

        self.update_line()
        self.idx_label.setText(str(self.ev_idx))

    def set_steps(self, steps):
        self.steps = steps

    def update_line(self):
        start = self.events[self.ev_idx, 0] - 1000 - self.raw.first_samp
        stop = self.events[self.ev_idx, 0] + 1000 - self.raw.first_samp
        y_data, = self.raw.get_data(picks=self.trig_ch, start=start, stop=stop)

        if self.line is None:
            self.line, = self.axes.plot(self.x_data, y_data, 'b')
            self.axes.axvline(x=0, color='r')
        else:
            self.line.set_ydata(y_data)

        self.canvas.draw()

    def previous(self):
        self.ev_idx -= 1
        if self.ev_idx < 0:
            self.ev_idx = 0
        self.update_line()
        self.idx_label.setText(str(self.ev_idx))

    def next(self):
        self.ev_idx += 1
        if self.ev_idx > len(self.events) - 1:
            self.ev_idx = len(self.events) - 1
        self.update_line()
        self.idx_label.setText(str(self.ev_idx))

    def minus(self):
        self.events[self.ev_idx, 0] -= self.steps
        self.update_line()

    def plus(self):
        self.events[self.ev_idx, 0] += self.steps
        self.update_line()

    def cancel(self):
        self.events = None
        self.close()

    def closeEvent(self, event):
        if self.events is not None:
            self.meeg.save_events(self.events)
        event.accept()


def manual_trigger_gui(mw):
    ManualTriggerGui(mw)


def rereference_eog(meeg, eog_tuple, eogecg_target):
    if eogecg_target == 'Raw (Unfiltered)':
        raw = meeg.load_raw()
    else:
        raw = meeg.load_filtered()

    # Remove old channels
    for old_ch_name in [f'EOG BP{idx}' for idx in range(10)]:
        if old_ch_name in raw.ch_names:
            raw = raw.drop_channels(old_ch_name)
            print(f'Dropped existing channel: {old_ch_name}')

    # Set Bipolar reference
    ch_name = f'EOG BP'
    if ch_name in raw.ch_names:
        raw = raw.drop_channels(ch_name)
        print(f'Dropped existing channel: {ch_name}')

    mne.set_bipolar_reference(raw, eog_tuple[0], eog_tuple[1], ch_name=ch_name,
                              drop_refs=False, copy=False)
    raw.set_channel_types({ch_name: 'eog'})

    if eogecg_target == 'Raw (Unfiltered)':
        meeg.save_raw(raw)
    else:
        meeg.save_filtered(raw)


def adjust_scales_lc_tests(meeg):
    raw = meeg.load_raw()
    raw_data = raw.get_data()
    # Enhance Direction-Channel
    raw_data[2] *= 1000

    # Diminish Channels to look like EEG
    raw_data /= 10 ** 6

    new_raw = RawArray(raw_data, raw.info)
    meeg.save_raw(new_raw)


def get_velo_trigger(meeg):
    events = meeg.load_events()
    for row_idx, value in enumerate(events[:, 2]):
        if value == 1 or value == 2:
            # Copy the time of the next (first-touch) Trigger-Value, + 1 to avoid duplicates
            n = 1
            while True:
                if row_idx != len(events[:, 0]) - 1:
                    next_id = events[row_idx + n, 2]
                    if next_id == 4:
                        events[row_idx, 0] = events[row_idx + n, 0] + 1
                        break
                    elif row_idx + n < len(events[:, 0]) - 1:
                        n += 1
                    else:
                        break
                else:
                    break

    # sort events by time (first column)
    events = events[events[:, 0].argsort()]

    meeg.save_events(events)


def get_wave_file(meeg, wav_input_type, ch_names, samplerate):
    import scipy
    from scipy.io.wavfile import write
    if not isinstance(ch_names, list):
        ch_names = [ch_names]
    if wav_input_type == 'Raw':
        data = meeg.load_raw()
    else:
        data = meeg.load_evokeds()[0]
    for ch_name in ch_names:
        datap = data.copy().pick_channels([ch_name])
        if wav_input_type == 'Raw':
            dp = datap.get_data()[0]
        else:
            dp = datap.data[0]
        dp /= max(abs(dp)) * 0.99
        dp = scipy.signal.resample(dp, len(dp) * int(samplerate / data.info['sfreq']))
        print('Writing Wav-File...')
        write(join(meeg.save_dir, f'{meeg.name}_{wav_input_type}_{ch_name}.wav'), samplerate, dp)


def make_fixed_length_events(meeg, fixed_id, fixed_duration):
    raw = meeg.load_raw()
    events = mne.make_fixed_length_events(raw, id=fixed_id, duration=fixed_duration)
    meeg.save_events(events)


def get_ecg_channel(meeg, ecg_channel, eogecg_target):
    if eogecg_target == 'Raw (Unfiltered)':
        raw = meeg.load_raw()
    else:
        raw = meeg.load_filtered()

    ecg_raw = raw.copy().pick(ecg_channel)
    ecg_raw.set_channel_types({ecg_channel: 'ecg'})
    ecg_raw.rename_channels({ecg_channel: 'ECG'})
    raw.add_channels([ecg_raw])

    if eogecg_target == 'Raw (Unfiltered)':
        meeg.save_raw(raw)
    else:
        meeg.save_filtered(raw)
