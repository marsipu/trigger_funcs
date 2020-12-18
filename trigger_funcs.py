import math

import matplotlib.pyplot as plt
import mne
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QComboBox, QDialog, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QSpinBox, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from scipy.signal import find_peaks

from mne_pipeline_hd.pipeline_functions.loading import MEEG
from mne_pipeline_hd.basic_functions.operations import find_6ch_binary_events
from mne_pipeline_hd.gui.loading_widgets import extract_info
from mne_pipeline_hd.pipeline_functions.decorators import small_func


def _get_trig_ch(raw):
    if raw.info['nchan'] > 160:
        trig_ch = 'EEG 064'
    elif 130 < raw.info['nchan'] < 160:
        trig_ch = 'EEG 029'
    elif raw.info['nchan'] == 1:
        trig_ch = '29/Weight'
    elif raw.info['nchan'] == 3:
        trig_ch = 'LoadCellTrigger'
    else:
        trig_ch = 'EEG 001'

    return trig_ch


def _get_load_cell_trigger(raw):
    trig_ch = _get_trig_ch(raw)
    eeg_raw = raw.copy().pick(trig_ch)
    # eeg_raw = eeg_raw.filter(0, 20, n_jobs=-1)
    eeg_series = eeg_raw.to_data_frame()[trig_ch]

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


def _get_load_cell_trigger_model(meeg, min_duration, shortest_event, adjust_timeline_by_msec):
    raw = meeg.load_raw()

    trig_ch = _get_trig_ch(raw)
    eeg_raw = raw.copy().pick(trig_ch)

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


def get_load_cell_events(meeg, min_duration, shortest_event, adjust_timeline_by_msec):
    raw = meeg.load_raw()

    (eeg_series, rolling_diff1000, rolling_diff100, rolling_diffstd200, rolling_diffstd100,
     rd1000_peaks, rd100_peaks, rdstd200_peaks, rdstd100_peaks, std2000, stdstd100) = _get_load_cell_trigger(raw)

    find_6ch_binary_events(meeg, min_duration, shortest_event, adjust_timeline_by_msec)
    events = meeg.load_events()

    for pk in rd1000_peaks:
        sp = np.asarray(eeg_series[pk - 500:pk + 500])
        rd100 = np.asarray(rolling_diff100[pk - 500:pk + 500])
        # Correct Offset
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


def plot_load_cell_epochs(meeg):
    raw = meeg.load_raw()
    trig_ch = _get_trig_ch(raw)
    eeg_raw = raw.copy().pick(trig_ch)
    events = meeg.load_events()

    event_id = {'Down': 5}
    eeg_epochs = mne.Epochs(eeg_raw, events, event_id=event_id, tmin=-1, tmax=1, baseline=None)
    # eeg_epochs.plot(title=meeg.name, event_id=event_id)

    data = eeg_epochs.get_data()
    fig, ax = plt.subplots(1, 1)
    for ep in data:
        ax.plot(range(-1000, 1001), ep[0])
        ax.plot(0, ep[0][1001], 'x')

    fig.suptitle(meeg.name)
    meeg.plot_save('trigger_epochs', matplotlib_figure=fig)
    fig.show()


def plot_part_trigger(meeg):
    raw = meeg.load_raw()

    # raw = raw.filter(0, 20, n_jobs=-1)
    trig_ch = _get_trig_ch(raw)

    (pd_data, rolling_diff1000, rolling_diff100, rolling_diffstd200, rolling_diffstd100,
     rd1000_peaks, rd100_peaks, rdstd200_peaks, rdstd100_peaks, std2000, stdstd100) = _get_load_cell_trigger(raw)

    tmin = 180000
    tmax = 220000

    plt.figure()
    plt.plot(pd_data[tmin:tmax], label='data')
    plt.plot(pd_data[tmin:tmax], 'ok', label='data_dots')
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


def plot_load_cell_trigger_raw(meeg, min_duration, shortest_event, adjust_timeline_by_msec):
    raw = meeg.load_raw()
    trig_ch = _get_trig_ch(raw)
    eeg_raw = raw.copy().pick(trig_ch)
    try:
        events = meeg.load_events()
    except FileNotFoundError:
        get_load_cell_events(meeg, min_duration, shortest_event, adjust_timeline_by_msec)
        events = meeg.load_events()
    events = events[np.nonzero(np.isin(events[:, 2], list(meeg.event_id.values())))]
    eeg_raw.plot(events, event_id=meeg.event_id, duration=90, scalings='auto', title=meeg.name)


def plot_ica_trigger(meeg):
    raw = meeg.load_raw()
    trig_ch = _get_trig_ch(raw)
    eeg_raw = raw.copy().pick(trig_ch)
    raw_filtered = meeg.load_filtered()
    raw_filtered = raw_filtered.copy().pick_types(meg=True, eeg=False, eog=False, stim=False, exclude=meeg.bad_channels)
    ica = meeg.load_ica()
    events = meeg.load_events()
    ica_sources = ica.get_sources(raw_filtered)
    try:
        ica_sources = ica_sources.add_channels([eeg_raw], force_update_info=True)
    except AssertionError:
        pass

    ica_sources.plot(events, n_channels=26, event_id=meeg.event_id, duration=30, scalings='auto')


def reload_info_dict(meeg):
    meeg.extract_info()


def plot_evokeds_half(meeg):
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
        fig.show()


def get_dig_eegs(meeg, n_eeg_channels, eeg_dig_first=True):
    """
    Function to get EEG-Montage from digitized EEG-Electrodes
    (without them having be labeled as EEG during Digitization)

    Notes
    -----
    By Laura Doll, adapted by Martin Schulz
    """
    raw = meeg.load_raw(pick_types=False)

    ch_pos = dict()
    hsp = None
    hsp_points = list()
    if 3 not in set([int(d['kind']) for d in raw.info['dig']]):
        extra_points = [dp for dp in raw.info['dig'] if int(dp['kind']) == 4]

        if eeg_dig_first:
            for dp in extra_points[:n_eeg_channels]:
                ch_pos[f'EEG {dp["ident"]:03}'] = dp['r']
                hsp_points = [dp['r'] for dp in extra_points[n_eeg_channels:]]
        else:
            for dp in extra_points[-n_eeg_channels:]:
                ch_pos[f'EEG {dp["ident"]:03}'] = dp['r']
                hsp_points = [dp['r'] for dp in extra_points[:-n_eeg_channels]]

        if len(hsp_points) > 0:
            hsp = np.asarray(hsp_points)

        lpa = [dp['r'] for dp in raw.info['dig'] if int(dp['kind']) == 1 and dp['ident'] == 1][0]
        nasion = [dp['r'] for dp in raw.info['dig'] if int(dp['kind']) == 1 and dp['ident'] == 2][0]
        rpa = [dp['r'] for dp in raw.info['dig'] if int(dp['kind']) == 1 and dp['ident'] == 3][0]

        hpi = np.asarray([dp['r'] for dp in raw.info['dig'] if int(dp['kind']) == 2])

        montage = mne.channels.make_dig_montage(ch_pos, nasion, lpa, rpa, hsp, hpi)

        print(f'Added {n_eeg_channels} EEG-Channels to montage, '
              f'{len(extra_points) - n_eeg_channels} Head-Shape-Points remaining')

        raw.set_montage(montage, on_missing='raise')
    else:
        print('EEG channels already added here')

    meeg.extract_info()
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
        self.trig_ch = _get_trig_ch(self.raw)

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


def rereference_eog(meeg, eog_tuple):
    raw = meeg.load_raw(pick_types=False)

    for idx in range(int(len(eog_tuple) / 2)):
        # Set Bipolar reference
        ch_name = f'EOG BP{idx}'
        if ch_name not in raw.ch_names:
            mne.set_bipolar_reference(raw, eog_tuple[idx], eog_tuple[idx + 1], ch_name=ch_name,
                                      drop_refs=False, copy=False)
            raw.set_channel_types({ch_name: 'eog'})

    meeg.extract_info()
    meeg.save_raw(raw)
