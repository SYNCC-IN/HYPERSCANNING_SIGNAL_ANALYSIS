import os

import numpy as np
import xmltodict
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.signal import filtfilt, butter, decimate, sosfiltfilt, iirnotch
import neurokit2 as nk
import joblib



class DataLoader:

    def __init__(self, ID, folder_EEG = None, folder_H10 = None, folder_ET = None, output_dir = None, plot_flag = False):
        self.ID = ID
        self.output_dir = output_dir
        self.folder_EEG = folder_EEG
        self.folder_H10 = folder_H10
        self.folder_ET = folder_ET
        self.plot_flag = plot_flag
        self.data = self._load_warsaw_pilot_data(folder = self.folder_EEG,file=self.ID)
        self.output = self._filter_warsaw_pilot_data(self.data)
        self._save_to_file()
    def _filter_warsaw_pilot_data(self,data, lowcut=4.0, highcut=40.0, q=8):

        '''Filter the Warsaw pilot data using a low, high pas and notch filter.
            And apply montage to M1 and M2 channels for EEG data.
            EEG data is decimated by q times to reduce the sampling frequency.
            The ECG data is filtered using a high pass filter and notch filter. And the ECG channels are mounted L-R
        Args:
            data (dictionry): {
                                'data': data,
                                'Fs_EEG': Fs_EEG,
                                'ChanNames': ChanNames,
                                'channels': channels
                                }
            lowcut (float): The low cutoff frequency for the bandpass filter.
            highcut (float): The high cutoff frequency for the bandpass filter.
            q: decimation of the EEG data
        Returns:
            filtered_data (dict): Filtered data with the structure:
            {
                "ID": str,                        # Dyad ID
                "data_EEG": np.ndarray,           # EEG data [n_channels x n_samples]
                "Fs_EEG": float,                  # EEG sampling rate (Hz)
                "times_EEG": np.ndarray,          # time vector (s) [1 x n_samples]
                "chanNames_EEG": list[str],       # list of channel names in order
                "channels_EEG": dict[str, int],   # mapping: channel name → index in 'data'

                "references": str,                # Information about reference electrodes or common average

                "filtration": {                   # Information about filtering
                    "notch": bool,                # If True, notch filter applied
                    "low_pass": float,            # Low-pass filter cutoff frequency (Hz)
                    "high_pass": float,           # High-pass filter cutoff frequency (Hz)
                    "type": str                   # Type of filter (e.g., 'FIR', 'IIR')
                }

                'EEG_channels_ch': list[str],     # child EEG channels after montage
                'EEG_channels_cg': list[str],     # caregiver EEG channels after montage

                'ECG_ch': np.ndarray,             # filtered ECG (child)
                'ECG_cg': np.ndarray,             # filtered ECG (caregiver)
                'Fs_ECG': int,                    # ECG sampling frequency
                't_ECG': np.ndarray,              # time vector for ECG

                'IBI_ch_interp': np.ndarray,      # interpolated IBI (child)
                'IBI_cg_interp': np.ndarray,      # interpolated IBI (caregiver)
                'Fs_IBI': int,                    # IBI sampling frequency (default: 4 Hz)
                't_IBI': np.ndarray               # time vector for interpolated IBI

                'ET_ch: np.ndarray,               # ET (child)
                'ET_cg': np.ndarray,              # ET (caregiver)
                'Fs_ET': int,                     # ET sampling frequency
                't_ET': np.ndarray                # time vector for interpolated IBI

                "event": list,                    # list of event markers (stimuli, triggers, etc.)
                "epoch": list or None,            #

                "paths": {
                    "directory_EEG": str,         # path to EEG data raw
                    "directory_ET": str,          # path to eye-tracking files
                    "directory_HRV": str,         # path to HRV -> IBI files
                    "output_dir": str,            # path where to save results/figures
                },

               "tasks": {
                    "dual_HRV": {
                        "SECORE": bool,           # True if active HRV during SECORE was recorded
                        "movies": bool,           # True if passive HRV recorded
                        "conversation": bool      # True if active HRV recorded
                    },
                    "dual_EEG": {
                        "movies": bool,           # True if passive EEG recorded
                        "conversation": bool,     # True if active EEG recorded
                    }
                    "dual_ET": {
                        "movies": bool,           # True if passive ET recorded
                        "conversation": bool      # True if active ET recorded
                    }
                }

               "child_info": {                    # Information about child
                    "birth_date": datetime.date,  # Child birth date
                    "age_years": int,             # Child age in months at the time of recording
                    "age_months": int,            # Child age in months at the time of recording
                    "age_days": int,              # Additional days beyond months
                    "rec_date: datetime.date,     # Date when recording was done
                    "group": str,                 # Child group: 'T' (Typical),  'ASD' (Autism Spectrum Disorder), 'P' (Premature)
                    "sex": str                    # Child sex: 'M' (male), 'F' (female)
                }

                "notes": str or None,    # notes from experiment
            }
        '''
        signal = data['data'].copy()
        signal *= 0.0715  # scale the signal to microvolts
        Fs_EEG = data['Fs_EEG']
        Fs_ECG = Fs_EEG  # ECG data is sampled at the same frequency as EEG data
        Fs_IBI = 4  # sampling frequency [Hz] for the IBI signals
        t_ECG = np.arange(0, signal.shape[1] / Fs_EEG, 1 / Fs_EEG)  # time vector for the ECG data

        channels = data['channels']
        b_notch, a_notch = iirnotch(50, 30, fs=Fs_EEG)

        # extract and filter the ECG data
        ECG_ch = data['data'][channels['EKG1'], :] - data['data'][channels['EKG2'], :]
        ECG_cg = data['data'][channels['EKG1_cg'], :] - data['data'][channels['EKG2_cg'], :]
        sos_ecg = butter(5, 0.5, btype='high', output="sos", fs=Fs_EEG)
        ECG_ch_filtered = sosfiltfilt(sos_ecg, ECG_ch)
        ECG_ch_filtered = filtfilt(b_notch, a_notch, ECG_ch_filtered)
        ECG_cg_filtered = sosfiltfilt(sos_ecg, ECG_cg)
        ECG_cg_filtered = filtfilt(b_notch, a_notch, ECG_cg_filtered)
        # interpolate IBI signals from ECG data
        IBI_ch_interp, t_IBI_ch = self._interpolate_IBI_signals(ECG_ch_filtered, Fs_ECG, Fs_IBI=Fs_IBI, label='')
        IBI_cg_interp, t_IBI_cg = self._interpolate_IBI_signals(ECG_cg_filtered, Fs_ECG, Fs_IBI=Fs_IBI, label='')
        # check if the IBI signals are of the same length
        if len(IBI_ch_interp) != len(IBI_cg_interp):
            min_length = min(len(IBI_ch_interp), len(IBI_cg_interp))
            IBI_ch_interp = IBI_ch_interp[:min_length]
            IBI_cg_interp = IBI_cg_interp[:min_length]
            t_IBI_ch = t_IBI_ch[:min_length]
            t_IBI_cg = t_IBI_cg[:min_length]

        t_IBI = t_IBI_ch  # use the time vector for the child IBI as it is the same length as the caregiver IBI
        # define EEG channels for child and caregiver
        EEG_channels_ch = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'M1', 'T3', 'C3', 'Cz', 'C4', 'T4', 'M2', 'T5',
                           'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
        EEG_channels_cg = ['Fp1_cg', 'Fp2_cg', 'F7_cg', 'F3_cg', 'Fz_cg', 'F4_cg', 'F8_cg', 'M1_cg', 'T3_cg', 'C3_cg',
                           'Cz_cg', 'C4_cg', 'T4_cg', 'M2_cg', 'T5_cg', 'P3_cg', 'Pz_cg', 'P4_cg', 'T6_cg', 'O1_cg',
                           'O2_cg']

        # design EEG filters
        b_low, a_low = butter(4, highcut, btype='low', fs=Fs_EEG)
        b_high, a_high = butter(4, lowcut, btype='high', fs=Fs_EEG)

        # filter the caregiver EEG data
        for i, ch in enumerate(EEG_channels_cg):
            if ch in data['channels']:
                idx = data['channels'][ch]
                signal[idx, :] = filtfilt(b_notch, a_notch, signal[idx, :], axis=0)
                signal[idx, :] = filtfilt(b_low, a_low, signal[idx, :], axis=0)
                signal[idx, :] = filtfilt(b_high, a_high, signal[idx, :], axis=0)
        # apply monage to the M1 M2 channels for caregiver EEG channels
        for i, ch in enumerate(EEG_channels_cg):
            if ch in data['channels']:
                idx = data['channels'][ch]
                signal[idx, :] = signal[idx, :] - 0.5 * (
                            signal[data['channels']['M1_cg'], :] + signal[data['channels']['M2_cg'], :])
                # remove channels M1 and M2 from the caregiver EEG channels, as they will not be used after linked ears montage
        EEG_channels_cg = ['Fp1_cg', 'Fp2_cg', 'F7_cg', 'F3_cg', 'Fz_cg', 'F4_cg', 'F8_cg', 'T3_cg', 'C3_cg', 'Cz_cg',
                           'C4_cg', 'T4_cg', 'T5_cg', 'P3_cg', 'Pz_cg', 'P4_cg', 'T6_cg', 'O1_cg', 'O2_cg']

        # filter the child EEG data
        for i, ch in enumerate(EEG_channels_ch):
            if ch in data['channels']:
                idx = data['channels'][ch]
                signal[idx, :] = filtfilt(b_notch, a_notch, signal[idx, :], axis=0)
                signal[idx, :] = filtfilt(b_low, a_low, signal[idx, :], axis=0)
                signal[idx, :] = filtfilt(b_high, a_high, signal[idx, :], axis=0)
                # apply monage to the M1 M2 channels for child EEG channels
        for i, ch in enumerate(EEG_channels_ch):
            if ch in data['channels']:
                idx = data['channels'][ch]
                signal[idx, :] = signal[idx, :] - 0.5 * (
                            signal[data['channels']['M1'], :] + signal[data['channels']['M2'], :])
                # remove channels M1 and M2 from the child EEG channels, as thye will not be used after linked ears montage
        EEG_channels_ch = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz',
                           'P4', 'T6', 'O1', 'O2']

        # decimate the data to reduce the sampling frequency q times
        signal_out = decimate(signal, q, axis=-1)
        Fs_EEG_q = Fs_EEG // q  # new sampling frequency for the EEG data after decimation
        # time vector for the EEG data after decimation
        t_EEG = np.arange(0, signal_out.shape[1] / Fs_EEG_q, 1 / Fs_EEG_q)  #

        filtered_data = {
            'id': self.ID,
            'data_EEG': signal_out,
            'times_EEG': t_EEG,  # time vector for the EEG data after decimation
            'Fs_EEG': Fs_EEG_q,  # signal is decimated to this frequency
            'ChanNames_EEG': data['ChanNames'],  # list of channel names
            'channels_EEG': data['channels'],  # channels dictionary with indexes referenig to 'data' array
            'EEG_channels_ch': EEG_channels_ch,  # list of names of child EEG channels
            'EEG_channels_cg': EEG_channels_cg,  # list of names of caregiver EEG channels
            'ECG_ch': ECG_ch_filtered,
            'ECG_cg': ECG_cg_filtered,
            'Fs_ECG': Fs_ECG,  # This is the original sampling frequency of the ECG data
            't_ECG': t_ECG,  # time vector for the ECG data
            'IBI_ch_interp': IBI_ch_interp,  # Interpolated IBI signal for the child
            'IBI_cg_interp': IBI_cg_interp,  # Interpolated IBI signal for the caregiver
            'Fs_IBI': Fs_IBI,  # Sampling frequency for the IBI signals
            't_IBI': t_IBI,  # Time vector for the interpolated IBI signal
            'ET_ch': None, # np.ndarray - will be added later
            'ET_cg': None, # np.ndarray - will be added later
            'Fs_ET': None, # int - will be added later
            't_ET' : None, # np.ndarray - will be added later
            'event': self._scan_for_events(self.data), # implemented as dictionary for more convenient
            'epoch': None,
            'paths': {
                'directory_EEG': self.folder_EEG,
                'directory_ET': self.folder_ET,
                'directory_HRV': self.folder_H10,
                'output_dir': self.output_dir
            },
            # tasks will be added later
            'tasks': {
                'dual_HRV': {
                    'SECORE': None, # bool
                    'movies': None, # bool
                    'conversation': None, # bool
                },
                'dual_EEG': {
                    'movies': None, # bool
                    'conversation': None, # bool
                },
                'dual_ET': {
                    'movies': None, #bool
                    'conversation': None, #bool
                }

            },
            # child info will be added later
            'child_info': {
                'birth_date': None, #datetime.date
                'age_years': None, #int
                'age_months': None, #int
                'age_days': None, #int
                'rec_date': None, #datetime.date
                'group': None, #str
                'sex': None #str
            },
            'notes': None # str or none
        }
        return filtered_data
    @staticmethod
    def load_output_data(filename):
        results = joblib.load(filename)
        for key, val in results.items():
            if isinstance(val, np.ndarray):
                print(f"{key}: {val.shape}")
            else:
                print(f"{key}: {val}")
        return results
    def _save_to_file(self):
        joblib.dump(self.output,self.output_dir+f"{self.ID}.joblib")
    # method for load Warsaw_Data_Frame.csv
    def _load_csv_data(self, csv_file):
        pass
    def _load_warsaw_pilot_data(self,folder, file):
        file = file+".obci"
        with open(os.path.join(folder, f"{file}.xml")) as fd:
            xml = xmltodict.parse(fd.read())

        N_ch = int(xml['rs:rawSignal']['rs:channelCount'])
        Fs_EEG = int(float(xml['rs:rawSignal']['rs:samplingFrequency']))
        ChanNames = xml['rs:rawSignal']['rs:channelLabels']['rs:label']
        channels = {}
        for i, name in enumerate(ChanNames):
            channels[name] = i
        data = np.fromfile(os.path.join(folder, f"{file}.raw"), dtype='float32').reshape((-1, N_ch))
        data = data.T

        if self.plot_flag:
            ECG_CH = data[ChanNames.index('EKG1'), :] - data[ChanNames.index('EKG2'), :]
            ECG_CG = data[ChanNames.index('EKG1_cg'), :] - data[ChanNames.index('EKG2_cg'), :]
            fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
            ax[0].plot(ECG_CH, label='Child ECG')
            ax[1].plot(ECG_CG, label='Caregiver ECG')
            plt.legend()
            plt.show()
        output = {
            'data': data,
            'Fs_EEG': Fs_EEG,
            'ChanNames': ChanNames,
            'channels': channels
        }
        return output

    def _scan_for_events(self, data, threshold=20000):
        '''Scan for events in the diode signal and plot them if required.
        Args:
            diode (np.ndarray): Diode signal.
            Fs_EEG (int): Sampling frequency of the EEG data.
            plot (bool): Whether to plot the diode signal and detected events.
        Returns:
            events (dict): Dictionary containing the start and end time of detected events measured in seconds from the start of the recording. The expected events are:
                - Movie_1
                - Movie_2
                - Movie_3
                - Talk_1
                - Talk_2'''
        events = {'Talk_1': None, 'Talk_2': None, 'Movie_1': None, 'Movie_2': None, 'Movie_3': None}
        diode_idx = data['channels']['Diode']
        diode = data['data'][diode_idx, :]
        Fs_EEG = data['Fs_EEG']
        x = np.zeros(diode.shape)
        d = diode.copy()
        d /= threshold
        x[d > 1] = 1
        if self.plot_flag:
            plt.figure(figsize=(12, 6))
            plt.plot(d, 'b', label='Diode Signal normalized by threshold')
            plt.plot(x, 'r', label='Diode Signal Thresholded')
            plt.title('Diode Signal with events')
            plt.xlabel('Samples')
            plt.ylabel('Signal Value')
            plt.legend()

        y = np.diff(x)
        up = np.zeros(y.shape, dtype=int)
        down = np.zeros(y.shape, dtype=int)
        up[y == 1] = 1
        down[y == -1] = 1
        if self.plot_flag:
            plt.plot(up, 'g', label='Up Events')
            plt.plot(down, 'm', label='Down Events')
            plt.legend()

        dt = 17  # ms between frames
        i = 0
        while i < len(down):
            if down[i] == 1:
                s1 = int(np.sum(up[i + int(0.5 * Fs_EEG) - 2 * dt: i + int(0.5 * Fs_EEG) + 2 * dt]))
                s2 = int(np.sum(up[i + int(1.0 * Fs_EEG) - 3 * dt: i + int(1.0 * Fs_EEG) + 3 * dt]))
                s3 = int(np.sum(up[i + int(1.5 * Fs_EEG) - 4 * dt: i + int(1.5 * Fs_EEG) + 4 * dt]))
                s4 = int(np.sum(up[i + int(2.0 * Fs_EEG) - 5 * dt: i + int(2.0 * Fs_EEG) + 5 * dt]))
                s5 = int(np.sum(up[i + int(2.5 * Fs_EEG) - 6 * dt: i + int(2.5 * Fs_EEG) + 6 * dt]))
                # plt.plot(x, 'b'), plt.plot(i,x[i],'bo')
                if s1 == 1 and s2 == 0 and s3 == 0 and s4 == 0 and s5 == 0:
                    print(f"Movie 1 starts at {i / Fs_EEG:.2f} seconds")
                    events['Movie_1'] = i / Fs_EEG
                    if self.plot_flag:
                        plt.plot(x, 'b'), plt.plot(i, x[i], 'ro')
                    i += int(2.5 * Fs_EEG)
                elif s1 == 1 and s2 == 0 and s3 == 1 and s4 == 0 and s5 == 0:
                    print(f"Movie 2 starts at {i / Fs_EEG:.2f} seconds")
                    events['Movie_2'] = i / Fs_EEG
                    if self.plot_flag:
                        plt.plot(x, 'b'), plt.plot(i, x[i], 'go')
                    i += int(2.5 * Fs_EEG)
                elif s1 == 1 and s2 == 0 and s3 == 1 and s4 == 0 and s5 == 1:
                    print(f"Movie 3 starts at {i / Fs_EEG:.2f} seconds")
                    events['Movie_3'] = i / Fs_EEG
                    if self.plot_flag:
                        plt.plot(x, 'b'), plt.plot(i, x[i], 'yo')
                    i += int(2.5 * Fs_EEG)
                elif s1 == 0 and s2 == 1 and s3 == 0 and s4 == 0 and s5 == 0:
                    if events['Talk_1'] is None:
                        print(f"Talk 1 starts at {i / Fs_EEG:.2f} seconds")
                        events['Talk_1'] = i / Fs_EEG
                        if self.plot_flag:
                            plt.plot(x, 'b'), plt.plot(i, x[i], 'co')
                        i += int(2.5 * Fs_EEG)
                    else:
                        print(f"Talk 2 starts at {i / Fs_EEG:.2f} seconds")
                        events['Talk_2'] = i / Fs_EEG
                        if self.plot_flag:
                            plt.plot(x, 'b'), plt.plot(i, x[i], 'mo')
                            plt.show()
                        i = len(down)  # talk 2 is the last event so finish scaning for events
            i += 1
        return events

    def _interpolate_IBI_signals(self,ECG, Fs_ECG, Fs_IBI=4, label=''):
        # Extract R-peaks location
        _, info_ECG = nk.ecg_process(ECG, sampling_rate=Fs_ECG, method='neurokit')
        rpeaks = info_ECG["ECG_R_Peaks"]

        IBI = np.diff(rpeaks) / Fs_ECG * 1000  # IBI in ms
        t = np.cumsum(IBI) / 1000  # time vector for the IBI signals [s]
        t_ECG = np.arange(0, t[-1], 1 / Fs_IBI)  # time vector for the interpolated IBI signals
        cs = CubicSpline(t, IBI)
        IBI_interp = cs(t_ECG)
        if self.plot_flag:
            plt.figure(figsize=(12, 6))
            plt.plot(t_ECG, IBI_interp)
            plt.xlabel('time [s]')
            plt.ylabel('IBI [ms]')
            plt.title(f'Interpolated IBI signal of {label} as a function of time')
            plt.show()
        return IBI_interp, t_ECG