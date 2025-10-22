import os

import numpy as np
import xmltodict
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.signal import filtfilt, butter, decimate, sosfiltfilt, iirnotch
import neurokit2 as nk
import joblib



class DataLoader:

    def __init__(self, ID):
        '''DataLoader class for loading and processing Warsaw pilot data.
        The constructor initializes the DataLoader with the given ID and an empty list of possible modalities.
        The data related to the modalities will be added by setter methods.
        Retrieving the data will be done by getter methods.
            '''
        
        self.ID = ID
        self.modalities = [] # list of modalities already loaded: 'EEG', 'H10', 'ET'
        self.folder ={}
        self.FS ={}
        self.data = {}



        
        #self.data = self._load_warsaw_pilot_data(folder = self.folder_EEG, file=self.ID)
        #self.output = self._filter_warsaw_pilot_data(self.data)
        #self._save_to_file()

    def _set_EEG_data(self, folder_EEG, debug_flag=False):
        '''Set the EEG data for the DataLoader instance by loading and filtering the Warsaw pilot data.
        We assume data were recorded as multiplexed signals in SVAROG system format.
        We also assume specific channel names for child and caregiver EEG data, as specified below.
        Args:
            folder_EEG (str): Path to the folder containing the EEG data files.
            debug_flag (bool): Whether to plot intermediate results for debugging/visualization.
        '''
        self.folder['EEG'] = folder_EEG
        # define EEG channels for child and caregiver
        self.channel_names['EEG_ch'] = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'M1', 'T3', 'C3', 'Cz', 'C4', 'T4', 'M2', 'T5',
                           'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
        self.channel_names['EEG_cg'] = ['Fp1_cg', 'Fp2_cg', 'F7_cg', 'F3_cg', 'Fz_cg', 'F4_cg', 'F8_cg', 'M1_cg', 'T3_cg', 'C3_cg',
                           'Cz_cg', 'C4_cg', 'T4_cg', 'M2_cg', 'T5_cg', 'P3_cg', 'Pz_cg', 'P4_cg', 'T6_cg', 'O1_cg',
                           'O2_cg']
        self.channel_names['EEG'] = self.channel_names['EEG_ch'] + self.channel_names['EEG_cg']
        self._read_raw_SVAROG_data(debug_flag)  # load the raw data, works inplace, no return value
        return

    def _read_raw_SVAROG_data(self, lowcut=4.0, highcut=40.0, q=8, debug_flag=False):
        file = self.ID + ".obci" # SVAROG files have .obci extension
        # read meta informations from xml file
        with open(os.path.join(self.folder['EEG'], f"{file}.xml")) as fd: 
            xml = xmltodict.parse(fd.read())

        N_ch = int(xml['rs:rawSignal']['rs:channelCount'])
        Fs_EEG = int(float(xml['rs:rawSignal']['rs:samplingFrequency']))
        ChanNames = xml['rs:rawSignal']['rs:channelLabels']['rs:label']
        # create a dictionary which maps channel names and their indexes
        channels = {}
        for i, name in enumerate(ChanNames):
            channels[name] = i
        self.channels['EEG'] = channels

        # if debug print N_chan, Fs_EEG, ChanNames
        if debug_flag:
            print(f"N_chan: {N_ch},\n Fs_EEG: {Fs_EEG},\n ChanNames: {ChanNames}")

        self.Fs['EEG'] = Fs_EEG
        self.Fs['ECG'] = Fs_EEG  # ECG data is sampled at the same frequency as EEG data
        # read raw data from .raw file
        data = np.fromfile(os.path.join(self.folder['EEG'], f"{file}.raw"), dtype='float32').reshape((-1, N_ch))
        data = data.T # transpose to have channels in rows and samples in columns
        
        # extract diode signal for event detection before any scaling and filtering
        self.diode = data[channels['diode'], :]
        # scan for events
        self.events = self._scan_for_events(threshold=20000)
        # scale the signal to microvolts    
        data *= 0.0715  

 
        # mount EEG data to M1 and M2 channels and filter the data
        data = self._mount_EEG_data(data, channels)

        # filter and decimate the EEG modality data
        self._filter_decimate_and_set_EEG_signals(data, lowcut=lowcut, highcut=highcut, q=q)

        # set the ECG modality with ECG signals
        self._extract_ECG_data(data, channels)

        # set the IBI modality computed from Porti ECG signals; IBIs are  interpolated to Fs_IBI [Hz]
        self._compute_IBI(self.data['ECG'], Fs_IBI = 4)



        return

    def _mount_EEG_data(self, data, channels):
        # mount EEG data to M1 and M2 channels; do it separately for caregiver and child as they have different references
        for ch in self.channel_names['EEG_ch']:
            if ch in channels:
                idx = channels[ch]
                data[idx, :] = data[idx, :] - 0.5 * (data[channels['M1'], :] + data[channels['M2'], :])
        for ch in self.channel_names['EEG_cg']:
            if ch in channels:
                idx = channels[ch]
                data[idx, :] = data[idx, :] - 0.5 * (data[channels['M1_cg'], :] + data[channels['M2_cg'], :])
        # adjust channel lists by removeing channels M1 and M2 from the caregiver and child EEG channels, as they will not be used after linked ears montage
        self.channel_names['EEG_ch'] = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz',
                           'P4', 'T6', 'O1', 'O2']
        self.channel_names['EEG_cg'] = ['Fp1_cg', 'Fp2_cg', 'F7_cg', 'F3_cg', 'Fz_cg', 'F4_cg', 'F8_cg', 'T3_cg', 'C3_cg', 'Cz_cg',
                           'C4_cg', 'T4_cg', 'T5_cg', 'P3_cg', 'Pz_cg', 'P4_cg', 'T6_cg', 'O1_cg', 'O2_cg']
        self.channel_names['EEG'] = self.channel_names['EEG_ch'] + self.channel_names['EEG_cg']
        return data

    def _filter_decimate_and_set_EEG_signals(self, data, lowcut=4.0, highcut=40.0, q=8):
        
        # design EEG filters
        b_notch, a_notch = iirnotch(50, 30, fs=self.Fs['EEG'])
        b_low, a_low = butter(4, highcut, btype='low', fs=self.Fs['EEG'])
        b_high, a_high = butter(4, lowcut, btype='high', fs=self.Fs['EEG'])

        #  arrays for filtered EEG signals
        EEG_cg = np.zeros((len(self.channel_names['EEG_cg']), data.shape[1]))
        channels_cg = {}
        EEG_ch = np.zeros((len(self.channel_names['EEG_ch']), data.shape[1]))
        channels_ch = {}

        # filter the caregiver EEG data
        chan_counter = 0
        for idx, ch in enumerate(self.channels['EEG']):
            if ch in self.channel_names['EEG_cg']:     
                signal = data[idx, :].copy()
                signal = filtfilt(b_notch, a_notch, signal, axis=0)
                signal = filtfilt(b_low, a_low, signal, axis=0)
                signal = filtfilt(b_high, a_high, signal, axis=0)
                EEG_cg[chan_counter, :] = signal
                channels_cg[ch] = chan_counter
                chan_counter += 1

        # filter the child EEG data
        chan_counter = 0
        for idx, ch in enumerate(self.channels['EEG']):
            if ch in self.channel_names['EEG_ch']:
                signal = data[idx, :].copy()
                signal = filtfilt(b_notch, a_notch, signal, axis=0)
                signal = filtfilt(b_low, a_low, signal, axis=0)
                signal = filtfilt(b_high, a_high, signal, axis=0)
                EEG_ch[chan_counter, :] = signal
                channels_ch[ch] = chan_counter
                chan_counter += 1

        # decimate the data to reduce the sampling frequency q times
        signal_cg = decimate(EEG_cg, q, axis=-1)
        signal_ch = decimate(EEG_ch, q, axis=-1)

        # set the filtered and decimated EEG data
        self.channels['EEG'] = {'cg': channels_cg, 'ch': channels_ch}
        self.data['EEG'] = {'cg': signal_cg, 'ch': signal_ch}
        self.Fs['EEG'] = self.Fs['EEG'] // q  # new sampling frequency for the EEG data after decimation
        # time vector for the EEG data after decimation
        self.time['EEG'] = np.arange(0, signal_out.shape[1] / self.Fs['EEG'], 1 / self.Fs['EEG'])
        self.Fs['EEG'] = self.Fs['EEG'] // q  # new sampling frequency for the EEG data after decimation
        self.modalities.append('EEG')
        return 
 
    def _extract_ECG_data(self, data, channels):
        t_ECG = np.arange(0, data.shape[1] / self.Fs['ECG'], 1 / self.Fs['ECG'])  # time vector for the ECG data in seconds

        # extract and filter the ECG data
        ECG_ch = data[channels['EKG1'], :] - data[channels['EKG2'], :]
        ECG_cg = data[channels['EKG1_cg'], :] - data[channels['EKG2_cg'], :]

        #design filters:
        b_notch, a_notch = iirnotch(50, 30, fs=self.Fs['ECG'])
        sos_ecg = butter(5, 0.5, btype='high', output="sos", fs=self.Fs['ECG'])
        ECG_ch_filtered = sosfiltfilt(sos_ecg, ECG_ch)
        ECG_ch_filtered = filtfilt(b_notch, a_notch, ECG_ch_filtered)
        ECG_cg_filtered = sosfiltfilt(sos_ecg, ECG_cg)
        ECG_cg_filtered = filtfilt(b_notch, a_notch, ECG_cg_filtered)
        self.data['ECG'] = {'ch': ECG_ch_filtered, 'cg': ECG_cg_filtered}
        self.times['ECG'] = t_ECG
        self.modalities.append('ECG')
        return None

    def _compute_IBI(self, Fs_IBI=4):
        # interpolate IBI signals from ECG data
        self.Fs['IBI'] = Fs_IBI 
        IBI_ch_interp, t_IBI_ch = self._interpolate_IBI_signals(self.data['ECG']['ch'], self.Fs['ECG'], label='child', plot_flag=False)
        IBI_cg_interp, t_IBI_cg = self._interpolate_IBI_signals(self.data['ECG']['cg'], self.Fs['ECG'], label='caregiver', plot_flag=False)
        # check if the IBI signals are of the same length
        if len(IBI_ch_interp) != len(IBI_cg_interp):
            min_length = min(len(IBI_ch_interp), len(IBI_cg_interp))
            IBI_ch_interp = IBI_ch_interp[:min_length]
            IBI_cg_interp = IBI_cg_interp[:min_length]
            t_IBI_ch = t_IBI_ch[:min_length]
            t_IBI_cg = t_IBI_cg[:min_length]

        t_IBI = t_IBI_ch  # use the time vector for the child IBI as it is the same length as the caregiver IBI
        self.data['IBI'] = {'ch': IBI_ch_interp, 'cg': IBI_cg_interp}   
        self.times['IBI'] = t_IBI
        self.modalities.append('IBI')
        return None

    def _interpolate_IBI_signals(self,ECG, label='', plot_flag=False):
        # Extract R-peaks location
        _, info_ECG = nk.ecg_process(ECG, sampling_rate=self.Fs['ECG'], method='neurokit')
        rpeaks = info_ECG["ECG_R_Peaks"]

        IBI = np.diff(rpeaks) / self.Fs['ECG'] * 1000  # IBI in ms
        t = np.cumsum(IBI) / 1000  # time vector for the IBI signals [s]
        t_ECG = np.arange(0, t[-1], 1 / self.Fs['IBI'])  # time vector for the interpolated IBI signals
        cs = CubicSpline(t, IBI)
        IBI_interp = cs(t_ECG)
        if plot_flag:
            plt.figure(figsize=(12, 6))
            plt.plot(t_ECG, IBI_interp)
            plt.xlabel('time [s]')
            plt.ylabel('IBI [ms]')
            plt.title(f'Interpolated IBI signal of {label} as a function of time')
            plt.show()
        return IBI_interp, t_ECG

    @staticmethod
    def load_output_data(filename):
        try:
            results = joblib.load(filename)
            for key, val in results.items():
                if isinstance(val, np.ndarray):
                    print(f"{key}: {val.shape}")
                else:
                    print(f"{key}: {val}")
            return results
        except FileNotFoundError:
            print(f"File not found {filename}")
    def _save_to_file(self):
        joblib.dump(self.output,self.output_dir+f"{self.ID}.joblib")
    # method for load Warsaw_Data_Frame.csv
    def _load_csv_data(self, csv_file):
        pass



    def _scan_for_events(self, threshold=20000, plot_flag=False):
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

        Fs_EEG = self.Fs['EEG']
        x = np.zeros(self.diode.shape)
        d = self.diode.copy()
        d /= threshold  
        x[d > 1] = 1
        if plot_flag:
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
        if plot_flag:
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

