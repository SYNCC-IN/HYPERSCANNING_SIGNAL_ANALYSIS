import numpy as np  # type: ignore
import os
import xmltodict  # type: ignore    
import matplotlib.pyplot as plt  # type: ignore
from scipy.signal import iirnotch, butter, sosfiltfilt, filtfilt, decimate  # type: ignore
from scipy.interpolate import CubicSpline  # type: ignore
import neurokit2 as nk  # type: ignore
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def load_warsaw_pilot_data(folder, file, plot=False):   

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

    if plot:       
        ECG_CH = data[ChanNames.index('EKG1'),:] - data[ChanNames.index('EKG2'),:]
        ECG_CG = data[ChanNames.index('EKG1_cg'),:] - data[ChanNames.index('EKG2_cg'),:]
        fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
        ax[0].plot(ECG_CH, label='Child ECG')
        ax[1].plot(ECG_CG, label='Caregiver ECG')
        plt.legend()
        plt.show()

    return {
            'data': data,
            'Fs_EEG': Fs_EEG,
            'ChanNames': ChanNames,
            'channels': channels
        }   

def scan_for_events(data, threshold= 20000,plot = True):
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
    diode = data['data'][ diode_idx,:]
    Fs_EEG = data['Fs_EEG']
    x = np.zeros(diode.shape)
    d = diode.copy()
    d /= threshold
    x[d>1]=1
    if plot:
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
    up[y==1]=1
    down[y==-1]=1
    if plot:
        plt.plot(up, 'g', label='Up Events')
        plt.plot(down, 'm', label='Down Events')
        plt.legend()
    
    dt = 17 #ms between frames
    i = 0
    while i< len(down):
        if down[i]==1:
            s1 = int(np.sum(up[i+int(0.5*Fs_EEG)-2*dt : i+int(0.5*Fs_EEG)+2*dt]) )
            s2 = int(np.sum(up[i+int(1.0*Fs_EEG)-3*dt: i+int(1.0*Fs_EEG)+3*dt]))
            s3 = int(np.sum(up[i+int(1.5*Fs_EEG)-4*dt : i+int(1.5*Fs_EEG)+4*dt]))
            s4 = int(np.sum(up[i+int(2.0*Fs_EEG)-5*dt : i+int(2.0*Fs_EEG)+5*dt]))
            s5 = int(np.sum(up[i+int(2.5*Fs_EEG)-6*dt : i+int(2.5*Fs_EEG)+6*dt]))
            #plt.plot(x, 'b'), plt.plot(i,x[i],'bo')
            if s1 ==1 and s2 == 0 and s3 == 0 and s4 == 0 and s5 == 0:
                print(f"Movie 1 starts at {i/Fs_EEG:.2f} seconds")
                events['Movie_1'] = i/Fs_EEG
                if plot:
                    plt.plot(x, 'b'), plt.plot(i,x[i],'ro')
                i += int(2.5*Fs_EEG)
            elif s1 == 1 and s2 == 0 and s3 == 1 and s4 == 0 and s5 == 0:
                print(f"Movie 2 starts at {i/Fs_EEG:.2f} seconds")
                events['Movie_2'] = i/Fs_EEG
                if plot:
                    plt.plot(x, 'b'), plt.plot(i,x[i],'go')
                i += int(2.5*Fs_EEG)
            elif s1 == 1 and s2 == 0 and s3 == 1 and s4 == 0    and s5 == 1:
                print(f"Movie 3 starts at {i/Fs_EEG:.2f} seconds")
                events['Movie_3'] = i/Fs_EEG
                if plot:
                    plt.plot(x, 'b'), plt.plot(i,x[i],'yo')
                i += int(2.5*Fs_EEG)
            elif s1 == 0 and s2 == 1 and s3 == 0 and s4 == 0 and s5 == 0:
                if  events['Talk_1'] is None:
                    print(f"Talk 1 starts at {i/Fs_EEG:.2f} seconds")
                    events['Talk_1'] = i/Fs_EEG
                    if plot:
                        plt.plot(x, 'b'), plt.plot(i,x[i],'co')
                    i += int(2.5*Fs_EEG)
                else:
                    print(f"Talk 2 starts at {i/Fs_EEG:.2f} seconds") 
                    events['Talk_2'] = i /Fs_EEG
                    if plot:
                        plt.plot(x, 'b'), plt.plot(i,x[i],'mo')   
                        plt.show()
                    i = len(down)       # talk 2 is the last event so finish scaning for events        
        i += 1
    return events

def interpolate_IBI_signals(ECG, Fs_ECG, Fs_IBI=4, plot=False, label = ''):
    # Extract R-peaks location
    _, info_ECG= nk.ecg_process(ECG, sampling_rate=Fs_ECG, method='neurokit')
    rpeaks = info_ECG["ECG_R_Peaks"]

    IBI = np.diff(rpeaks)/Fs_ECG*1000 # IBI in ms
    t = np.cumsum(IBI)/1000 # time vector for the IBI signals [s]
    t_ECG = np.arange(0, t[-1], 1/Fs_IBI)    # time vector for the interpolated IBI signals
    cs = CubicSpline(t, IBI)
    IBI_interp = cs(t_ECG)
    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(t_ECG, IBI_interp)
        plt.xlabel('time [s]')
        plt.ylabel('IBI [ms]')
        plt.title(f'Interpolated IBI signal of {label} as a function of time')
        plt.show()
    return IBI_interp, t_ECG

def get_IBI_signal_from_ECG_for_selected_event(filtered_data, events, selected_event, plot=False, label=''):
    '''Get IBI signal from ECG data for a specific event.
    Args:
        filtered_data (dict): Filtered data with the structure returned by filter_warsaw_pilot_data function.
        events (dict): Dictionary containing the start and end time of detected events.
        selected_event (str): Name of the event to extract IBI signal for
        Fs_IBI (int): Sampling frequency for the IBI signals.
        plot (bool): Whether to plot the IBI signal.
        label (str): Label for the plot.
    Returns:
        IBI_ch_interp (np.ndarray): Interpolated IBI signal for the child.
        IBI_cg_interp (np.ndarray): Interpolated IBI signal for the caregiver.
        t_ECG (np.ndarray): Time vector for the interpolated IBI signal.
    '''
    if selected_event not in events:
        raise ValueError(f"Event '{selected_event}' not found in events dictionary.")   
        # extract the ECG signal for the selected event
    IBI_ch_interp = filtered_data['IBI_ch_interp']
    IBI_cg_interp = filtered_data['IBI_cg_interp']
    t_IBI = filtered_data['t_IBI']
    t_idx = events[selected_event] # get the time of the event in the data
    if t_idx is not None:
        # extract 60 seconds after the event
        #find the index in t_IBI
        start_idx = np.searchsorted(t_IBI, t_idx)  # find the index in t_IBI    
        end_idx = start_idx + int(60 * filtered_data['Fs_IBI'])  # extract 60 seconds after the event
        # check if the start and end indices are within the bounds of the data
        if start_idx < 0 or end_idx > filtered_data['data'].shape[1]:
            raise ValueError(f"Event '{selected_event}' is out of bounds.")
    else:
        raise ValueError(f"Event '{selected_event}' is None.")

    # cut the IBI signal of the selected event
    IBI_ch_interp = IBI_ch_interp[start_idx:end_idx]
    IBI_cg_interp = IBI_cg_interp[start_idx:end_idx]
    t_IBI = t_IBI[start_idx:end_idx]

    return IBI_ch_interp, IBI_cg_interp, t_IBI

def get_data_for_selected_channel_and_event(filtered_data, selected_channels, events, selected_event):
    '''Get data for selected channels and event from the filtered data.
    Args:
        data (dict): Filtered data with the structure returned by filter_warsaw_pilot_data function.
        selected_channels (list): List of channel names to extract data for.
        events (dict): Dictionary containing the start and end time of detected events.
        selected_event (str): Name of the event to extract data for.
    Returns:    
        data_selected (np.ndarray): Data array with the shape (N_samples, N_channels) for the selected channels and event.
    '''
    if selected_event not in events:
        raise ValueError(f"Event '{selected_event}' not found in events dictionary.")
    idx = events[selected_event]
    if idx is not None:
        # extract 60 seconds after the event
        data_selected = np.zeros((len(selected_channels), int(60 * filtered_data['Fs_EEG'])))
        start_idx = int(idx * filtered_data['Fs_EEG'])  # convert the event time to the index in the filtered data
        end_idx = start_idx + int(60 * filtered_data['Fs_EEG'])  # extract 60 seconds after the event
        # check if the start and end indices are within the bounds of the data
        if start_idx < 0 or end_idx > filtered_data['data'].shape[1]:
            raise ValueError(f"Event '{selected_event}' is out of bounds.") 
    for i, ch in enumerate(selected_channels):
        if ch in filtered_data['channels']:
            idx_ch = filtered_data['channels'][ch]
            data_selected[i, :] = filtered_data['data'][idx_ch, start_idx:end_idx]
    return data_selected

def filter_warsaw_pilot_data(data, lowcut=4.0, highcut=40.0, q = 8):

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
            'data': filtered signal,
            't_EEG': time vector for the EEG data after decimation,
            'Fs_EEG': sampling frequency of the EEG data after decimation,
            'ChanNames': list of channel names,
            'channels': channels dictionary with indexes referring to 'data' array,
            'EEG_channels_ch': list of names of child EEG channels,
            'EEG_channels_cg': list of names of caregiver EEG channels,
            'ECG_ch': filtered ECG signal for the child,
            'ECG_cg': filtered ECG signal for the caregiver,
            't_ECG': time vector for the ECG data
            'Fs_ECG': sampling frequency of the ECG data,
            'IBI_ch_interp': Interpolated IBI signal for the child,
            'IBI_cg_interp': Interpolated IBI signal for the caregiver,
            'Fs_IBI': Sampling frequency for the IBI signals,
            't_IBI': Time vector for the interpolated IBI signal

            }
    '''
    signal = data['data'].copy()
    signal *= 0.0715 # scale the signal to microvolts
    Fs_EEG = data['Fs_EEG']
    Fs_ECG = Fs_EEG # ECG data is sampled at the same frequency as EEG data
    Fs_IBI = 4 # sampling frequency [Hz] for the IBI signals
    t_ECG = np.arange(0, signal.shape[1] / Fs_EEG, 1 / Fs_EEG) # time vector for the ECG data

    channels= data['channels']
    b_notch, a_notch = iirnotch(50, 30, fs=Fs_EEG)

    # extract and filter the ECG data
    ECG_ch = data['data'][channels['EKG1'],:] - data['data'][channels['EKG2'],:]
    ECG_cg = data['data'][channels['EKG1_cg'],:] - data['data'][channels['EKG2_cg'],:]
    sos_ecg = butter(5, 0.5, btype='high' , output="sos", fs=Fs_EEG)
    ECG_ch_filtered = sosfiltfilt(sos_ecg, ECG_ch)
    ECG_ch_filtered = filtfilt(b_notch, a_notch, ECG_ch_filtered)
    ECG_cg_filtered = sosfiltfilt(sos_ecg, ECG_cg)
    ECG_cg_filtered = filtfilt(b_notch, a_notch, ECG_cg_filtered)
    # interpolate IBI signals from ECG data
    IBI_ch_interp, t_IBI_ch = interpolate_IBI_signals(ECG_ch_filtered, Fs_ECG, Fs_IBI=Fs_IBI, plot=False, label='')
    IBI_cg_interp, t_IBI_cg = interpolate_IBI_signals(ECG_cg_filtered, Fs_ECG, Fs_IBI=Fs_IBI, plot=False, label='')
    # check if the IBI signals are of the same length
    if len(IBI_ch_interp) != len(IBI_cg_interp):
        min_length = min(len(IBI_ch_interp), len(IBI_cg_interp))
        IBI_ch_interp = IBI_ch_interp[:min_length]
        IBI_cg_interp = IBI_cg_interp[:min_length]
        t_IBI_ch = t_IBI_ch[:min_length]
        t_IBI_cg = t_IBI_cg[:min_length]

    t_IBI = t_IBI_ch  # use the time vector for the child IBI as it is the same length as the caregiver IBI 
    # define EEG channels for child and caregiver
    EEG_channels_ch = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'M1', 'T3', 'C3', 'Cz', 'C4', 'T4', 'M2', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2'] 
    EEG_channels_cg = ['Fp1_cg', 'Fp2_cg', 'F7_cg', 'F3_cg', 'Fz_cg', 'F4_cg', 'F8_cg', 'M1_cg', 'T3_cg', 'C3_cg', 'Cz_cg', 'C4_cg', 'T4_cg', 'M2_cg', 'T5_cg', 'P3_cg', 'Pz_cg', 'P4_cg', 'T6_cg', 'O1_cg', 'O2_cg']
   
    # design EEG filters
    b_low, a_low = butter(4, highcut, btype='low', fs = Fs_EEG)
    b_high, a_high = butter(4, lowcut, btype='high', fs = Fs_EEG)

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
            signal[idx, :] = signal[idx, :] - 0.5*(signal[data['channels']['M1_cg'],:] +signal[data['channels']['M2_cg'],:] ) 
    # remove channels M1 and M2 from the caregiver EEG channels, as they will not be used after linked ears montage
    EEG_channels_cg = ['Fp1_cg', 'Fp2_cg', 'F7_cg', 'F3_cg', 'Fz_cg', 'F4_cg', 'F8_cg', 'T3_cg', 'C3_cg', 'Cz_cg', 'C4_cg', 'T4_cg', 'T5_cg', 'P3_cg', 'Pz_cg', 'P4_cg', 'T6_cg', 'O1_cg', 'O2_cg'] 
    

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
            signal[idx, :] = signal[idx, :] - 0.5*(signal[data['channels']['M1'],:] +signal[data['channels']['M2'],:] ) 
    # remove channels M1 and M2 from the child EEG channels, as thye will not be used after linked ears montage
    EEG_channels_ch = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2'] 
    
    # decimate the data to reduce the sampling frequency q times
    signal_out = decimate(signal, q, axis=-1) 
    Fs_EEG_q = Fs_EEG // q # new sampling frequency for the EEG data after decimation
    # time vector for the EEG data after decimation
    t_EEG = np.arange(0, signal_out.shape[1] / Fs_EEG_q, 1 / Fs_EEG_q) # 

    filtered_data = {
        'data': signal_out,
        't_EEG': t_EEG, # time vector for the EEG data after decimation
        'Fs_EEG': Fs_EEG_q, # signal is decimated to this frequency
        'ChanNames': data['ChanNames'], # list of channel names
        'channels': data['channels'], #  channels dictionary with indexes referenig to 'data' array
        'EEG_channels_ch': EEG_channels_ch, # list of names of child EEG channels
        'EEG_channels_cg': EEG_channels_cg, # list of names of caregiver EEG channels
        'ECG_ch': ECG_ch_filtered,
        'ECG_cg': ECG_cg_filtered,
        'Fs_ECG': Fs_ECG, # This is the original sampling frequency of the ECG data
        't_ECG': t_ECG, # time vector for the ECG data
        'IBI_ch_interp': IBI_ch_interp, # Interpolated IBI signal for the child
        'IBI_cg_interp': IBI_cg_interp, # Interpolated IBI signal for the caregiver
        'Fs_IBI': Fs_IBI, # Sampling frequency for the IBI signals
        't_IBI': t_IBI # Time vector for the interpolated IBI signal
    }
    return filtered_data

def clean_data_with_ICA(data, selected_channels, event):
    '''Clean data with ICA to remove artifacts.
    Args:
        data (np.ndarray): Data array with the shape (N_channels, N_samples) for the selected channels and event.
        selected_channels (list): List of channel names to extract data for.
        event (str): Name of the event to extract data for.
        plot (bool): Whether to plot the data before and after ICA.
    Returns:
        data_cleaned (np.ndarray): Cleaned data array with the shape (N_channels, N_samples) for the selected channels and event.
    '''
    from sklearn.decomposition import FastICA
    ica = FastICA(n_components=len(selected_channels), max_iter=1000, whiten="unit-variance")
    S_ = ica.fit_transform(data.T)  # get components
   
    fig, ax = plt.subplots(len(selected_channels), 1, figsize=(12, 8), sharex=True)
    for i, ch in enumerate(selected_channels):
        ax[i].plot(S_[:,i])
        ax[i].set_ylabel(ch)
    plt.tight_layout()
    plt.show()   
    idx_to_remove = input(f'Event {event}: select components to remove and press Enter to continue...  ')  
    if idx_to_remove != '': 
        idx_to_remove = [int(i) for i in idx_to_remove.split(',')]
        S_[:,idx_to_remove] = 0 # set the selected components to zero
        print('Selected components to remove: ', idx_to_remove)
    data_cleaned = ica.inverse_transform(S_).T  # reconstruct the data from the components   
    return data_cleaned


def describe_dict(d):
    for key, value in d.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: ndarray, shape={value.shape}, dtype={value.dtype}")
        elif isinstance(value, list):
            print(f"{key}: list, length={len(value)}")
        elif isinstance(value, dict):
            print(f"{key}: dict, keys={list(value.keys())}")
        else:
            print(f"{key}: {type(value).__name__}, value={value}")

### PLOTS ####


def plot_EEG_channels(filtered_data, events, selected_channels, title='Filtered EEG Channels'):
    """
    Plot the filtered EEG channels with events highlighted.
    """
    
    colors = ['r', 'g', 'y', 'c', 'm']  # colors for different events
    fig, ax = plt.subplots(1, 1, figsize=(12, 8), sharex=True)
    offset = 0
    spacing = 200  # vertical spacing between channels
    yticks = []
    yticklabels = []
    
    for i, ch in enumerate(selected_channels):
        if ch in filtered_data['channels']:
            idx = filtered_data['channels'][ch]
            x_ch = filtered_data['data'][idx, :]
            # clip the amplitudes
            x_ch = np.clip(x_ch, -100, 100)
            ax.plot(filtered_data['t_EEG'], x_ch + offset, label=ch)
            yticks.append(offset)
            yticklabels.append(ch)
            offset += spacing
            
    for i, event in enumerate(events):
        if events[event] is not None:
            ax.axvspan(events[event], events[event] + 60, color=colors[i], alpha=0.2, label=f'{event} (60s)')

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_title(title)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Channels')
    plt.tight_layout()   
    plt.legend()
    plt.show()

def plot_EEG_channels_pl(filtered_data, events, selected_channels, title='Filtered EEG Channels', renderer='auto'):
    """
    Plot the filtered EEG channels with events highlighted using Plotly.
    Replicates the matplotlib version with vertical offsets in a single plot.
    Features interactive hover information and zooming capabilities.
    
    Parameters:
    -----------
    filtered_data : dict
        Dictionary containing EEG data, channels, and time vectors
    events : dict
        Dictionary containing event timings
    selected_channels : list
        List of channel names to plot
    title : str, optional
        Title for the plot (default: 'Filtered EEG Channels')
    renderer : str, optional
        Plotly renderer to use: 'auto', 'browser', 'notebook', 'html' (default: 'auto')
    """
    colors = ['red', 'green', 'blue', 'orange', 'purple']  # colors for different events
    
    # Create a single figure
    fig = go.Figure()
    
    offset = 0
    spacing = 200  # vertical spacing between channels
    yticks = []
    yticklabels = []
    
    # Plot each channel with vertical offset
    for i, ch in enumerate(selected_channels):
        if ch in filtered_data['channels']:
            idx = filtered_data['channels'][ch]
            x_ch = filtered_data['data'][idx, :]
            # clip the amplitudes
            x_ch = np.clip(x_ch, -100, 100)
            
            # Add trace for this channel with offset
            fig.add_trace(go.Scatter(
                x=filtered_data['t_EEG'], 
                y=x_ch + offset, 
                mode='lines', 
                name=ch,
                line=dict(width=1),
                showlegend=True
            ))
            
            yticks.append(offset)
            yticklabels.append(ch)
            offset += spacing
    
    # Add event highlights as vertical rectangles spanning all channels
    event_colors_used = []
    for i, event in enumerate(events):
        if events[event] is not None:
            color_idx = i % len(colors)
            fig.add_vrect(
                x0=events[event], 
                x1=events[event] + 60,
                fillcolor=colors[color_idx], 
                opacity=0.2,
                layer="below", 
                line_width=0,
                annotation_text=f'{event} (60s)',
                annotation_position="top left",
                annotation=dict(
                    font=dict(size=10, color=colors[color_idx]),
                    bgcolor="white",
                    bordercolor=colors[color_idx],
                    borderwidth=1
                )
            )
            if color_idx not in event_colors_used:
                # Add invisible trace for legend
                fig.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode='markers',
                    marker=dict(size=10, color=colors[color_idx]),
                    name=f'{event} Events',
                    showlegend=True
                ))
                event_colors_used.append(color_idx)
    
    # Update layout to match matplotlib appearance with enhanced interactivity
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=16)
        ),
        xaxis_title="Time [s]",
        yaxis_title="EEG Channels",
        yaxis=dict(
            tickvals=yticks,
            ticktext=yticklabels,
            showgrid=False,
            zeroline=False
        ),
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        ),
        height=600,
        width=1200,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.01,
            font=dict(size=10)
        ),
        #hovermode='x unified',
        plot_bgcolor='white'
    )
    
    # Add range selector for time navigation
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=30, label="30s", step="second", stepmode="backward"),
                    dict(count=60, label="1m", step="second", stepmode="backward"),
                    dict(count=300, label="5m", step="second", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True, thickness=0.05),
            type="linear"
        )
    )
    
    # Show the figure based on renderer preference
    if renderer == 'html':
        # Save as HTML file
        import os
        html_file = f"{title.replace(' ', '_').replace('(', '').replace(')', '').lower()}.html"
        fig.write_html(html_file)
        print(f"Plot saved as HTML file: {os.path.abspath(html_file)}")
        print("Open this file in your web browser to view the interactive plot.")
    elif renderer == 'auto':
        # Try different renderers in order of preference
        try:
            fig.show(renderer="browser")
        except:
            try:
                fig.show(renderer="notebook")
            except:
                # Fallback: save as HTML file
                import os
                html_file = f"{title.replace(' ', '_').replace('(', '').replace(')', '').lower()}.html"
                fig.write_html(html_file)
                print(f"Plot saved as HTML file: {os.path.abspath(html_file)}")
                print("Open this file in your web browser to view the interactive plot.")
    else:
        # Use specified renderer
        try:
            fig.show(renderer=renderer)
        except Exception as e:
            print(f"Failed to display with renderer '{renderer}': {e}")
            # Fallback to HTML
            import os
            html_file = f"{title.replace(' ', '_').replace('(', '').replace(')', '').lower()}.html"
            fig.write_html(html_file)
            print(f"Plot saved as HTML file: {os.path.abspath(html_file)}")
            print("Open this file in your web browser to view the interactive plot.")


def overlay_EEG_channels_hyperscanning(data_ch, data_cg, all_channels, event, selected_channels_ch, selected_channels_cg, title='Filtered EEG Channels - Hyperscanning'):
    """
    Overlay EEG channels for child and caregiver during a specific event.
    """
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax[0].set_title(f'Child EEG channels for {event}')
    ax[1].set_title(f'Caregiver EEG channels for {event}')
    for i, ch in enumerate(selected_channels_ch):   
        if ch in all_channels:
            #idx_ch = filtered_data['channels'][ch]
            ax[0].plot(data_ch[i, :], label=ch)
    for i, ch in enumerate(selected_channels_cg):       
        if ch in all_channels:
            #idx_cg = filtered_data['channels'][ch]
            ax[1].plot(data_cg[i, :], label=ch)
    ax[0].set_ylabel('Amplitude [uV]')
    ax[1].set_ylabel('Amplitude [uV]')
    ax[1].set_xlabel('Samples')
    ax[0].legend(loc='upper right')
    ax[1].legend(loc='upper right')
    plt.suptitle(title)
    plt.tight_layout()

def overlay_EEG_channels_hyperscanning_pl(data_ch, data_cg, all_channels, event, selected_channels_ch, selected_channels_cg, title='Filtered EEG Channels - Hyperscanning', renderer='auto'):
    """
    Plot child and caregiver EEG channels for hyperscanning analysis using Plotly.
    Creates two subplots: one for child channels and one for caregiver channels.
    
    Parameters:
    -----------
    data_ch : numpy.ndarray
        Child EEG data (channels x samples)
    data_cg : numpy.ndarray  
        Caregiver EEG data (channels x samples)
    all_channels : dict
        Dictionary of all available channels
    event : str
        Name of the event being plotted
    selected_channels_ch : list
        List of selected child channel names
    selected_channels_cg : list
        List of selected caregiver channel names
    title : str, optional
        Title for the plot (default: 'Filtered EEG Channels - Hyperscanning')
    renderer : str, optional
        Plotly renderer to use: 'auto', 'browser', 'notebook', 'html' (default: 'auto')
    """
    
    # Create subplots with 2 rows
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(f'Child EEG channels for {event}', f'Caregiver EEG channels for {event}'),
        shared_xaxes=True,
        vertical_spacing=0.1
    )
    
    # Color palette for channels
    colors = px.colors.qualitative.Set3
    
    # Plot child EEG channels
    for i, ch in enumerate(selected_channels_ch):
        if ch in all_channels:
            color_idx = i % len(colors)
            fig.add_trace(
                go.Scatter(
                    x=list(range(data_ch.shape[1])),
                    y=data_ch[i, :],
                    mode='lines',
                    name=ch,
                    line=dict(color=colors[color_idx], width=1.5),
                    legendgroup='child',
                    legendgrouptitle_text="Child Channels"
                ),
                row=1, col=1
            )
    
    # Plot caregiver EEG channels  
    for i, ch in enumerate(selected_channels_cg):
        if ch in all_channels:
            color_idx = i % len(colors)
            fig.add_trace(
                go.Scatter(
                    x=list(range(data_cg.shape[1])),
                    y=data_cg[i, :],
                    mode='lines',
                    name=ch,
                    line=dict(color=colors[color_idx], width=1.5),
                    legendgroup='caregiver',
                    legendgrouptitle_text="Caregiver Channels"
                ),
                row=2, col=1
            )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=16)
        ),
        height=800,
        width=1200,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.01,
            font=dict(size=10),
            groupclick="toggleitem"
        ),
        #hovermode='x unified',
        plot_bgcolor='white'
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Samples", row=2, col=1)
    fig.update_yaxes(title_text="Amplitude (µV)", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude (µV)", row=2, col=1)
    
    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    # Show the figure based on renderer preference
    if renderer == 'html':
        # Save as HTML file
        import os
        html_file = f"{title.replace(' ', '_').replace('(', '').replace(')', '').lower()}_{event}.html"
        fig.write_html(html_file)
        print(f"Plot saved as HTML file: {os.path.abspath(html_file)}")
        print("Open this file in your web browser to view the interactive plot.")
    elif renderer == 'auto':
        # Try different renderers in order of preference
        try:
            fig.show(renderer="browser")
        except:
            try:
                fig.show(renderer="notebook")
            except:
                # Fallback: save as HTML file
                import os
                html_file = f"{title.replace(' ', '_').replace('(', '').replace(')', '').lower()}_{event}.html"
                fig.write_html(html_file)
                print(f"Plot saved as HTML file: {os.path.abspath(html_file)}")
                print("Open this file in your web browser to view the interactive plot.")
    else:
        # Use specified renderer
        try:
            fig.show(renderer=renderer)
        except Exception as e:
            print(f"Failed to display with renderer '{renderer}': {e}")
            # Fallback to HTML
            import os
            html_file = f"{title.replace(' ', '_').replace('(', '').replace(')', '').lower()}_{event}.html"
            fig.write_html(html_file)
            print(f"Plot saved as HTML file: {os.path.abspath(html_file)}")
            print("Open this file in your web browser to view the interactive plot.")

