import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, iirnotch, filtfilt, sosfiltfilt, decimate, hilbert, welch
from scipy.stats import zscore  # type: ignore
from mtmvar import mvar_criterion, AR_coeff, mvar_H, mvar_plot,  mvar_plot_dense, mvar_spectra # type: ignore
from utils import load_warsaw_pilot_data, scan_for_events, filter_warsaw_pilot_data, get_IBI_signal_from_ECG_for_selected_event, get_data_for_selected_channel_and_event, clean_data_with_ICA

if __name__ == "__main__":
    folder = './DATA/W_010/' #'./DATA/W_009/'
    file  =  'W_010.obci'   #'W_009.obci'
    
    debug_PLOT = True
    HRV_DTF = True # if True, the DTF will be estimated for the IBI signals from the ECG amplifier
    EEG_DTF = True # if True, the DTF will be estimated for the EEG signals from child and caregiver separately
    EEG_HRV_DTF = True

    data = load_warsaw_pilot_data(folder, file, plot=False)
    events = scan_for_events(data, plot = True) #indexes of events in the data, this is done before filtering to avoid artifacts in the diode signal
    filtered_data = filter_warsaw_pilot_data(data)
    if debug_PLOT:
        print("Filtered data shape:", filtered_data['data'].shape)
        print("Filtered EEG channels:", filtered_data['EEG_channels_ch'])
        print("Filtered ECG channels:", filtered_data['EEG_channels_cg'])
        print("Events detected:", events)

        # separately (in subplots) for child and caregiver, plot the filtered ECG and overall it with the interpolated IBI signals, highlithing the events

        fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        # Plot Child ECG on left y-axis
        ax[0].plot(filtered_data['t_ECG'], filtered_data['ECG_ch'], label='Child ECG', color='tab:blue')
        ax[0].set_ylabel('ECG (uV)', color='tab:blue')
        ax[0].tick_params(axis='y', labelcolor='tab:blue')

        # Create a twin y-axis to plot IBI
        ax0b = ax[0].twinx()
        ax0b.plot(filtered_data['t_IBI'], filtered_data['IBI_ch_interp'], label='Child IBI', color='tab:orange')
        ax0b.set_ylabel('IBI (ms)', color='tab:orange')
        ax0b.tick_params(axis='y', labelcolor='tab:orange')
        ax[0].plot(filtered_data['t_IBI'], filtered_data['IBI_ch_interp'], label='Child IBI')
        colors = ['r', 'g', 'y', 'c', 'm']  # colors for different events
        for i, event in enumerate(events):
            if events[event] is not None:
                ax[0].axvspan(events[event], events[event] + 60, color=colors[i], alpha=0.2, label=f'{event} (60s)')
        ax[0].legend()

        ax[1].plot(filtered_data['t_ECG'], filtered_data['ECG_cg'], label='Caregiver ECG', color='tab:blue')
        ax[1].set_ylabel('ECG (uV)', color='tab:blue')
        ax[1].tick_params(axis='y', labelcolor='tab:blue')     
        # Create a twin y-axis to plot IBI
        ax1b = ax[1].twinx()
        ax1b.plot(filtered_data['t_IBI'], filtered_data['IBI_cg_interp'], label='Caregiver IBI', color='tab:orange')
        ax1b.set_ylabel('IBI (ms)', color='tab:orange')
        ax1b.tick_params(axis='y', labelcolor='tab:orange')
        for i, event in enumerate(events):
            if events[event] is not None:
                ax[1].axvspan(events[event], events[event] + 60, color=colors[i], alpha=0.2, label=f'{event} (60s)')
        ax[1].legend()
        ax[1].set_xlabel('Time (s)')
        plt.suptitle('Filtered ECG and IBI signals with events highlighted')
        plt.tight_layout()
        plt.show()


        # plot the filtered EEG channels
        fig, ax = plt.subplots(1, 1, figsize=(12, 8), sharex=True)
        offset = 0
        spacing = 200  # vertical spacing between channels
        yticks = []
        yticklabels = []
        for i, ch in enumerate(filtered_data['EEG_channels_ch']):
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
        ax.set_title('Filtered Child EEG Channels (offset for clarity)')
        ax.set_xlabel('Samples')
        ax.set_ylabel('Channels')
        plt.tight_layout()   
        plt.show()

        # plot the filtered EEG channels for caregiver
        fig, ax = plt.subplots(1, 1, figsize=(12, 8), sharex=True)
        offset = 0
        spacing = 200  # vertical spacing between channels
        yticks = []
        yticklabels = []
        for i, ch in enumerate(filtered_data['EEG_channels_cg']):
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
        ax.set_title('Filtered Caregiver EEG Channels (offset for clarity)')
        ax.set_xlabel('Samples')
        ax.set_ylabel('Channels')
        plt.tight_layout()
        plt.legend()
        plt.show()
        
    if HRV_DTF:
        # for each event extract the IBI signals from the ECG amplifier
        # of child and of the caregiver
        # costruct a numpy data array with the shape (N_samples, 2) 
        # and estimate DTF for each event
        selected_events = ['Movie_1', 'Movie_2', 'Movie_3'] # events to extract data for ; #, 'Talk_1', 'Talk_2'

        f = np.arange(0.01, 1, 0.01) # frequency vector for the DTF estimation
        for event in selected_events:
            if event in events:
                # t_event = events[event] # get the time of the event in the data
                # # find the closest index in the IBI signals
                # start_idx = np.argmin(np.abs(t_ECG - t_event))
                # end_idx = start_idx + int(60 * Fs_IBI)

                # extract 60 seconds after the event
                data = np.zeros((2, 60*filtered_data['Fs_IBI']))
                IBI_ch_interp, IBI_cg_interp, t_IBI = get_IBI_signal_from_ECG_for_selected_event(filtered_data, events, event, plot=False, label='IBI signals for ' + event )
                # zscore the IBI signals
                IBI_ch_interp = zscore(IBI_ch_interp) # normalize the IBI   signals
                IBI_cg_interp = zscore(IBI_cg_interp) # normalize the IBI   signals
                data[0, :] = IBI_ch_interp #[start_idx:end_idx]
                data[1, :] = IBI_cg_interp #[start_idx:end_idx]

                crit, p_range, p_opt = mvar_criterion(data, max_p = 15, crit_type = 'AIC', do_plot = False)
                print('Optimal model order for all channels: p = ', str(p_opt))
                Ar, V = AR_coeff(data, p_opt)
                H, _ = mvar_H(Ar, f, filtered_data['Fs_IBI'])
                DTF = np.abs(H)**2
                S, f = mvar_spectra(H, V, f  )
                """Let's  plot the results in the table form."""
                mvar_plot(S, DTF,   f, 'From ', 'To ',['Child', 'Caregiver'],  'DTF '+ event ,'sqrt')
        plt.show()

    
    if EEG_DTF:
        # for each event extract the EEG signals 
        # of child and of the caregiver
        # costruct a numpy data array with the shape (N_samples, 19) 
        # and estimate DTF for each event separately for child and caregiver EEG channels

        f = np.arange(1,30,0.5 ) # frequency vector for the DTF estimation
        selected_channels_ch  = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4', 'O1', 'O2'] #, , 'T3','T4',  'T6',  'T5'
        selected_channels_cg = ['Fp1_cg', 'Fp2_cg', 'F7_cg', 'F3_cg', 'Fz_cg', 'F4_cg', 'F8_cg', 'C3_cg', 'Cz_cg', 'C4_cg', 'P3_cg', 'Pz_cg', 'P4_cg', 'O1_cg', 'O2_cg'] # , 'T3_cg', , 'T4_cg', , 'T5_cg', , 'T6_cg'
        selected_events = ['Movie_1', 'Movie_2', 'Movie_3'] # events to extract data for ; #, 'Talk_1', 'Talk_2'
        for event in selected_events:
            data_ch = get_data_for_selected_channel_and_event(filtered_data, selected_channels_ch, events, event)
            data_cg = get_data_for_selected_channel_and_event(filtered_data, selected_channels_cg, events, event)

            # ICA = True # if True, apply ICA to the EEG data to remove artifacts
            # if ICA: # clean EEG data with ICA separately for child and caregiver EEG channels
            #     data_ch = clean_data_with_ICA(data_ch, selected_channels_ch, event)
            #     data_cg = clean_data_with_ICA(data_cg, selected_channels_cg, event)

            # plot the data for the child and caregiver EEG channels
            fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            ax[0].set_title(f'Child EEG channels for {event}')
            ax[1].set_title(f'Caregiver EEG channels for {event}')
            for i, ch in enumerate(selected_channels_ch):   
                if ch in filtered_data['channels']:
                    idx_ch = filtered_data['channels'][ch]
                    ax[0].plot(data_ch[i, :], label=ch)
            for i, ch in enumerate(selected_channels_cg):       
                if ch in filtered_data['channels']:
                    idx_cg = filtered_data['channels'][ch]
                    ax[1].plot(data_cg[i, :], label=ch)
            ax[0].set_ylabel('Amplitude (uV)')
            ax[1].set_ylabel('Amplitude (uV)')
            ax[1].set_xlabel('Samples (after decimation)')
            ax[0].legend(loc='upper right')
            ax[1].legend(loc='upper right')
            plt.tight_layout()

            p_opt =9
            #crit, p_range, p_opt = mvar_criterion(data_ch, max_p = 15, crit_type = 'AIC', do_plot = False)
            print('Optimal model order for all child channels: p = ', str(p_opt))
            Ar, V = AR_coeff(data_ch, p_opt)
            H, _ = mvar_H(Ar, f, filtered_data['Fs_EEG'])
            DTF = np.abs(H)**2
            S, f = mvar_spectra(H, V, f  )
            mvar_plot_dense(S, DTF,   f, 'From ', 'To ',selected_channels_ch ,  'DTF ch '+ event ,'sqrt')

            #crit, p_range, p_opt = mvar_criterion(data_cg, max_p = 15, crit_type = 'AIC', do_plot = False)
            print('Optimal model order for all caregiver channels: p = ', str(p_opt))
            Ar, V = AR_coeff(data_cg, p_opt)
            H, _ = mvar_H(Ar, f, filtered_data['Fs_EEG'])
            DTF = np.abs(H)**2
            S, f = mvar_spectra(H, V, f  )
            mvar_plot_dense(S, DTF,   f, 'From ', 'To ', selected_channels_cg,  'DTF cg '+ event ,'sqrt')  
            plt.show()


    if EEG_HRV_DTF:
    # Something interesting seems to happen in the theta band in the Fz electrode of both child and caregiver
    # Let's filter the channel Fz in the theta band, get the instantaneous amplitude of the activity and evaluate DTF for the system consisting of
    # HRV and Fz theta instantaneous amplitude of both members
        f = np.arange(0.01,1,0.01) # frequency vector for the DTF estimation
        selected_channels_ch  = ['Fz'] 
        selected_channels_cg = ['Fz_cg']
        selected_events = ['Movie_1', 'Movie_2', 'Movie_3'] 
        # design a bandpass filter for the theta band
        lowcut = 5.0  # Hz
        highcut = 7.5 # Hz
        sos_theta = butter(4, [lowcut, highcut], btype='band', fs=filtered_data['Fs_EEG'], output='sos')
        for event in selected_events:
            data_ch = get_data_for_selected_channel_and_event(filtered_data, selected_channels_ch, events, event)
            data_cg = get_data_for_selected_channel_and_event(filtered_data, selected_channels_cg, events, event)
            # compute and plot spectra of the selected channels using Welch's method
            f_ch, Pxx_ch = welch(data_ch[0, :], fs=filtered_data['Fs_EEG'], nperseg=1024)
            f_cg, Pxx_cg = welch(data_cg[0, :], fs=filtered_data['Fs_EEG'], nperseg=1024)
            plt.figure(figsize=(12, 6))
            plt.plot(f_ch, Pxx_ch, label='Child Fz channel')
            plt.plot(f_cg, Pxx_cg, label='Caregiver Fz_cg channel')
            plt.title(f'Power Spectrum of {event} for Child and Caregiver Fz channels') 
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Power Spectral Density (uV^2/Hz)')
            # Highlight the theta band
            plt.axvspan(lowcut, highcut, color='yellow', alpha=0.5, label='Theta Band (5-7.5 Hz)')
            plt.xlim(0, 30)  # Limit x-axis to 30 Hz
            plt.legend()
            plt.grid()
            plt.show()

            # filter the data in the theta band
            data_ch_theta = sosfiltfilt(sos_theta, data_ch[0, :])
            data_cg_theta = sosfiltfilt(sos_theta, data_cg[0, :])
            # get the instantaneous amplitude (envelpe) of the filtered signal using Hilbert transform
            data_ch_theta_amp = np.abs(hilbert(data_ch_theta))
            data_cg_theta_amp = np.abs(hilbert(data_cg_theta))
            # plot the envelope for the child and caregiver EEG channels, add the filtered signal as the background
            fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            ax[0].set_title(f'Child EEG channel Fz theta amplitude for {event}')
            ax[1].set_title(f'Caregiver EEG channel Fz_cg theta amplitude for {event}')
            ax[0].plot(data_ch_theta_amp,'r', label='Fz theta amplitude')
            ax[1].plot(data_cg_theta_amp, 'r', label='Fz_cg theta amplitude')
            ax[0].plot(data_ch_theta, 'k', alpha=0.5, label='Fz theta filtered signal')
            ax[1].plot(data_cg_theta, 'k', alpha=0.5, label='Fz_cg theta filtered signal')
            ax[0].set_ylabel('Amplitude (uV)')
            ax[1].set_ylabel('Amplitude (uV)')
            ax[0].legend(loc='upper right')
            ax[1].legend(loc='upper right')
            plt.tight_layout()
            plt.show()

            # downsample the envelope to the same frequency as the IBI signals
            data_ch_theta_amp = decimate(data_ch_theta_amp, filtered_data['Fs_EEG']//filtered_data['Fs_IBI'], axis=-1)
            data_cg_theta_amp = decimate(data_cg_theta_amp, filtered_data['Fs_EEG']//filtered_data['Fs_IBI'], axis=-1)

            # zscore the theta amplitude signals
            data_ch_theta_amp = zscore(data_ch_theta_amp) # normalize the theta amplitude signals
            data_cg_theta_amp = zscore(data_cg_theta_amp) # normalize the theta amplitude signals
            
            # Now we have the theta amplitude signals for both child and caregiver, let's get the IBI signals for the selected event
            IBI_ch_interp, IBI_cg_interp, t_IBI = get_IBI_signal_from_ECG_for_selected_event(filtered_data, events, event, plot=False, label='IBI signals for ' + event )
            # zscore the IBI signals
            IBI_ch_interp = zscore(IBI_ch_interp) # normalize the IBI   signals
            IBI_cg_interp = zscore(IBI_cg_interp) # normalize the IBI   signals
            
            # construct a numpy data array with the shape (4, N_samples), it will contain HRV and Fz theta amplitude of both child and caregiver
            DTF_data = np.zeros((4, len(data_ch_theta_amp)))
            # fill the data array with the IBI signals and Fz theta amplitude signals
            DTF_data[0, :] = IBI_ch_interp
            DTF_data[1, :] = IBI_cg_interp
            DTF_data[2, :] = data_ch_theta_amp
            DTF_data[3, :] = data_cg_theta_amp
            # plot the data for the child and caregiver IBI signals and Fz theta amplitude signals
            fig, ax = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
            ax[0].set_title(f'Child IBI signal and Fz theta amplitude for {event}')
            ax[1].set_title(f'Caregiver IBI signal and Fz_cg theta amplitude for {event}')
            ax[2].set_title(f'Child Fz theta amplitude for {event}')
            ax[3].set_title(f'Caregiver Fz_cg theta amplitude for {event}')
            ax[0].plot(t_IBI, IBI_ch_interp, 'b', label='Child IBI signal')
            ax[1].plot(t_IBI, IBI_cg_interp, 'b', label='Caregiver IBI signal')
            ax[2].plot(t_IBI, data_ch_theta_amp, 'r', label='Child Fz theta amplitude')
            ax[3].plot(t_IBI, data_cg_theta_amp, 'r', label='Caregiver Fz_cg theta amplitude')
            ax[0].set_ylabel('IBI (ms)')
            ax[1].set_ylabel('IBI (ms)')
            ax[2].set_ylabel('Fz theta amplitude (uV)')
            ax[3].set_ylabel('Fz_cg theta amplitude (uV)')
            ax[3].set_xlabel('Samples (after decimation)')
            ax[0].legend(loc='upper right')
            ax[1].legend(loc='upper right')
            ax[2].legend(loc='upper right')
            ax[3].legend(loc='upper right')
            plt.tight_layout()
            plt.show()
            # estimate DTF for the system consisting of HRV and Fz theta amplitude of both child and caregiver
            crit, p_range, p_opt = mvar_criterion(DTF_data, max_p = 15, crit_type = 'AIC', do_plot = True)
            print('Optimal model order for all channels: p = ', str(p_opt))
            Ar, V = AR_coeff(DTF_data, p_opt)
            H, _ = mvar_H(Ar, f, filtered_data['Fs_IBI'])
            DTF = np.abs(H)**2
            S, f = mvar_spectra(H, V, f  )
            """Let's  plot the results in the table form."""
            mvar_plot(S, DTF,   f, 'From ', 'To ',['Child IBI', 'Caregiver IBI', 'Child Fz theta amp', 'Caregiver Fz_cg theta amp'],  'DTF '+ event ,'sqrt')
            plt.show()
