import os
from contextlib import redirect_stdout
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, hilbert, resample_poly
from scipy.fft import next_fast_len

try:
    from .mtmvar import (
        full_freq_dtf,
        multivariate_spectra,
        mvar_plot,
        mvar_criterion,
    )
except ImportError:  # pragma: no cover - fallback for direct script execution
    from src.mtmvar import (
        full_freq_dtf,
        multivariate_spectra,
        mvar_plot,
        mvar_criterion,
    )


class EEG_IBI_FFDTF_Pipeline:
    def __init__(self, cleaned_signals_folder: Path, output_ffDTF_folder: Path, target_events: list, smoke_test: bool = False, smoke_dyads_n: int = 1,
                 left_frontal_eeg_channel: str = "F3", right_frontal_eeg_channel: str = "F4", fs_downsampled:float = 8.0,
                 n_windows:int = 3, window_size:int = None, ar_p:int = 5,
                 plot_global_enabled: bool = True, save_global_enabled: bool = True, plot_windowed_enabled: bool = True, save_windowed_enabled: bool = True,):
        """
        Pipeline for dyadic EEG-IBI analysis using full frequency Direct Transfer Function (ffDTF).

        This class implements a full processing workflow for dyadic recordings,
        including EEG preprocessing, frontal alpha asymmetry (FAA) computation,
        IBI preprocessing, signal synchronization, resampling, and preparation
        for multivariate connectivity analysis using ffDTF.

        During initialization, the pipeline automatically scans the provided
        directory and loads EEG and IBI .nc files for all valid dyads and target events.

        Parameters
        ----------
        cleaned_signals_folder : Path
            Directory containing preprocessed EEG and IBI .nc files.

        output_ffDTF_folder : Path
            Directory where all computed ffDTF results and intermediate outputs
            (NPZ files and optional figures) will be saved.

        target_events : list
            List of experimental conditions/events to include in the analysis.

        smoke_test : bool, default=True
            If True, processes only a subset of dyads for quick validation.

        smoke_dyads_n : int, default=1
            Number of dyads used in smoke test mode.

        left_frontal_eeg_channel : str, default="F3"
            Name of the left frontal EEG channel used to compute Frontal Alpha Asymmetry (FAA).

        right_frontal_eeg_channel : str, default="F4"
            Name of the right frontal EEG channel used to compute Frontal Alpha Asymmetry (FAA).

        fs_downsampled : float, default=8.0
            Target sampling frequency used for signal resampling and alignment
            before ffDTF estimation.

        plot_global_enabled : bool, default=True
            Enables plotting of global ffDTF connectivity results.

        save_global_enabled : bool, default=True
            Enables saving of global ffDTF figures to disk.

        plot_windowed_enabled : bool, default=True
            Enables plotting of windowed (dynamic) ffDTF connectivity results.

        save_windowed_enabled : bool, default=True
            Enables saving of windowed ffDTF figures to disk.

        Notes
        -----
        The pipeline assumes temporally aligned EEG-IBI recordings per dyad and event.
        All connectivity analyses are performed on synchronized, downsampled signals.
        """
         
        self.cleaned_signals_folder = Path(cleaned_signals_folder)
        self.output_ffDTF_folder = Path(output_ffDTF_folder)
        self.target_events = target_events
        self.smoke_test = smoke_test
        self.smoke_dyads_n = smoke_dyads_n
        self.left_chan = left_frontal_eeg_channel
        self.right_chan = right_frontal_eeg_channel
        self.fs_ds = float(fs_downsampled)

        self.freq_min = 1.0                                 # Minimum frequency for analysis.
        self.freq_step = 0.1                                # Frequency resolution.
        nyquist_limit = self.fs_ds / 2.0
        self.freq_max = nyquist_limit - self.freq_step      # Dynamic maximum frequency

        self.n_windows = n_windows
        self.window_size = window_size
        self.ar_p = ar_p
        self.plot_global_enabled = plot_global_enabled
        self.save_global_enabled = save_global_enabled
        self.plot_windowed_enabled = plot_windowed_enabled
        self.save_windowed_enabled = save_windowed_enabled
        
        # Tables for storing file paths and dyads to process
        self.eeg_files = []
        self.ibi_files = []
        self.dyads_to_process = []
        
        # Automatically prepare file lists after initialization
        self._prepare_file_lists()


    def _prepare_file_lists(self):
        """
        Prepare EEG and IBI file lists for selected events and dyads.
        """
        signal_types = ["EEG", "IBI"]
        selected_files = {"EEG": [], "IBI": []}
        
        # A set automatically ensures that dyads are not duplicated
        all_dyads_set = set()

        def _extract_dyad(p):
            """Helper to extract dyad ID from filename."""
            parts = p.stem.split('_')
            return f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else p.stem

        for sig_type in signal_types:
            folder = self.cleaned_signals_folder / sig_type
            
            all_files = sorted([
                p for p in folder.rglob("*.nc")
                if sig_type in p.name and any(ev in p.name for ev in self.target_events)
            ])
            
            if not all_files:
                raise FileNotFoundError(f"No {sig_type} files found for events {self.target_events} under: {folder}")

            # Assign files only for the given signal type
            selected_files[sig_type] = all_files

            # Extract dyad names from files and add them to the set
            for p in all_files:
                dyad_id = _extract_dyad(p)
                all_dyads_set.add(dyad_id)

        # Sort unique dyads alphabetically
        all_dyads = sorted(list(all_dyads_set))
        
        # Smoke test condition
        if self.smoke_test:
            self.dyads_to_process = all_dyads[:self.smoke_dyads_n]
        else:
            self.dyads_to_process = all_dyads

        # Store total counts before filtering for logging
        total_eeg = len(selected_files["EEG"])
        total_ibi = len(selected_files["IBI"])

        # Filter the file lists to keep ONLY the files for the dyads we are actually processing
        self.eeg_files = [p for p in selected_files["EEG"] if _extract_dyad(p) in self.dyads_to_process]
        self.ibi_files = [p for p in selected_files["IBI"] if _extract_dyad(p) in self.dyads_to_process]

        # Prepare console output
        mode = f"SMOKE TEST (first {self.smoke_dyads_n} dyads)" if self.smoke_test else "FULL ANALYSIS"
        print(f"\n=== Initialization Complete: {mode} ===")
        print(f" [INFO] Target events : {self.target_events}")
        # Dodałem użyte/wszystkie również do diad dla pełnej spójności
        print(f" [INFO] Dyads loaded  : {len(self.dyads_to_process)}/{len(all_dyads)} ({', '.join(self.dyads_to_process)})")
        print(f" [OK]   EEG files     : {len(self.eeg_files)}/{total_eeg} ready")
        print(f" [OK]   IBI files     : {len(self.ibi_files)}/{total_ibi} ready\n")


    def _find_file(self, file_list, dyad, film, role):
        """
        Find a single file matching dyad, film, and role metadata.
        Returns (file, True) if found, (None, False) if missing, 
        or raises ValueError if multiple files are found.
        """
        find_flag = True
        found = [f for f in file_list if dyad in f.name and f"_{film}" in f.name and f"_{role}_" in f.name]
        
        if not found:
            find_flag = False
            return None, find_flag
            
        if len(found) > 1:
            raise ValueError(f"Found multiple files for dyad: {dyad}, film: {film}, role: {role} -> {found}")
            
        return found[0], find_flag
    

    def _load_eeg_and_ibi(self, eeg_file, ibi_file, role):
        """
        Load EEG and IBI data from .nc files.

        Data are loaded and converted to NumPy arrays for processing.

        Parameters
        ----------
        eeg_file : Path
            Path to EEG .nc file.

        ibi_file : Path
            Path to IBI .nc file.

        role : str
            Subject identifier (e.g., 'Child' or 'Care Giver').

        Returns
        -------
        time_s : ndarray
            Time vector of the recording.

        eeg_data : ndarray
            EEG signal array.
            Shape: (n_channels, n_samples)

        fs_eeg : float
            EEG sampling frequency [Hz].

        channel_names : list of str
            List of EEG channel labels.

        ibi_data : ndarray
            Inter-beat interval signal.
            Shape: (n_samples,) or (n_beats,)

        fs_ibi : float
            IBI sampling frequency [Hz].

        event_duration_s : float
            Duration of the recorded event in seconds.
        """
        with xr.open_dataarray(eeg_file) as da_eeg:
            eeg_data = da_eeg.values.T.copy()
            time_s = da_eeg.coords['time'].values.copy()
            channel_names = da_eeg.coords['channel'].values.tolist() 
            event_duration_s = float(da_eeg.attrs['event_duration_s'])
            raw_fs_eeg = da_eeg.attrs.get('sampling_freq') or da_eeg.attrs.get('sfreq')
            
            if raw_fs_eeg is not None:
                fs_eeg = float(raw_fs_eeg)
            else:
                print(f" [WARN] {role}: Missing EEG sampling freq. Defaulting to 128.0 Hz")
                fs_eeg = 128.0 

        with xr.open_dataarray(ibi_file) as da_ibi:
            ibi_data = da_ibi.values.T.copy()
            
            raw_fs_ibi = da_ibi.attrs.get('sampling_freq') or da_ibi.attrs.get('sfreq')
            
            if raw_fs_ibi is not None:
                fs_ibi = float(raw_fs_ibi)
            else:
                fs_ibi = fs_eeg
                print(f" [INFO] {role}: Missing IBI sampling freq. Copying from EEG ({fs_ibi} Hz)")
        
        return time_s, eeg_data, fs_eeg, channel_names, ibi_data, fs_ibi, event_duration_s


    def _alpha_bandpass_filter(self, data, fs, lowcut=8, highcut=12, order=4, axis=-1):
        """
        Apply Butterworth bandpass filter for alpha band extraction.

        The function isolates the alpha frequency band (default 8-12 Hz)
        using a zero-phase Butterworth filter (sosfiltfilt in implementation).

        Parameters
        ----------
        data : ndarray
            Input EEG signal.
            Shape: (n_channels, n_samples) or (n_samples,)

        fs : float
            Sampling frequency of the signal [Hz].

        lowcut : float, default=8
            Lower cutoff frequency [Hz].

        highcut : float, default=12
            Upper cutoff frequency [Hz].

        order : int, default=4
            Order of the Butterworth filter.

        axis : int, default=-1
            Axis along which the filtering is applied.

        Returns
        -------
        filtered_data : ndarray
            Bandpass-filtered signal in alpha range (8-12 Hz).
            Same shape as input.
        """
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        
        sos = butter(order, [low, high], btype='band', output='sos')
        filtered_data = sosfiltfilt(sos, data, axis=axis)
        return filtered_data
    

    def _compute_asymmetry(self, filtered_eeg, channel_names, metric='amp'):
        """
        Compute Frontal Alpha Asymmetry (FAA) using Hilbert transform.

        Parameters
        ----------
        filtered_eeg : ndarray
            EEG signal after bandpass filtering.
            Shape: (n_channels, n_samples)

        channel_names : list of str
            List of EEG channel names corresponding to rows in filtered_eeg.

        metric : str, default='amp'
            Type of Hilbert envelope used:
            - 'amp'   -> amplitude envelope
            - 'power' -> squared amplitude (power)

        Returns
        -------
        faa : ndarray
            Frontal Alpha Asymmetry time series:
            log(right) - log(left)
            Shape: (n_samples,)
        """
        try:
            left_idx = channel_names.index(self.left_chan)
            right_idx = channel_names.index(self.right_chan)
        except ValueError as e:
            raise ValueError(f"Channels {self.left_chan} or {self.right_chan} not found: {e}")

        eeg_left = filtered_eeg[left_idx, :]
        eeg_right = filtered_eeg[right_idx, :]

        orig_len = len(eeg_left)
        fast_len = next_fast_len(orig_len)

        env_left = np.abs(hilbert(eeg_left, N=fast_len)[:orig_len])
        env_right = np.abs(hilbert(eeg_right, N=fast_len)[:orig_len])

        if metric == 'power':
            left = env_left ** 2
            right = env_right ** 2
        elif metric == 'amp':
            left = env_left
            right = env_right
        else:
            raise ValueError("metric must be 'power' or 'amp'")

        faa = np.log(right + 1e-12) - np.log(left + 1e-12)

        return faa
    

    def _downsample_signal(self, signal, fs=128, fs_new=8):
        """
        Anti-aliased downsampling of a 1D signal.

        Uses scipy's resample_poly, which automatically applies an optimal anti-aliasing FIR filter

        Parameters
        ----------
        signal : ndarray
            Input 1D time series signal.

        fs : float, default=128
            Original sampling frequency [Hz].

        fs_new : float, default=8
            Target sampling frequency [Hz].
            Must be lower than fs.

        Returns
        -------
        signal_ds : ndarray
            Downsampled signal.
            Shape: (len(signal) * fs_new / fs,)
        """

        if fs_new >= fs:
            raise ValueError("fs_new must be lower than fs")

        ratio = fs / fs_new
        if not np.isclose(ratio, round(ratio)):
            raise ValueError("fs must be divisible by fs_new")

        # Calculate the decimation factor (must be an integer)
        down = int(round(ratio))

        # resample_poly automatically applies an anti-aliasing filter by default
        signal_ds = resample_poly(signal, up=1, down=down)

        return signal_ds
    

    def _crop_signal(self, signal, fs, drop_front_sec=10, keep_duration_sec=60):
        """
        Crops the signal to a specific duration, dropping the initial segment.

        This method removes the first `drop_front_sec` seconds and extracts exactly the next 
        `keep_duration_sec` seconds. The rest of the signal is discarded.

        Parameters
        ----------
        signal : np.ndarray
            Input signal. Time must be the last dimension 
        fs : float
            Current sampling frequency of the signal [Hz].
        drop_front_sec : float, default=10
            Seconds to drop from the beginning of the signal.
        keep_duration_sec : float, default=60
            Seconds to keep after the dropped front segment.

        Returns
        -------
        np.ndarray
            Cropped signal.
        """
        
        required_sec = drop_front_sec + keep_duration_sec
        total_samples = signal.shape[-1]
        total_sec = total_samples / fs

        if total_sec >= required_sec:
            start_idx = int(drop_front_sec * fs)
            end_idx = int(required_sec * fs)
            
            cropped_signal = signal[..., start_idx:end_idx]
            return cropped_signal
        else:
            raise ValueError(
                f"Cropping failed: The signal is too short. "
                f"It needs to be at least {required_sec} seconds long, "
                f"but the provided signal is only {total_sec:.2f} seconds."
            )
    

    def _create_windows(self, signals, n_windows=3, window_size=None):
        """
        Creates n_windows that evenly span the entire length of the signal.

        If window_size is None, the signal is divided into n_windows strictly
        non-overlapping consecutive segments. If window_size is provided, the 
        windows will overlap evenly to perfectly cover the entire signal from 
        the first to the last sample.

        Parameters
        ----------
        signals : np.ndarray
            Shape (channels, time). The input signal to be windowed.
        n_windows : int, default=3
            Exact number of windows to generate.
        window_size : int, optional
            Number of samples per window. If None, it is calculated automatically
            to avoid overlap. Must be large enough so that n_windows cover the 
            entire signal.

        Returns
        -------
        list of np.ndarray
            List containing n_windows segments of the signal.
        """
        T = signals.shape[1]

        if window_size is None:
            # Auto-calculate exact window size for zero overlap
            if T % n_windows != 0:
                raise ValueError(
                    f"Cannot evenly divide signal of length {T} into {n_windows} "
                    f"non-overlapping windows. Provide a specific window_size."
                )
            window_size = T // n_windows
        else:
            # Calculate the mathematical ceiling for minimum required size
            min_required_size = (T + n_windows - 1) // n_windows
            
            if window_size < min_required_size:
                raise ValueError(
                    f"window_size={window_size} is too short. To cover {T} samples "
                    f"with {n_windows} windows without leaving gaps, the minimum "
                    f"window_size is {min_required_size}."
                )
            
            if window_size > T:
                raise ValueError(f"window_size ({window_size}) cannot exceed signal length ({T}).")

        # The last window must end exactly at the end of the signal
        max_start = T - window_size
        
        # Prevent generating identical, fully overlapping windows
        if max_start < n_windows - 1 and n_windows > 1:
            raise ValueError(
                f"window_size={window_size} is too large to generate {n_windows} "
                f"distinct windows. Decrease window_size or n_windows."
            )

        if n_windows == 1:
            positions = [0]
        else:
            positions = np.linspace(0, max_start, n_windows, dtype=int)

        # Generate the list of windows based on calculated start positions
        windows = [signals[:, start:start + window_size] for start in positions]

        return windows


    def _compute_ffDTF(
            self,
            dyad,
            signals,
            chan_names,
            fs,
            max_model_order=20,
            crit_type="AIC",
            plot=True,
            save_plot=False,
            save_path=None,
            fig_name=None):
        """
        Compute multivariate frequency-domain connectivity (ffDTF) and spectra
        for a dyadic signal segment.

        This function estimates MVAR-based frequency-domain connectivity
        and optionally generates visualization and/or saves figures.

        Parameters
        ----------
        dyad : str
            Identifier of dyad (used in plot titles and file naming).

        signals : np.ndarray
            Multivariate signal array of shape (channels, time).

        chan_names : list of str
            Names of input channels.

        fs : float
            Sampling frequency of the signal.

        max_model_order : int, default=20
            Maximum MVAR model order considered.

        crit_type : str, default="AIC"
            Criterion used for model order selection.

        plot : bool, default=True
            If True, generates a figure of ffDTF and spectra.

        save_plot : bool, default=False
            If True, saves figure to disk (requires save_path).

        save_path : Path or None
            Directory where figures will be stored if save_plot=True.

        fig_name : str or None
            Optional filename for saved figure.

        Returns
        -------
        ff_dtf : np.ndarray
            Estimated frequency-domain directed transfer function.

        spectra : np.ndarray
            Multivariate spectra estimates.

        p_opt : int
            Selected optimal MVAR model order.
        """

        freqs = np.arange(self.freq_min, self.freq_max + self.freq_step, self.freq_step)

        if self.ar_p is None:
            crit, model_order_range, p_opt = mvar_criterion(signals, max_model_order, crit_type, plot=False)
        else:
            p_opt = self.ar_p

        with open(os.devnull, 'w') as f, redirect_stdout(f):
            ff_dtf = full_freq_dtf(
                signals, freqs, fs,
                max_model_order=max_model_order,
                optimal_model_order=p_opt,
                crit_type=crit_type,
            )

            spectra = multivariate_spectra(
                signals, freqs, fs,
                max_model_order=max_model_order,
                optimal_model_order=p_opt,
                crit_type=crit_type,
            )

        if plot or save_plot:
            if fig_name is None:
                fig_name = f"{dyad}_ffDTF.png"

            mvar_plot(
                spectra, ff_dtf, freqs,
                x_label="", y_label="",
                chan_names=chan_names,
                top_title=f"{fig_name[:-4]}",
                scale="linear",
            )

            fig = plt.gcf()
            fig.tight_layout()

            if save_path is not None and save_plot:
                save_path = Path(save_path)
                save_path.mkdir(parents=True, exist_ok=True)

                fig_file = save_path / fig_name
                fig.savefig(fig_file, dpi=300, bbox_inches="tight")

            if plot:
                plt.show()
                
            # close all for preventing memory leakage
            plt.close('all') 

        return ff_dtf, spectra, p_opt


    def _save_single_result(self, dyad, film, result):
        """
        Saves a single dyad-film ffDTF result immediately after computation.
        """

        dyad_dir = Path(self.output_ffDTF_folder) / dyad
        dyad_dir.mkdir(parents=True, exist_ok=True)

        file_path = dyad_dir / f"{dyad}_{film}_ffDTF.npz"

        np.savez_compressed(
            file_path,
            ff_dtf_global=result["mvar"]["ff_dtf_global"],
            spectra_global=result["mvar"]["spectra_global"],
            ff_dtf_windowed=np.array(result["mvar"]["ff_dtf_windowed"]),
            spectra_windowed=np.array(result["mvar"]["spectra_windowed"]),
            p_opt_g=result["mvar"]["p_opt_g"],
            p_opt_w=result["mvar"]["p_opt_w"],
            meta=json.dumps(result["meta"])
        )

        print(f"[SAVED] {dyad} | {film} --> {file_path}\n")


    def run_pipeline(self):
        if not self.dyads_to_process:
            raise RuntimeError("No loaded dyads. Check the files.")

        for dyad in self.dyads_to_process:
            for film in self.target_events:
                print(f"--- Processing dyad: {dyad} | Film: {film} ---")

                eeg_ch, find_flag_eeg_ch = self._find_file(self.eeg_files, dyad, film, "ch")
                ibi_ch, find_flag_ibi_ch = self._find_file(self.ibi_files, dyad, film, "ch")
                eeg_cg, find_flag_eeg_cg = self._find_file(self.eeg_files, dyad, film, "cg")
                ibi_cg, find_flag_ibi_cg = self._find_file(self.ibi_files, dyad, film, "cg")

                # Check missing files for dyad and film
                missing_files = []
                if not find_flag_eeg_ch: missing_files.append("EEG (ch)")
                if not find_flag_ibi_ch: missing_files.append("IBI (ch)")
                if not find_flag_eeg_cg: missing_files.append("EEG (cg)")
                if not find_flag_ibi_cg: missing_files.append("IBI (cg)")

                # If missing_files list isn't empty, display missing files and jump to next film
                if missing_files:
                    print(f" [SKIP] Missing files: {', '.join(missing_files)} -> Skipping {film}")
                    continue

                print(f" [OK] Loaded EEG (ch) : {eeg_ch.name}")
                print(f" [OK] Loaded IBI (ch) : {ibi_ch.name}")
                print(f" [OK] Loaded EEG (cg) : {eeg_cg.name}")
                print(f" [OK] Loaded IBI (cg) : {ibi_cg.name}")

                _ ,eeg_data_ch, fs_eeg, channel_names, ibi_data_ch, fs_ibi, _ = self._load_eeg_and_ibi(eeg_ch, ibi_ch, role = "Child")
                _ ,eeg_data_cg, fs_eeg, channel_names, ibi_data_cg, fs_ibi, _ = self._load_eeg_and_ibi(eeg_cg, ibi_cg, role = "Care Giver")

                filtered_eeg_ch = self._alpha_bandpass_filter(eeg_data_ch, fs_eeg)
                filtered_eeg_cg = self._alpha_bandpass_filter(eeg_data_cg, fs_eeg)

                faa_ch = self._compute_asymmetry(filtered_eeg_ch, channel_names, metric="amp")
                faa_cg = self._compute_asymmetry(filtered_eeg_cg, channel_names, metric="amp")

                ibi_ch = ibi_data_ch.squeeze()
                ibi_cg = ibi_data_cg.squeeze()

                faa_ch_ds = self._downsample_signal(faa_ch, fs_eeg, self.fs_ds)
                ibi_ch_ds = self._downsample_signal(ibi_ch, fs_ibi, self.fs_ds)
                faa_cg_ds = self._downsample_signal(faa_cg, fs_eeg, self.fs_ds)
                ibi_cg_ds = self._downsample_signal(ibi_cg, fs_ibi, self.fs_ds)


                faa_ch_croped = self._crop_signal(faa_ch_ds, self.fs_ds, drop_front_sec=10, keep_duration_sec=60)
                ibi_ch_croped = self._crop_signal(ibi_ch_ds, self.fs_ds, drop_front_sec=10, keep_duration_sec=60)
                faa_cg_croped = self._crop_signal(faa_cg_ds, self.fs_ds, drop_front_sec=10, keep_duration_sec=60)
                ibi_cg_croped = self._crop_signal(ibi_cg_ds, self.fs_ds, drop_front_sec=10, keep_duration_sec=60)

                signals_to_ffDTF = np.vstack([faa_ch_croped, ibi_ch_croped, faa_cg_croped, ibi_cg_croped])

                # Standardization of multivariate signals (channel-wise z-score normalization)
                # Ensures zero mean and unit variance per channel, improving MVAR estimation stability
                # and preventing scale dominance between EEG-derived FAA and physiological IBI signals
                signals_to_ffDTF = (signals_to_ffDTF - np.mean(signals_to_ffDTF, axis=1, keepdims=True)) / np.std(signals_to_ffDTF, axis=1, keepdims=True)

                chan_names_to_ffDTF = ["faa_ch", "ibi_ch", "faa_cg", "ibi_cg"]

                print(" [OK] Pre-processing complete (Alpha -> FAA -> Downsample -> Crop -> Z-Score)")

                # Checking relation p_opt to fixed self.ar_p
                if self.ar_p is not None:
                    _, _, suggested_p = mvar_criterion(signals_to_ffDTF, 20, "AIC", plot=False)
                    print(f" [INFO] AIC suggested p={suggested_p} for global signal. Forcing fixed p={self.ar_p}.")

                # Windowed ffDTF (dynamic connectivity)
                windowed_signals = self._create_windows(signals_to_ffDTF, self.n_windows, self.window_size)

                ffdtf_results_windowed = []
                spectra_windowed = []
                tab_p_opt_w = []

                window_save_dir = self.output_ffDTF_folder / dyad

                print(f" [INFO] Computing windowed ffDTF ({len(windowed_signals)} windows)...")

                for i, w in enumerate(windowed_signals):
                    ff_dtf_w, spectra_w, p_opt_w  = self._compute_ffDTF(
                        dyad,
                        w,
                        chan_names_to_ffDTF,
                        self.fs_ds,
                        plot=self.plot_windowed_enabled,
                        save_plot=self.save_windowed_enabled,
                        save_path=window_save_dir,
                        fig_name=f"{dyad}_{film}_win{i}_ffDTF.png"
                    )

                    ffdtf_results_windowed.append(ff_dtf_w)
                    spectra_windowed.append(spectra_w)
                    tab_p_opt_w.append(p_opt_w)

                # Global ffDTF (baseline)
                global_save_dir = self.output_ffDTF_folder / dyad
                fig_name_global = f"{dyad}_{film}_ffDTF_global.png"

                print(" [INFO] Computing global ffDTF...")

                ff_dtf_global, spectra_global, p_opt_g  = self._compute_ffDTF(
                    dyad,
                    signals_to_ffDTF,
                    chan_names_to_ffDTF,
                    self.fs_ds,
                    plot=self.plot_global_enabled,
                    save_plot=self.save_global_enabled,
                    save_path=global_save_dir,
                    fig_name=fig_name_global
                )

                # Build result object
                result = {
                    "mvar": {
                        "ff_dtf_global": ff_dtf_global,
                        "spectra_global": spectra_global,
                        "ff_dtf_windowed": ffdtf_results_windowed,
                        "spectra_windowed": spectra_windowed,
                        "p_opt_g": p_opt_g,
                        "p_opt_w": tab_p_opt_w
                    },
                    "meta": {
                        "dyad": dyad,
                        "film": film,
                        "fs": self.fs_ds,
                        "fs_original": fs_eeg,
                        "chan_names": chan_names_to_ffDTF,
                        "faa_chan_names": (self.left_chan, self.right_chan),
                        "windowing": {
                            "n_windows": self.n_windows,
                            "window_size": self.window_size,
                        },
                        "computed_at": datetime.now().isoformat(),
                    }
                }

                # Save result object
                self._save_single_result(dyad, film, result)

                # Force RAM cleanup by removing temporary objects already saved to disk
                del windowed_signals
                del signals_to_ffDTF
                del ffdtf_results_windowed
                del spectra_windowed