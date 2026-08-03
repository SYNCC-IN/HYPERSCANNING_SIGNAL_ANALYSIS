import os
import warnings
from pathlib import Path
from typing import Optional, Tuple
import xarray as xr
from .ncdf import load_xarray_from_netcdf, get_export_metadata
import numpy as np
import xarray as xr
from src.plot_utils import plot_xarray_signals

_EEG_10_20_CHANNELS = frozenset({
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'T3',  'C3',  'Cz', 'C4', 'T4',
    'T5',  'P3',  'Pz', 'P4', 'T6',
    'O1',  'O2',
})
_MASTOID_CHANNELS = frozenset({'M1', 'M2'})


def load_eeg_ncdf_as_mne_raw(
    ncdf_path: str,
    montage: Optional[str] = "standard_1020",
    scale_to_volts: float = 1e-6,
    data_xr: Optional[xr.DataArray] = None,
) -> Tuple["mne.io.RawArray", dict]:
    """Load an exported EEG NetCDF file and convert it to MNE RawArray.

    Channel types are assigned as follows:
    - Channels in the standard 10-20 set (Fp1, Fp2, …, O2) → 'eeg'
    - M1, M2 (linked-ears reference) → 'misc'
    This means ``picks='eeg'`` in downstream MNE calls automatically
    excludes the mastoid reference channels.

    Args:
        ncdf_path: Path to exported EEG NetCDF file.
        montage: MNE montage name. If None, montage is not set.
        scale_to_volts: Multiplicative scale to convert values to volts.
        data_xr: Pre-loaded DataArray. If provided, the file is not loaded again.

    Returns:
        mne.io.RawArray: Continuous EEG signal in MNE format.
        original_attrs: Copy of the original DataArray attributes before any modifications.
    """
    try:
        import mne
    except ImportError as exc:
        raise ImportError("mne is required for EEG quality analysis.") from exc

    import numpy as np

    if data_xr is None:
        data_xr = load_xarray_from_netcdf(ncdf_path)
    original_attrs = data_xr.attrs.copy()

    if not isinstance(data_xr, xr.DataArray):
        raise TypeError(f"Expected xarray.DataArray in '{ncdf_path}', got {type(data_xr)}")

    if "time" not in data_xr.dims or "channel" not in data_xr.dims:
        raise ValueError(
            f"Expected DataArray with 'time' and 'channel' dims, got: {data_xr.dims}"
        )

    data_t = data_xr.transpose("channel", "time")
    ch_names = [str(ch) for ch in data_t.coords["channel"].values]
    data_values = np.asarray(data_t.values, dtype=float) * float(scale_to_volts)

    sfreq_attr = data_xr.attrs.get("sampling_freq")
    if sfreq_attr is None or (isinstance(sfreq_attr, str) and not sfreq_attr.strip()):
        raise ValueError("Sampling frequency could not be inferred from 'sampling_freq' attribute.")
    else:
        sfreq = float(sfreq_attr)       


    # Assign initial channel types: eeg for known 10-20 channels, misc for
    # mastoids, eeg as fallback for anything unrecognised (safe default).
    ch_types = [
        'misc' if ch in _MASTOID_CHANNELS else 'eeg'
        for ch in ch_names
    ]

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data_values, info, verbose=False)
    # get the monatege and interpolation info from the metadata if available
    metadata = get_export_metadata(data_xr)
    if 'interpolation' in metadata:
        interpolation = str(metadata['interpolation'])
    else:
        interpolation = ""
    if 'references' in metadata:
        references = str(metadata['references'])
    else:
        references = ""
    raw.info['temp'] = f"; {interpolation}; {references}"

    # Set montage after channel types so that MNE does not warn about
    # missing positions for misc (mastoid) channels.
    if montage:
        try:
            raw.set_montage(montage, on_missing="ignore")
        except ValueError as exc:
            warnings.warn(
                f"Could not set montage '{montage}': {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    return raw, original_attrs

def load_eeg_signals(
    ncdf_path: str,
    channel_subset: Optional[list] = None,
    low_cutoff_hz: Optional[float] = None,
    high_cutoff_hz: Optional[float] = None,
):
    """Load an EEG NetCDF file, optionally filter it, trim the time margin,
    drop M1/M2 mastoid channels, and return a normalised signal array.

    Filtering order (zero-phase ``filtfilt``):

    1. 4th-order Butterworth high-pass at ``low_cutoff_hz`` (if given).
    2. 4th-order Butterworth low-pass at ``high_cutoff_hz`` (if given).
    3. 50 Hz IIR notch (Q = 15) when fs > 100 Hz.

    Filtering is applied to the full signal *before* the time-margin is
    trimmed.  Signals are z-score normalised per channel after trimming.

    Args:
        ncdf_path: Path to the exported EEG NetCDF file.
        channel_subset: Optional list of channel names to keep.  Channels not
            present in the file are silently skipped.
        low_cutoff_hz: High-pass cutoff frequency in Hz.
        high_cutoff_hz: Low-pass cutoff frequency in Hz.

    Returns:
        tuple:
            - **signals** (*np.ndarray*, shape ``(n_chan, n_samp)``): z-scored EEG.
            - **channel_names** (*list[str]*): ordered channel labels.
            - **fs** (*float*): sampling frequency in Hz.
            - **time_s** (*np.ndarray*, shape ``(n_samp,)``): time axis in seconds
              (0 = event start).
            - **event_duration_s** (*float*): duration of the event window in seconds.
    """
    import numpy as np
    from scipy.signal import butter, filtfilt, iirnotch

    da = xr.open_dataarray(ncdf_path)
    fs = float(da.attrs.get("sampling_freq", da.attrs.get("sampling_frequency_Hz", 128.0)))

    if low_cutoff_hz is not None or high_cutoff_hz is not None:
        if "time" not in da.dims or "channel" not in da.dims:
            raise ValueError(
                f"Expected 'time' and 'channel' dimensions in {ncdf_path}, got {da.dims}"
            )

        data_tc = da.transpose("time", "channel").values.astype(np.float64)
        nyquist = fs / 2.0

        if low_cutoff_hz is not None:
            wn_hp = float(low_cutoff_hz) / nyquist
            if not 0.0 < wn_hp < 1.0:
                raise ValueError(
                    f"Invalid low_cutoff_hz={low_cutoff_hz}. "
                    f"Must satisfy 0 < cutoff < {nyquist:.3f} Hz."
                )
            b_hp, a_hp = butter(4, wn_hp, btype="highpass")
            data_tc = filtfilt(b_hp, a_hp, data_tc, axis=0)

        if high_cutoff_hz is not None:
            wn_lp = float(high_cutoff_hz) / nyquist
            if not 0.0 < wn_lp < 1.0:
                raise ValueError(
                    f"Invalid high_cutoff_hz={high_cutoff_hz}. "
                    f"Must satisfy 0 < cutoff < {nyquist:.3f} Hz."
                )
            b_lp, a_lp = butter(4, wn_lp, btype="lowpass")
            data_tc = filtfilt(b_lp, a_lp, data_tc, axis=0)

        notch_freq = 50.0
        if notch_freq < nyquist:
            b_n, a_n = iirnotch(notch_freq, Q=15, fs=fs)
            data_tc = filtfilt(b_n, a_n, data_tc, axis=0)

        da_proc = xr.DataArray(
            data_tc,
            dims=("time", "channel"),
            coords={
                "time": da.coords["time"].values,
                "channel": da.coords["channel"].values,
            },
            attrs=da.attrs,
        )

    t = da_proc.coords["time"].values
    event_duration_s = float(da_proc.attrs.get("event_duration", t[-1]))
    mask = (t >= 0.0) & (t <= event_duration_s)
    da_trimmed = da_proc.isel(time=mask)

    drop_ch = [ch for ch in ["M1", "M2"] if ch in da_trimmed.coords["channel"].values]
    if drop_ch:
        da_trimmed = da_trimmed.drop_sel(channel=drop_ch)

    if channel_subset is not None:
        available = list(da_trimmed.coords["channel"].values)
        sel = [ch for ch in channel_subset if ch in available]
        if not sel:
            raise ValueError(
                f"None of the requested channels {channel_subset} found in "
                f"{ncdf_path}. Available: {available}"
            )
        da_trimmed = da_trimmed.sel(channel=sel)

    signals = da_trimmed.values.T.astype(np.float64)  # (n_chan, n_samp)
    stds = np.std(signals, axis=1, keepdims=True)
    stds[stds == 0] = 1.0
    signals = (signals - np.mean(signals, axis=1, keepdims=True)) / stds

    channel_names = list(da_trimmed.coords["channel"].values)
    time_s = da_trimmed.coords["time"].values.astype(np.float64)
    da.close()
    return signals, channel_names, fs, time_s, event_duration_s


def run_eeg_autoreject_quality_report(
    ncdf_path: str,
    epoch_duration_s: float = 2.0,
    n_interpolate: tuple[int, ...] = (1, 2, 4),
    cv: int = 5,
    random_state: int = 42,
    n_jobs: int = -1,
    montage: Optional[str] = "standard_1020",
    scale_to_volts: float = 1e-6,
    verbose: bool = True,
) -> dict:
    """Create AutoReject quality report for EEG exported to NetCDF.

    Steps:
        1. Load NetCDF and convert to MNE Raw.
        2. Split signal into fixed-length epochs.
        3. Fit AutoReject and collect reject log.
        4. Build tabular summaries and visualization.

    Returns:
        dict with keys:
            - raw
            - epochs
            - autoreject
            - reject_log
            - epoch_summary
            - channel_summary
            - global_summary
            - figure
            - axis
    """
    try:
        import mne
    except ImportError as exc:
        raise ImportError("mne is required for EEG quality analysis.") from exc

    try:
        from autoreject import AutoReject  # type: ignore[reportMissingImports]
    except ImportError as exc:
        raise ImportError(
            "autoreject is required for quality reporting. Install it with: pip install autoreject"
        ) from exc

    import numpy as np
    import pandas as pd

    # Load NCDF once – reuse the DataArray for both metadata extraction and MNE conversion.
    _data_xr_meta = load_xarray_from_netcdf(ncdf_path)
    time_margin_s = float(_data_xr_meta.attrs.get("time_margin_s", 0.0))

    # Sanitize event_duration: prefer explicit event metadata, then fall back to
    # task-level metadata when exporting chunked data.
    _raw_event_duration = _data_xr_meta.attrs.get("event_duration")
    if _raw_event_duration is None:
        _raw_event_duration = _data_xr_meta.attrs.get("task_duration")

    try:
        event_duration: Optional[float] = float(_raw_event_duration)
    except (TypeError, ValueError):
        event_duration = None
    else:
        if not np.isfinite(event_duration):
            event_duration = None
    # First value of the time coordinate is the pre-event start (e.g. -10 s when margin=10).
    time_offset = float(np.asarray(_data_xr_meta.coords["time"].values)[0])

    raw,_ = load_eeg_ncdf_as_mne_raw(
        ncdf_path=ncdf_path,
        montage=montage,
        scale_to_volts=scale_to_volts,
        data_xr=_data_xr_meta,
    )
    del _data_xr_meta

    epochs = mne.make_fixed_length_epochs(
        raw,
        duration=float(epoch_duration_s),
        preload=True,
        verbose=verbose,
    )

    if len(epochs) == 0:
        raise ValueError("No epochs created. Check signal length and epoch_duration_s.")

    ar = AutoReject(
        n_interpolate=list(n_interpolate),
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    ar.fit(epochs)

    _, reject_log = ar.transform(epochs, return_log=True)
    labels = np.asarray(reject_log.labels)

    if hasattr(reject_log, "bad_epochs"):
        bad_epochs = np.asarray(reject_log.bad_epochs, dtype=bool)
    else:
        bad_epochs = np.any(labels == 2, axis=1)

    # Epoch times in MNE's 0-based frame; shift to actual NCDF time coordinate.
    _epoch_starts_mne = (epochs.events[:, 0] - raw.first_samp) / raw.info["sfreq"]
    epoch_starts_actual = _epoch_starts_mne + time_offset
    epoch_ends_actual = epoch_starts_actual + float(epoch_duration_s)

    # An epoch is "in margin" when it lies entirely outside the event window.
    epoch_in_margin = [
        (end <= 0.0 or (event_duration is not None and start >= event_duration))
        for start, end in zip(epoch_starts_actual, epoch_ends_actual)
    ]

    epoch_summary = pd.DataFrame(
        {
            "epoch_idx": np.arange(len(epochs), dtype=int),
            "start_s": epoch_starts_actual,
            "end_s": epoch_ends_actual,
            "interpolated_channels": (labels == 1).sum(axis=1).astype(int),
            "rejected": bad_epochs,
            "in_margin": epoch_in_margin,
        }
    )

    n_epochs = len(epochs)
    channel_summary = pd.DataFrame(
        {
            "channel": list(epochs.ch_names),
            "interpolated_epochs": (labels == 1).sum(axis=0).astype(int),
            "bad_labels": (labels == 2).sum(axis=0).astype(int),
        }
    )
    channel_summary["interpolated_pct"] = 100.0 * channel_summary["interpolated_epochs"] / n_epochs
    channel_summary["bad_labels_pct"] = 100.0 * channel_summary["bad_labels"] / n_epochs
    channel_summary = channel_summary.sort_values(
        ["bad_labels", "interpolated_epochs", "channel"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    global_summary = {
        "ncdf_path": ncdf_path,
        "n_channels": int(len(epochs.ch_names)),
        "n_epochs": int(n_epochs),
        "epoch_duration_s": float(epoch_duration_s),
        "rejected_epochs": int(bad_epochs.sum()),
        "rejected_epochs_pct": float(100.0 * bad_epochs.mean()),
        "total_interpolations": int((labels == 1).sum()),
    }

    # Rejected windows outside margins only — margin artifacts are expected and uninformative.
    rejected_windows = epoch_summary.loc[
        epoch_summary["rejected"] & ~epoch_summary["in_margin"],
        ["start_s", "end_s"],
    ].reset_index(drop=True)

    # Buduj DataArray z poprawną osią czasu (NCDF coordinate, nie 0-based MNE)
    eeg_picks   = mne.pick_types(raw.info, eeg=True)
    eeg_data    = raw.get_data(picks=eeg_picks) * 1e6        # V → μV, (n_ch, n_times)
    eeg_times   = raw.times + time_offset                    # 0-based → NCDF coordinate
    eeg_channels = [raw.ch_names[i] for i in eeg_picks]

    eeg_xr = xr.DataArray(
        data=eeg_data.T,                                     # (n_times, n_ch)
        dims=['time', 'channel'],
        coords={'time': eeg_times, 'channel': eeg_channels},
    )

    # Przekształć DataFrame odrzuconych okien → lista regions
    regions = [
        {'span': (float(row['start_s']), float(row['end_s'])),
        'name':  'rejected',
        'color': '#d62728',
        'alpha': 0.18}
        for _, row in rejected_windows.iterrows()
    ]

    note = str(raw.info.get('temp', ''))
    fig, ax = plot_xarray_signals(
        eeg_xr,
        regions=regions,
        event_duration=event_duration,
        time_margin_s=time_margin_s,
        title=f"AutoReject suggested rejections. {note}",
    )

    return {
        "raw": raw,
        "epochs": epochs,
        "autoreject": ar,
        "reject_log": reject_log,
        "epoch_summary": epoch_summary,
        "channel_summary": channel_summary,
        "global_summary": global_summary,
        "figure": fig,
        "axis": ax,
    }

def check_exported_data_quality(dyad: str, modality: str, member: str, task: str, export_folder: str) -> bool:
    """
    Check the quality of exported data for a given dyad.

    Args:
        dyad: Dyad ID to check.
        modality: The modality to check (e.g., 'EEG', 'ECG').
        export_folder: Path to the folder containing exported data.
    Returns:
        saves figure and returns True if the figures were saved
        
    """
    # Read the exported data for the given dyad and modality
    if member == 'ch':
        nc_path = os.path.join(export_folder, modality, dyad, 'child', f"{dyad}_{modality}_{member}_{task}.nc")
    elif member == 'cg':
        nc_path = os.path.join(export_folder, modality, dyad, 'caregiver', f"{dyad}_{modality}_{member}_{task}.nc")
    else:
        raise ValueError(f"Invalid member: {member}. Must be 'ch' or 'cg'.")
    
    if modality == 'EEG':
        # Run AutoReject quality report
        report = run_eeg_autoreject_quality_report(ncdf_path=str(nc_path))
        fig = report['figure']
        ax = report['axis']
        # create a folder for the figures if it doesn't exist
        fig_folder = os.path.join(export_folder, modality, 'Quality_reports')
        if not os.path.exists(fig_folder):
            os.makedirs(fig_folder)
        # Save the figure to a file
        fig_filename = os.path.join(fig_folder, f"{dyad}_{modality}_{member}_{task}_quality_report.png")
        fig.savefig(fig_filename)
        print(f"Saved EEG quality report figure to {fig_filename}")

    return True