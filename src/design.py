"""MVAR design-variable construction for the interbrain ffDTF + HRV pipeline (Stage 2).

Turns the continuous, ROI-selected EEG and IBI signals assembled by
`src.assemble.assemble_dyad` into the four per-dyad-per-film design
variables (`child:ROI`, `cg:ROI`, `child:HRV`, `cg:HRV`) that feed the MVAR
model in later stages. The ROI variables are individualized-band amplitude
envelopes; the HRV variables are the raw (interpolated) IBI signal,
downsampled only -- no band-pass, no Hilbert (see `scripts/stage02_envelopes.py`
for the rationale). Both are computed on the whole continuous `passive_movies`
chunk and segmented to a film window only afterwards, so filter/Hilbert edge
transients fall in the discarded margins/gaps rather than inside the retained
window -- see `scripts/stage02_envelopes.py` for the orchestration built on
top of this module.
"""

import numpy as np
import xarray as xr

try:
    from .envelopes import filter_individual_band, hilbert_envelope, downsample
except ImportError:  # pragma: no cover - fallback for direct script execution
    from src.envelopes import filter_individual_band, hilbert_envelope, downsample

DESIGN_VARIABLES = ["child:ROI", "cg:ROI", "child:HRV", "cg:HRV"]


def roi_band_envelope(roi_signals, sfreq, center_freq, bandwidth, order, target_sfreq, reduction):
    """Continuous downsampled amplitude envelope of an individualized ROI band.

    Parameters
    ----------
    roi_signals : np.ndarray, shape (n_roi_channels, n_times)
        Continuous ROI-selected signal, e.g. `dyad['eeg'][role]['data']` from
        `src.assemble.assemble_dyad`.
    sfreq : float
        Sampling frequency of `roi_signals` in Hz.
    center_freq : float
        Individualized rhythm center frequency in Hz.
    bandwidth : float
        Half-width of the individualized passband in Hz (see
        `src.envelopes.filter_individual_band`).
    order : int
        Butterworth filter order.
    target_sfreq : float
        Desired envelope sampling frequency in Hz.
    reduction : {"average_envelopes", "average_raw"}
        "average_envelopes": filter + Hilbert each ROI channel separately,
        then average the resulting envelopes. "average_raw": average the raw
        ROI channels first, then filter + Hilbert once. The two coincide for
        a single-channel ROI.

    Returns
    -------
    envelope : np.ndarray
        Downsampled instantaneous amplitude envelope.
    env_sfreq : float
        Realized envelope sampling frequency in Hz.
    """
    if reduction == "average_envelopes":
        channel_envelopes = []
        for channel_signal in roi_signals:
            filtered = filter_individual_band(channel_signal, sfreq, center_freq, bandwidth, order)
            channel_envelopes.append(hilbert_envelope(filtered))
        envelope_full = np.mean(channel_envelopes, axis=0)
    elif reduction == "average_raw":
        averaged = roi_signals.mean(axis=0)
        filtered = filter_individual_band(averaged, sfreq, center_freq, bandwidth, order)
        envelope_full = hilbert_envelope(filtered)
    else:
        raise ValueError(f"Unknown reduction {reduction!r}; expected 'average_envelopes' or 'average_raw'")

    return downsample(envelope_full, sfreq, target_sfreq)


def segment_signal(signal, fs, t0, film_start_s, film_end_s):
    """Slice a continuous downsampled signal to one film window.

    Used for all four design variables -- the ROI amplitude envelopes and
    the raw downsampled IBI alike.

    Parameters
    ----------
    signal : np.ndarray, shape (n_times,)
        Continuous downsampled signal.
    fs : float
        Sampling frequency of `signal` in Hz.
    t0 : float
        Time (s) of the source continuous chunk's first sample (i.e. the
        `time[0]` of the signal `signal` was derived from), so the
        downsampled time axis lines up with `film_start_s`/`film_end_s`,
        which are expressed in that same reference (see `coverage.csv`).
    film_start_s : float
        Film window start, in the same time reference as `t0`.
    film_end_s : float
        Film window end, in the same time reference as `t0`.

    Returns
    -------
    segment : np.ndarray
        Signal values with `film_start_s <= t <= film_end_s`.
    time : np.ndarray
        Downsampled time axis for `segment`, in the same reference as `t0`.
    """
    time = t0 + np.arange(signal.size) / fs
    mask = (time >= film_start_s) & (time <= film_end_s)
    return signal[mask], time[mask]


def stack_design_variables(child_roi, cg_roi, child_hrv, cg_hrv, fs, attrs):
    """Stack the four MVAR design variables into one labeled DataArray.

    Parameters
    ----------
    child_roi, cg_roi : np.ndarray, shape (n_times,)
        Already-segmented individualized-band amplitude envelopes.
    child_hrv, cg_hrv : np.ndarray, shape (n_times,)
        Already-segmented raw (interpolated) IBI signal, downsampled only.
    fs : float
        Realized common sampling frequency of all four variables, in Hz.
    attrs : dict
        Written verbatim as the returned DataArray's attrs (e.g. `fs`,
        `roi_label`, `roi_channels`, band parameters, `hrv_signal`, `film`,
        `dyad_id`, `group`, `age_months`, `target_sfreq`, `zscored`,
        `roi_reduction`, `bw_convention`).

    Returns
    -------
    xarray.DataArray, dims ("variable", "time")
        `variable` coordinate fixed to `DESIGN_VARIABLES`
        (`["child:ROI", "cg:ROI", "child:HRV", "cg:HRV"]`).
    """
    lengths = {child_roi.size, cg_roi.size, child_hrv.size, cg_hrv.size}
    if len(lengths) != 1:
        raise ValueError(f"Design variables must share one length, got {lengths}")

    n_times = child_roi.size
    data = np.stack([child_roi, cg_roi, child_hrv, cg_hrv], axis=0)
    time = np.arange(n_times) / fs
    return xr.DataArray(
        data,
        dims=("variable", "time"),
        coords={"variable": DESIGN_VARIABLES, "time": time},
        attrs=attrs,
    )
