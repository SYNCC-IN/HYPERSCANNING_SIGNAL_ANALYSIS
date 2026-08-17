"""Power spectral density computation for EEG channel data."""

import numpy as np
from mne.time_frequency import psd_array_multitaper


def compute_psd_multitaper(data, sfreq, fmin, fmax, bandwidth):
    """Compute multitaper power spectral density for multichannel EEG data.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        EEG signal.
    sfreq : float
        Sampling frequency in Hz.
    fmin : float
        Lowest frequency of interest in Hz.
    fmax : float
        Highest frequency of interest in Hz.
    bandwidth : float
        Multitaper frequency smoothing bandwidth in Hz.

    Returns
    -------
    freqs : ndarray, shape (n_freqs,)
        Frequency bins in Hz.
    psd : ndarray, shape (n_channels, n_freqs)
        Power spectral density per channel.
    """
    psd, freqs = psd_array_multitaper(
        data, sfreq=sfreq, fmin=fmin, fmax=fmax, bandwidth=bandwidth, verbose=False,
    )
    return freqs, psd


def average_psd_across_conditions(psd_dict):
    """Average PSD arrays across conditions (e.g. movies).

    Parameters
    ----------
    psd_dict : dict
        Mapping of ``{condition_name: psd_array}``, where each ``psd_array`` has
        the same shape (e.g. n_channels x n_freqs).

    Returns
    -------
    ndarray
        Arithmetic mean PSD across conditions, same shape as each input array.
    """
    if not psd_dict:
        raise ValueError('psd_dict is empty; no conditions to average PSD over.')
    return np.mean(np.stack(list(psd_dict.values()), axis=0), axis=0)
