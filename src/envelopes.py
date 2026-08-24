"""Instantaneous amplitude envelope utilities for narrow-band signals."""

from fractions import Fraction

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, hilbert, resample_poly, sosfiltfilt


def bandpass_filter(signal, sfreq, low, high, order):
    """Zero-phase Butterworth band-pass filter of a 1-D signal.

    Parameters
    ----------
    signal : np.ndarray, shape (n_times,)
        Signal to filter.
    sfreq : float
        Sampling frequency in Hz.
    low : float
        Lower passband edge in Hz.
    high : float
        Upper passband edge in Hz.
    order : int
        Butterworth filter order.

    Returns
    -------
    np.ndarray, shape (n_times,)
        Filtered signal with zero phase distortion.
    """
    sos = butter(order, [low, high], btype="bandpass", fs=sfreq, output="sos")
    return sosfiltfilt(sos, signal)


def filter_individual_band(signal, sfreq, center_freq, bandwidth, order):
    """Band-pass filter around an individualized rhythm.

    Parameters
    ----------
    signal : np.ndarray, shape (n_times,)
        Signal to filter.
    sfreq : float
        Sampling frequency in Hz.
    center_freq : float
        Individualized rhythm center frequency in Hz.
    bandwidth : float
        Half-width of the individualized passband in Hz.
    order : int
        Butterworth filter order.

    Returns
    -------
    np.ndarray, shape (n_times,)
        Signal filtered between ``center_freq - bandwidth`` and
        ``center_freq + bandwidth``.
    """
    return bandpass_filter(
        signal,
        sfreq,
        center_freq - bandwidth,
        center_freq + bandwidth,
        order,
    )


def hilbert_envelope(signal):
    """Compute the instantaneous amplitude envelope via the analytic signal.

    Parameters
    ----------
    signal : np.ndarray, shape (n_times,)
        Narrow-band filtered signal.

    Returns
    -------
    np.ndarray, shape (n_times,)
        Absolute value of the Hilbert analytic signal.
    """
    return np.abs(hilbert(signal))


def downsample(signal, sfreq, target_sfreq):
    """Anti-alias and resample a 1-D signal to a target sampling frequency.

    Parameters
    ----------
    signal : np.ndarray, shape (n_times,)
        Signal to resample.
    sfreq : float
        Original sampling frequency in Hz.
    target_sfreq : float
        Desired output sampling frequency in Hz.

    Returns
    -------
    down_signal : np.ndarray
        Anti-aliased, resampled signal.
    down_sfreq : float
        Realized output sampling frequency in Hz.
    """
    ratio = Fraction(str(target_sfreq)) / Fraction(str(sfreq))
    down_signal = resample_poly(signal, ratio.numerator, ratio.denominator)
    down_sfreq = sfreq * ratio.numerator / ratio.denominator
    return down_signal, down_sfreq


def eeg_band_envelope(signal, sfreq, center_freq, bandwidth, order, target_sfreq):
    """Compute a downsampled amplitude envelope for an individualized EEG band.

    Parameters
    ----------
    signal : np.ndarray, shape (n_times,)
        EEG signal to process.
    sfreq : float
        EEG sampling frequency in Hz.
    center_freq : float
        Individualized rhythm center frequency in Hz.
    bandwidth : float
        Half-width of the individualized passband in Hz.
    order : int
        Butterworth filter order.
    target_sfreq : float
        Desired envelope sampling frequency in Hz.

    Returns
    -------
    envelope : np.ndarray
        Downsampled instantaneous amplitude envelope.
    env_sfreq : float
        Realized envelope sampling frequency in Hz.
    """
    filtered = filter_individual_band(
        signal, sfreq, center_freq, bandwidth, order
    )
    envelope = hilbert_envelope(filtered)
    return downsample(envelope, sfreq, target_sfreq)


def hrv_hf_envelope(
    ibi_signal,
    ibi_sfreq,
    hf_low,
    hf_high,
    order,
    target_sfreq,
):
    """Compute the downsampled instantaneous amplitude envelope of HRV HF.

    HF band edges are supplied by the caller so they can be age-adjusted.
    This function expects an IBI/tachogram signal already interpolated to a
    regular grid and does not perform RR interval interpolation.

    Parameters
    ----------
    ibi_signal : np.ndarray, shape (n_times,)
        Evenly-sampled interbeat-interval or tachogram signal.
    ibi_sfreq : float
        Sampling frequency of the regular IBI signal in Hz.
    hf_low : float
        Lower HF band edge in Hz.
    hf_high : float
        Upper HF band edge in Hz.
    order : int
        Butterworth filter order.
    target_sfreq : float
        Desired envelope sampling frequency in Hz.

    Returns
    -------
    envelope : np.ndarray
        Downsampled HF-band instantaneous amplitude envelope.
    env_sfreq : float
        Realized envelope sampling frequency in Hz.
    """
    filtered = bandpass_filter(ibi_signal, ibi_sfreq, hf_low, hf_high, order)
    envelope = hilbert_envelope(filtered)
    return downsample(envelope, ibi_sfreq, target_sfreq)


def average_channels(signals):
    """Average signals across channels.

    Parameters
    ----------
    signals : np.ndarray, shape (n_channels, n_times)
        Channel-wise signals. Single-channel ROIs use one channel.

    Returns
    -------
    np.ndarray, shape (n_times,)
        Mean signal across channels.
    """
    return signals.mean(axis=0)


def plot_signal_filtered_envelope(raw, filtered, envelope, sfreq, title):
    """Plot a raw signal alongside its filtered signal and amplitude envelope.

    Parameters
    ----------
    raw : np.ndarray, shape (n_times,)
        Unfiltered signal.
    filtered : np.ndarray, shape (n_times,)
        Band-pass filtered signal.
    envelope : np.ndarray, shape (n_times,)
        Instantaneous amplitude envelope at ``sfreq``.
    sfreq : float
        Sampling frequency in Hz.
    title : str
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the sanity plot.
    """
    time = np.arange(raw.size) / sfreq
    figure, axes = plt.subplots(nrows=2, sharex=True)
    axes[0].plot(time, raw)
    axes[0].set_ylabel("Raw amplitude")
    axes[1].plot(time, filtered, label="Filtered")
    axes[1].plot(time, envelope, label="Envelope")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Amplitude")
    axes[1].legend()
    figure.suptitle(title)
    figure.tight_layout()
    return figure


def plot_dyad_envelopes(env_child, env_caregiver, sfreq, title, labels):
    """Plot child and caregiver envelopes on a shared time axis.

    Parameters
    ----------
    env_child : np.ndarray, shape (n_times,)
        Child amplitude envelope.
    env_caregiver : np.ndarray, shape (n_times,)
        Caregiver amplitude envelope.
    sfreq : float
        Envelope sampling frequency in Hz.
    title : str
        Figure title.
    labels : tuple[str, str]
        Labels for the child and caregiver traces.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the overlaid envelopes.
    """
    child_time = np.arange(env_child.size) / sfreq
    caregiver_time = np.arange(env_caregiver.size) / sfreq
    figure, axis = plt.subplots()
    axis.plot(child_time, env_child, label=labels[0])
    axis.plot(caregiver_time, env_caregiver, label=labels[1])
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Amplitude")
    axis.set_title(title)
    axis.legend()
    figure.tight_layout()
    return figure


def plot_eeg_hrv_envelopes(env_eeg, eeg_sfreq, env_hrv, hrv_sfreq, title):
    """Plot EEG-band and HRV-HF envelopes on separate shared-time panels.

    Parameters
    ----------
    env_eeg : np.ndarray, shape (n_times,)
        EEG-band amplitude envelope.
    eeg_sfreq : float
        EEG envelope sampling frequency in Hz.
    env_hrv : np.ndarray, shape (n_times,)
        HRV-HF amplitude envelope.
    hrv_sfreq : float
        HRV envelope sampling frequency in Hz.
    title : str
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the EEG and HRV envelope panels.
    """
    eeg_time = np.arange(env_eeg.size) / eeg_sfreq
    hrv_time = np.arange(env_hrv.size) / hrv_sfreq
    figure, axes = plt.subplots(nrows=2, sharex=True)
    axes[0].plot(eeg_time, env_eeg)
    axes[0].set_ylabel("EEG amplitude")
    axes[1].plot(hrv_time, env_hrv)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("HRV amplitude")
    figure.suptitle(title)
    figure.tight_layout()
    return figure
