"""specparam (spectral parameterization) fitting, extraction, and quality checks."""

import pandas as pd
from specparam import SpectralModel


def fit_specparam(freqs, psd_1d, freq_range, peak_width_limits, max_n_peaks, min_peak_height, aperiodic_mode):
    """Fit a specparam SpectralModel to a single power spectrum.

    Parameters
    ----------
    freqs : ndarray, shape (n_freqs,)
        Frequency bins in Hz.
    psd_1d : ndarray, shape (n_freqs,)
        Power spectral density for one channel.
    freq_range : tuple of float
        (fmin, fmax) fitting range in Hz.
    peak_width_limits : tuple of float
        (min, max) allowed peak bandwidth in Hz.
    max_n_peaks : int
        Maximum number of peaks to fit.
    min_peak_height : float
        Minimum peak height (in log power) above the aperiodic fit.
    aperiodic_mode : str
        ``'fixed'`` or ``'knee'``.

    Returns
    -------
    specparam.SpectralModel
        Fitted model.
    """
    model = SpectralModel(
        peak_width_limits=peak_width_limits,
        max_n_peaks=max_n_peaks,
        min_peak_height=min_peak_height,
        aperiodic_mode=aperiodic_mode,
        verbose=False,
    )
    model.fit(freqs, psd_1d, freq_range=freq_range)
    return model


def extract_peak_params(model):
    """Extract detected peak parameters from a fitted specparam model.

    Parameters
    ----------
    model : specparam.SpectralModel
        Fitted model.

    Returns
    -------
    list of dict
        One dict per detected peak with keys ``center_freq``, ``power``,
        ``bandwidth``. Empty list if no peaks were detected.
    """
    peak_params = model.get_params('peak')
    return [
        {'center_freq': row[0], 'power': row[1], 'bandwidth': row[2]}
        for row in peak_params
    ]


def extract_aperiodic_params(model):
    """Extract aperiodic (1/f) parameters from a fitted specparam model.

    Parameters
    ----------
    model : specparam.SpectralModel
        Fitted model.

    Returns
    -------
    dict
        ``{'offset': float, 'exponent': float}`` for ``'fixed'`` mode, or
        ``{'offset': float, 'knee': float, 'exponent': float}`` for ``'knee'`` mode.
    """
    mode = model.modes.aperiodic.name
    if mode == 'knee':
        offset, knee, exponent = model.get_params('aperiodic')
        return {'offset': offset, 'knee': knee, 'exponent': exponent}
    offset, exponent = model.get_params('aperiodic')
    return {'offset': offset, 'exponent': exponent}


def extract_fit_quality(model):
    """Extract goodness-of-fit metrics from a fitted specparam model.

    Parameters
    ----------
    model : specparam.SpectralModel
        Fitted model.

    Returns
    -------
    dict
        ``{'r_squared': float, 'error': float}``.
    """
    return {
        'r_squared': model.get_metrics('gof', 'rsquared'),
        'error': model.get_metrics('error', 'mae'),
    }


def batch_fit_specparam(freqs, psd_2d, channel_names, freq_range, peak_width_limits,
                         max_n_peaks, min_peak_height, aperiodic_mode):
    """Fit specparam across all channels of one participant's PSD.

    Parameters
    ----------
    freqs : ndarray, shape (n_freqs,)
        Frequency bins in Hz.
    psd_2d : ndarray, shape (n_channels, n_freqs)
        Power spectral density, one row per channel.
    channel_names : list of str
        Channel names matching the rows of ``psd_2d``.
    freq_range, peak_width_limits, max_n_peaks, min_peak_height, aperiodic_mode
        Passed through to :func:`fit_specparam`.

    Returns
    -------
    peaks_df : pandas.DataFrame
        One row per detected peak, columns: ``channel``, ``center_freq``, ``power``,
        ``bandwidth``.
    quality_df : pandas.DataFrame
        One row per channel, columns: ``channel``, ``r_squared``, ``error``,
        ``n_peaks``, ``aperiodic_exponent``, ``aperiodic_offset``.
    """
    peak_rows = []
    quality_rows = []
    for ch_idx, channel in enumerate(channel_names):
        model = fit_specparam(
            freqs, psd_2d[ch_idx, :], freq_range, peak_width_limits,
            max_n_peaks, min_peak_height, aperiodic_mode,
        )
        peaks = extract_peak_params(model)
        aperiodic = extract_aperiodic_params(model)
        quality = extract_fit_quality(model)

        for peak in peaks:
            peak_rows.append({'channel': channel, **peak})

        quality_rows.append({
            'channel': channel,
            'r_squared': quality['r_squared'],
            'error': quality['error'],
            'n_peaks': len(peaks),
            'aperiodic_exponent': aperiodic['exponent'],
            'aperiodic_offset': aperiodic['offset'],
        })

    peaks_df = pd.DataFrame(peak_rows, columns=['channel', 'center_freq', 'power', 'bandwidth'])
    quality_df = pd.DataFrame(quality_rows)
    return peaks_df, quality_df


def find_peak_in_window(peaks, freq_window):
    """Find the strongest peak within a frequency window.

    Parameters
    ----------
    peaks : list of dict
        Detected peaks: [{center_freq, power, bandwidth}, ...].
    freq_window : tuple of float
        (low, high) Hz.

    Returns
    -------
    dict or None
        The strongest peak (by power) within the window, or None if no peak found.
    """
    low, high = freq_window
    in_window = [p for p in peaks if low <= p['center_freq'] <= high]
    if not in_window:
        return None
    return max(in_window, key=lambda p: p['power'])
