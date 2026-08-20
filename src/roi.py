"""ROI (region of interest) definitions and averaging."""

import numpy as np


def define_rois_theory():
    """Return theory-driven ROI channel groupings for the 19-channel 10-20 montage.

    Returns
    -------
    dict
        Mapping of ROI name to list of channel names.
    """
    return {
        'frontal_midline': ['Fz'],
        'frontal_lateral': ['F3', 'F4'],
        'central': ['C3', 'Cz', 'C4'],
        'parietal': ['P3', 'Pz', 'P4'],
        'occipital': ['O1', 'O2'],
        'temporal': ['T7', 'T8', 'P7', 'P8'],
    }


def validate_roi_with_prevalence(prevalence_df, roi_channels, freq_bin, min_prevalence=0.5):
    """Check whether all channels in an ROI meet a minimum peak-prevalence threshold.

    Parameters
    ----------
    prevalence_df : pandas.DataFrame
        Prevalence table indexed by channel, as returned by
        :func:`src.peaks.compute_peak_prevalence`, with a column labelled
        ``'{low}-{high}'`` matching ``freq_bin``.
    roi_channels : list of str
        Channels belonging to the ROI.
    freq_bin : tuple of float
        ``(low, high)`` frequency bin edges in Hz.
    min_prevalence : float, optional
        Minimum required prevalence (0-1) for every channel in the ROI.

    Returns
    -------
    passes : bool
        True if every channel in the ROI meets ``min_prevalence``.
    channel_prevalence : pandas.Series
        Per-channel prevalence values for the ROI at this frequency bin.
    """
    label = f'{freq_bin[0]}-{freq_bin[1]}'
    channel_prevalence = prevalence_df.loc[roi_channels, label]
    passes = bool((channel_prevalence >= min_prevalence).all())
    return passes, channel_prevalence


def validate_roi_two_bands(prevalence_df, roi_channels, slow_window, fast_window, min_prevalence=0.5):
    """Validate ROI viability separately for slow and fast bands.

    Parameters
    ----------
    prevalence_df : pd.DataFrame
        Peak prevalence per channel per frequency bin (from step 3), with
        columns labelled ``'{low}-{high}'`` matching ``slow_window`` and
        ``fast_window``.
    roi_channels : list of str
        Channels in this ROI.
    slow_window : tuple of float
        (low, high) Hz for slow rhythm validation.
    fast_window : tuple of float
        (low, high) Hz for fast rhythm validation.
    min_prevalence : float
        Threshold for viability.

    Returns
    -------
    dict
        roi_channels : list of str
        slow_prevalence : float — mean prevalence across ROI channels in slow window
        fast_prevalence : float — mean prevalence across ROI channels in fast window
        slow_viable : bool
        fast_viable : bool
    """
    slow_label = f'{slow_window[0]}-{slow_window[1]}'
    fast_label = f'{fast_window[0]}-{fast_window[1]}'
    slow_prevalence = float(prevalence_df.loc[roi_channels, slow_label].mean())
    fast_prevalence = float(prevalence_df.loc[roi_channels, fast_label].mean())
    return {
        'roi_channels': roi_channels,
        'slow_prevalence': slow_prevalence,
        'fast_prevalence': fast_prevalence,
        'slow_viable': slow_prevalence >= min_prevalence,
        'fast_viable': fast_prevalence >= min_prevalence,
    }


def check_peak_survival(channel_peaks_list, roi_peaks, band_window, freq_tolerance=1.0):
    """Check if a peak detected at individual channels survives ROI averaging.

    Parameters
    ----------
    channel_peaks_list : list of dict
        Peaks from individual channel fits within the ROI.
        Each dict: {channel, center_freq, power, bandwidth}.
    roi_peaks : list of dict
        Peaks from the ROI-averaged PSD fit.
        Each dict: {center_freq, power, bandwidth}.
    band_window : tuple of float
        (low, high) Hz — which band to check survival in.
    freq_tolerance : float
        Max Hz shift between channel-level and ROI-level peak to count as "survived".

    Returns
    -------
    dict
        channel_peak_present : bool — was a peak in band_window at individual channels?
        roi_peak_present : bool — is a peak in band_window in the ROI-averaged fit?
        survived : bool — both present and within freq_tolerance?
        channel_cf : float or None — mean center_freq at channel level
        roi_cf : float or None — center_freq at ROI level
        cf_shift : float or None — |roi_cf - channel_cf|
    """
    low, high = band_window
    channel_in_band = [p for p in channel_peaks_list if low <= p['center_freq'] <= high]
    roi_in_band = [p for p in roi_peaks if low <= p['center_freq'] <= high]

    channel_peak_present = len(channel_in_band) > 0
    roi_peak_present = len(roi_in_band) > 0

    channel_cf = float(np.mean([p['center_freq'] for p in channel_in_band])) if channel_peak_present else None
    roi_cf = float(max(roi_in_band, key=lambda p: p['power'])['center_freq']) if roi_peak_present else None

    cf_shift = abs(roi_cf - channel_cf) if channel_cf is not None and roi_cf is not None else None
    survived = bool(cf_shift is not None and cf_shift <= freq_tolerance)

    return {
        'channel_peak_present': channel_peak_present,
        'roi_peak_present': roi_peak_present,
        'survived': survived,
        'channel_cf': channel_cf,
        'roi_cf': roi_cf,
        'cf_shift': cf_shift,
    }


def average_psd_within_roi(psd_2d, channel_names, roi_channels):
    """Average PSD across the channels belonging to an ROI.

    Parameters
    ----------
    psd_2d : ndarray, shape (n_channels, n_freqs)
        Power spectral density, one row per channel.
    channel_names : list of str
        Channel names matching the rows of ``psd_2d``.
    roi_channels : list of str
        Channels to average over.

    Returns
    -------
    ndarray, shape (n_freqs,)
        Mean PSD across the ROI channels.
    """
    if not roi_channels:
        raise ValueError('roi_channels is empty; cannot average PSD over zero channels.')
    idx = [channel_names.index(ch) for ch in roi_channels]
    return np.mean(psd_2d[idx, :], axis=0)
