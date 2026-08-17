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
