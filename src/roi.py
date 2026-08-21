"""ROI (region of interest) definitions and averaging."""

import numpy as np
import pandas as pd


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


def validate_roi_individual_bands(peaks_df, band_assignments_df, roi_channels, roi_label,
                                   slow_window=(3, 7), fast_window=(7, 14)):
    """Validate ROI peak prevalence using each participant's individualized band windows.

    For each participant appearing in ``peaks_df``, looks up their slow/fast
    rhythm center frequency and bandwidth (or single dominant rhythm) for this
    ROI from ``band_assignments_df``, then checks whether any detected peak at
    ``roi_channels`` falls within the resulting individual frequency window.

    - ``assignment_note == 'two_rhythms'``: the individual slow window is
      ``[slow_cf - slow_bw, slow_cf + slow_bw]`` and the individual fast
      window is ``[fast_cf - fast_bw, fast_cf + fast_bw]``; a peak counts as
      present in a band if its center_freq falls inside that band's window.
    - ``assignment_note == 'single_dominant'``: the window
      ``[dominant_cf - dominant_bw, dominant_cf + dominant_bw]`` is compared
      against the fixed ``slow_window``/``fast_window`` reference ranges, and
      the participant is counted present in whichever reference range(s) it
      overlaps. If it overlaps neither, the participant is absent for both
      bands.
    - ``assignment_note == 'no_peaks'``, or no band assignment at all for this
      ROI (participant missing from ``band_assignments_df`` for ``roi_label``):
      absent for both bands.

    Parameters
    ----------
    peaks_df : pd.DataFrame
        All peaks (e.g. from all_peaks.csv). Columns: participant_id, role,
        group, channel, center_freq, power, bandwidth.
    band_assignments_df : pd.DataFrame
        Band assignments from Step 4. Columns include: participant_id, role,
        group, roi, slow_cf, slow_bw, fast_cf, fast_bw, dominant_cf,
        dominant_bw, assignment_note.
    roi_channels : list of str
        Channels in this ROI.
    roi_label : str
        ROI name — used to filter ``band_assignments_df`` by its 'roi' column.
    slow_window, fast_window : tuple of float, optional
        Fixed (low, high) Hz reference ranges used only to classify
        ``single_dominant`` participants as slow or fast.

    Returns
    -------
    pd.DataFrame
        One row per participant (from ``peaks_df``): participant_id, role,
        group, slow_present (bool), fast_present (bool).
    """
    def _any_in_window(freqs, low, high):
        return bool(((freqs >= low) & (freqs <= high)).any())

    def _windows_overlap(a_low, a_high, b_low, b_high):
        return a_low <= b_high and b_low <= a_high

    participants = peaks_df[['participant_id', 'role', 'group']].drop_duplicates()
    roi_assignments = band_assignments_df[band_assignments_df['roi'] == roi_label].set_index('participant_id')
    roi_peaks = peaks_df[peaks_df['channel'].isin(roi_channels)]

    rows = []
    for _, prow in participants.iterrows():
        pid = prow['participant_id']
        slow_present, fast_present = False, False

        if pid in roi_assignments.index:
            assignment = roi_assignments.loc[pid]
            note = assignment['assignment_note']
            participant_freqs = roi_peaks.loc[roi_peaks['participant_id'] == pid, 'center_freq']

            if note == 'two_rhythms':
                slow_present = _any_in_window(
                    participant_freqs, assignment['slow_cf'] - assignment['slow_bw'],
                    assignment['slow_cf'] + assignment['slow_bw'],
                )
                fast_present = _any_in_window(
                    participant_freqs, assignment['fast_cf'] - assignment['fast_bw'],
                    assignment['fast_cf'] + assignment['fast_bw'],
                )
            elif note == 'single_dominant':
                dom_low = assignment['dominant_cf'] - assignment['dominant_bw']
                dom_high = assignment['dominant_cf'] + assignment['dominant_bw']
                slow_present = _windows_overlap(dom_low, dom_high, *slow_window)
                fast_present = _windows_overlap(dom_low, dom_high, *fast_window)
            # 'no_peaks': leave both False

        rows.append({
            'participant_id': pid,
            'role': prow['role'],
            'group': prow['group'],
            'slow_present': slow_present,
            'fast_present': fast_present,
        })

    return pd.DataFrame(rows)


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
