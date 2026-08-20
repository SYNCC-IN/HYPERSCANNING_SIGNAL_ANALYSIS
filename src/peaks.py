"""Peak inventory: prevalence across the scalp and frequency, and channel clustering."""

import numpy as np
import pandas as pd


def classify_channel_cluster(channel_name, channel_clusters):
    """Map a channel name to a scalp cluster label.

    Parameters
    ----------
    channel_name : str
        Channel name (e.g. ``'Fz'``, ``'P3'``).
    channel_clusters : dict
        Mapping of ``{cluster_label: [channel_names]}``.

    Returns
    -------
    str
        The cluster label containing ``channel_name``, or ``'unassigned'``
        if it is not found in any cluster.
    """
    for label, channels in channel_clusters.items():
        if channel_name in channels:
            return label
    return 'unassigned'


def compute_peak_prevalence(peaks_df, channel_names, freq_bins, role, group=None, n_participants=None):
    """Compute the fraction of participants with a detected peak in each freq bin x channel.

    Parameters
    ----------
    peaks_df : pandas.DataFrame
        Peak inventory table with columns ``participant_id``, ``role``, ``group``,
        ``channel``, ``center_freq``.
    channel_names : list of str
        Channels to include as rows of the output.
    freq_bins : list of tuple of float
        List of ``(low, high)`` frequency bin edges in Hz.
    role : str
        Role to filter on (``'child'`` or ``'caregiver'``).
    group : str or None, optional
        If given, restrict to this diagnostic group only.
    n_participants : int or None, optional
        Total participant count to use as the prevalence denominator. If None,
        it is inferred from the unique ``participant_id`` values in ``peaks_df``
        after filtering, which *undercounts* participants who have zero
        detected peaks in every channel (they never appear in ``peaks_df`` at
        all). Callers with access to a complete participant roster (e.g. the
        fit-quality table, which has a row per channel regardless of peak
        count) should pass the correct total explicitly.

    Returns
    -------
    pandas.DataFrame
        Prevalence values (0-1), indexed by channel, one column per frequency bin
        (labelled ``'{low}-{high}'``).
    """
    subset = peaks_df[peaks_df['role'] == role]
    if group is not None:
        subset = subset[subset['group'] == group]

    if n_participants is None:
        n_participants = subset['participant_id'].nunique()

    columns = {}
    for low, high in freq_bins:
        label = f'{low}-{high}'
        in_bin = subset[(subset['center_freq'] >= low) & (subset['center_freq'] < high)]
        counts = in_bin.groupby('channel')['participant_id'].nunique().reindex(channel_names, fill_value=0)
        columns[label] = counts / n_participants if n_participants > 0 else counts.astype(float)

    return pd.DataFrame(columns, index=channel_names)


CLUSTER_DTYPES = {
    'cluster_id': 'int64', 'mean_center_freq': 'float64', 'std_center_freq': 'float64',
    'total_power': 'float64', 'mean_power': 'float64', 'n_channels': 'int64',
    'channel_list': 'object', 'is_robust': 'bool',
}
CLUSTER_COLUMNS = list(CLUSTER_DTYPES)


def cluster_peaks_across_channels(peaks_df, freq_tolerance=1.0, min_channels=3):
    """Cluster detected peaks across channels by center frequency proximity.

    For a single participant: sort all detected peaks by center_freq, then
    greedily assign peaks to clusters. A peak joins the most recently opened
    cluster if its center_freq is within freq_tolerance of that cluster's
    running mean center_freq; otherwise a new cluster starts. Because peaks
    are processed in ascending frequency order, the running mean of earlier
    clusters can only be lower, so checking against the most recent cluster
    is equivalent to checking against all open clusters.

    Parameters
    ----------
    peaks_df : pd.DataFrame
        Peaks for ONE participant. Must contain columns:
        channel, center_freq, power, bandwidth.
    freq_tolerance : float
        Maximum difference in Hz between a peak's center_freq and
        a cluster's mean center_freq for the peak to join that cluster.
    min_channels : int
        Minimum number of channels for a cluster to be considered
        topographically robust.

    Returns
    -------
    pd.DataFrame
        One row per cluster with columns:
        - cluster_id : int
        - mean_center_freq : float (mean across member peaks)
        - std_center_freq : float
        - total_power : float (sum of power across member channels)
        - mean_power : float
        - n_channels : int
        - channel_list : str (comma-separated channel names)
        - is_robust : bool (n_channels >= min_channels)
    """
    sorted_peaks = peaks_df.sort_values('center_freq')

    clusters = []
    for _, peak in sorted_peaks.iterrows():
        if clusters and abs(peak['center_freq'] - clusters[-1]['mean_center_freq']) <= freq_tolerance:
            cluster = clusters[-1]
            cluster['members'].append(peak)
            cluster['mean_center_freq'] = np.mean([m['center_freq'] for m in cluster['members']])
        else:
            clusters.append({'members': [peak], 'mean_center_freq': peak['center_freq']})

    rows = []
    for cluster_id, cluster in enumerate(clusters):
        members = pd.DataFrame(cluster['members'])
        channels = members['channel'].unique()
        rows.append({
            'cluster_id': cluster_id,
            'mean_center_freq': members['center_freq'].mean(),
            'std_center_freq': members['center_freq'].std(ddof=0),
            'total_power': members['power'].sum(),
            'mean_power': members['power'].mean(),
            'n_channels': len(channels),
            'channel_list': ','.join(sorted(channels)),
            'is_robust': len(channels) >= min_channels,
        })

    return pd.DataFrame(rows, columns=CLUSTER_COLUMNS).astype(CLUSTER_DTYPES)


ROI_CLUSTER_DTYPES = {
    'cluster_id': 'int64', 'weighted_center_freq': 'float64', 'summed_power': 'float64',
    'n_channels': 'int64', 'n_peaks': 'int64', 'channel_list': 'object', 'mean_bandwidth': 'float64',
}
ROI_CLUSTER_COLUMNS = list(ROI_CLUSTER_DTYPES)


def cluster_peaks_within_roi(peaks_df, roi_channels, freq_tolerance=1.0):
    """Cluster peaks within a single ROI by center frequency proximity.

    For one participant and one ROI: filter peaks to roi_channels, sort by
    center_freq, then greedily assign peaks to clusters. A peak joins the
    most recently opened cluster if its center_freq is within
    freq_tolerance of that cluster's running power-weighted mean
    center_freq; otherwise a new cluster starts. Because peaks are
    processed in ascending frequency order, checking against only the most
    recent cluster is equivalent to checking against all open clusters.

    Parameters
    ----------
    peaks_df : pd.DataFrame
        Peaks for ONE participant. Columns: channel, center_freq, power, bandwidth.
    roi_channels : list of str
        Channel names belonging to this ROI.
    freq_tolerance : float
        Max Hz distance from cluster's power-weighted mean to join.

    Returns
    -------
    pd.DataFrame
        One row per cluster:
        - cluster_id : int
        - weighted_center_freq : float (power-weighted mean of member peaks' center_freq)
        - summed_power : float (sum of power across member peaks)
        - n_channels : int (number of unique channels in cluster)
        - n_peaks : int (total peaks, may exceed n_channels if one channel contributed two peaks)
        - channel_list : str (comma-separated)
        - mean_bandwidth : float
    """
    roi_peaks = peaks_df[peaks_df['channel'].isin(roi_channels)].sort_values('center_freq')

    clusters = []
    for _, peak in roi_peaks.iterrows():
        if clusters and abs(peak['center_freq'] - clusters[-1]['weighted_center_freq']) <= freq_tolerance:
            cluster = clusters[-1]
        else:
            cluster = {'members': [], 'weighted_center_freq': peak['center_freq']}
            clusters.append(cluster)
        cluster['members'].append(peak)
        freqs = np.array([m['center_freq'] for m in cluster['members']])
        powers = np.array([m['power'] for m in cluster['members']])
        cluster['weighted_center_freq'] = np.sum(freqs * powers) / np.sum(powers)

    rows = []
    for cluster_id, cluster in enumerate(clusters):
        members = pd.DataFrame(cluster['members'])
        channels = members['channel'].unique()
        rows.append({
            'cluster_id': cluster_id,
            'weighted_center_freq': cluster['weighted_center_freq'],
            'summed_power': members['power'].sum(),
            'n_channels': len(channels),
            'n_peaks': len(members),
            'channel_list': ','.join(sorted(channels)),
            'mean_bandwidth': members['bandwidth'].mean(),
        })

    return pd.DataFrame(rows, columns=ROI_CLUSTER_COLUMNS).astype(ROI_CLUSTER_DTYPES)


def cluster_peaks_all_rois(peaks_df, channel_clusters, freq_tolerance=1.0):
    """Run within-ROI peak clustering for all ROIs for one participant.

    Parameters
    ----------
    peaks_df : pd.DataFrame
        Peaks for ONE participant.
    channel_clusters : dict
        Mapping {roi_label: [channel_names]}.
    freq_tolerance : float
        Passed to cluster_peaks_within_roi.

    Returns
    -------
    pd.DataFrame
        Concatenated clusters across all ROIs, with added column 'roi'.
    """
    roi_tables = []
    for roi_label, roi_channels in channel_clusters.items():
        roi_clusters = cluster_peaks_within_roi(peaks_df, roi_channels, freq_tolerance=freq_tolerance)
        roi_clusters['roi'] = roi_label
        roi_tables.append(roi_clusters)
    return pd.concat(roi_tables, ignore_index=True)


def count_peaks_per_roi(peaks_df, roi_channels, freq_range=(3, 14)):
    """Count peaks per participant within an ROI and frequency range.

    Parameters
    ----------
    peaks_df : pd.DataFrame
        Peaks for ONE participant. Columns: channel, center_freq, power, bandwidth.
    roi_channels : list of str
        Channels belonging to the ROI.
    freq_range : tuple of float
        (low, high) frequency bounds to consider.

    Returns
    -------
    dict
        n_peaks : int — total peaks in ROI within freq_range
        peak_freqs : list of float — center_freq of each peak, sorted ascending
        peak_powers : list of float — power of each peak, same order
    """
    subset = peaks_df[
        peaks_df['channel'].isin(roi_channels)
        & (peaks_df['center_freq'] >= freq_range[0])
        & (peaks_df['center_freq'] <= freq_range[1])
    ].sort_values('center_freq')

    return {
        'n_peaks': len(subset),
        'peak_freqs': subset['center_freq'].tolist(),
        'peak_powers': subset['power'].tolist(),
    }
