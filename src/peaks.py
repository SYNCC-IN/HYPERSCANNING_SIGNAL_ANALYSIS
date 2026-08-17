"""Peak inventory: prevalence across the scalp and frequency, and channel clustering."""

import pandas as pd

CHANNEL_CLUSTERS = {
    'Fp1': 'frontal', 'Fp2': 'frontal', 'F7': 'frontal', 'F3': 'frontal',
    'Fz': 'frontal', 'F4': 'frontal', 'F8': 'frontal',
    'C3': 'central', 'Cz': 'central', 'C4': 'central',
    'P3': 'parietal', 'Pz': 'parietal', 'P4': 'parietal',
    'O1': 'occipital', 'O2': 'occipital',
    'T3': 'temporal', 'T4': 'temporal', 'T5': 'temporal', 'T6': 'temporal',
    'T7': 'temporal', 'T8': 'temporal', 'P7': 'temporal', 'P8': 'temporal',
}


def classify_channel_cluster(channel_name):
    """Map a standard 10-20 channel name to a scalp cluster label.

    Parameters
    ----------
    channel_name : str
        Channel name (e.g. ``'Fz'``, ``'P3'``).

    Returns
    -------
    str
        One of ``'frontal'``, ``'central'``, ``'parietal'``, ``'occipital'``,
        ``'temporal'``.
    """
    return CHANNEL_CLUSTERS[channel_name]


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
