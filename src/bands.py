"""Slow/fast rhythm band assignment from within-ROI peak clusters, and IAF metrics."""

import numpy as np
import pandas as pd

PRIMARY_IAF_ROI = 'parietal'
FALLBACK_IAF_ROI = 'sensorimotor'


def assign_two_bands(roi_clusters_df, participant_id, role, min_gap=1.5):
    """Assign slow and fast rhythm bands from within-ROI peak clusters.

    Takes the within-ROI clustered peaks for ONE participant at ONE ROI.
    If two or more clusters exist with a frequency gap >= min_gap somewhere
    among them, the two strongest clusters (by summed_power) are assigned as
    slow and fast rhythms. Otherwise (0 or 1 cluster, or 2+ clusters none of
    which are >= min_gap apart) the strongest cluster is assigned as a single
    dominant rhythm.

    Parameters
    ----------
    roi_clusters_df : pd.DataFrame
        Within-ROI clusters for one participant, one ROI. Output of
        cluster_peaks_within_roi. Columns include: cluster_id,
        weighted_center_freq, summed_power, n_channels, mean_bandwidth.
    participant_id : str
        Participant identifier.
    role : str
        'child' or 'caregiver'.
    min_gap : float
        Minimum frequency gap (Hz) between two clusters to be
        considered separate rhythms.

    Returns
    -------
    dict
        participant_id, role, n_clusters, n_rhythm_assignment,
        slow_cf, slow_power, slow_bw, fast_cf, fast_power, fast_bw, freq_gap,
        dominant_cf, dominant_power, dominant_bw, assignment_note.
        Frequency/power/bandwidth fields are None where not assigned.
    """
    result = {
        'participant_id': participant_id,
        'role': role,
        'n_clusters': len(roi_clusters_df),
        'n_rhythm_assignment': None,
        'slow_cf': None, 'slow_power': None, 'slow_bw': None,
        'fast_cf': None, 'fast_power': None, 'fast_bw': None,
        'freq_gap': None,
        'dominant_cf': None, 'dominant_power': None, 'dominant_bw': None,
        'assignment_note': None,
    }

    if len(roi_clusters_df) == 0:
        result['n_rhythm_assignment'] = 0
        result['assignment_note'] = 'no_peaks'
        return result

    if len(roi_clusters_df) == 1:
        cluster = roi_clusters_df.iloc[0]
        result['n_rhythm_assignment'] = 1
        result['dominant_cf'] = float(cluster['weighted_center_freq'])
        result['dominant_power'] = float(cluster['summed_power'])
        result['dominant_bw'] = float(cluster['mean_bandwidth'])
        result['assignment_note'] = 'single_dominant'
        return result

    freqs = roi_clusters_df['weighted_center_freq'].sort_values().tolist()
    has_qualifying_pair = any(
        freqs[j] - freqs[i] >= min_gap
        for i in range(len(freqs)) for j in range(i + 1, len(freqs))
    )

    if not has_qualifying_pair:
        strongest = roi_clusters_df.loc[roi_clusters_df['summed_power'].idxmax()]
        result['n_rhythm_assignment'] = 1
        result['dominant_cf'] = float(strongest['weighted_center_freq'])
        result['dominant_power'] = float(strongest['summed_power'])
        result['dominant_bw'] = float(strongest['mean_bandwidth'])
        result['assignment_note'] = 'single_dominant'
        return result

    top2 = roi_clusters_df.sort_values('summed_power', ascending=False).head(2).sort_values('weighted_center_freq')
    slow, fast = top2.iloc[0], top2.iloc[1]
    result['n_rhythm_assignment'] = 2
    result['slow_cf'] = float(slow['weighted_center_freq'])
    result['slow_power'] = float(slow['summed_power'])
    result['slow_bw'] = float(slow['mean_bandwidth'])
    result['fast_cf'] = float(fast['weighted_center_freq'])
    result['fast_power'] = float(fast['summed_power'])
    result['fast_bw'] = float(fast['mean_bandwidth'])
    result['freq_gap'] = result['fast_cf'] - result['slow_cf']
    result['assignment_note'] = 'two_rhythms'
    return result


def assign_bands_all_rois(roi_clusters_all_df, participant_id, role, rois, min_gap=1.5):
    """Run assign_two_bands for each ROI for one participant.

    Parameters
    ----------
    roi_clusters_all_df : pd.DataFrame
        Within-ROI clusters for one participant, all ROIs.
        Must contain 'roi' column.
    participant_id : str
        Participant identifier.
    role : str
        'child' or 'caregiver'.
    rois : list of str
        ROI labels to process.
    min_gap : float
        Passed to assign_two_bands.

    Returns
    -------
    pd.DataFrame
        One row per ROI with all fields from assign_two_bands plus 'roi' column.
    """
    rows = []
    for roi in rois:
        roi_clusters = roi_clusters_all_df[roi_clusters_all_df['roi'] == roi]
        assignment = assign_two_bands(roi_clusters, participant_id, role, min_gap=min_gap)
        assignment['roi'] = roi
        rows.append(assignment)
    return pd.DataFrame(rows)


def _iaf_and_slow_freq(participant_rows):
    """Extract (iaf, slow_freq, freq_gap) for one participant from primary/fallback ROI rows."""
    iaf, slow_freq, freq_gap = None, None, None
    for roi in (PRIMARY_IAF_ROI, FALLBACK_IAF_ROI):
        roi_row = participant_rows[participant_rows['roi'] == roi]
        if len(roi_row) == 0:
            continue
        roi_row = roi_row.iloc[0]
        if iaf is None:
            if pd.notna(roi_row['fast_cf']):
                iaf = float(roi_row['fast_cf'])
            elif pd.notna(roi_row['dominant_cf']):
                iaf = float(roi_row['dominant_cf'])
        if slow_freq is None and pd.notna(roi_row['slow_cf']):
            slow_freq = float(roi_row['slow_cf'])
        if freq_gap is None and pd.notna(roi_row['freq_gap']):
            freq_gap = float(roi_row['freq_gap'])
    return iaf, slow_freq, freq_gap


def compute_iaf_metrics(band_assignments_df):
    """Compute individual alpha frequency metrics and dyadic distances.

    For each participant:
    - iaf = fast_cf at parietal ROI (primary), fallback to sensorimotor if parietal missing.
      If only single dominant rhythm, iaf = dominant_cf.
    - slow_freq = slow_cf at parietal ROI (primary), fallback to sensorimotor.
    - freq_gap = freq_gap (fast_cf - slow_cf) at parietal ROI (primary), fallback to
      sensorimotor. None for participants without a two_rhythms assignment.

    For each dyad (matched by dyad_id):
    - iaf_distance = |iaf_caregiver - iaf_child|
    - slow_distance = |slow_freq_caregiver - slow_freq_child|
    - iaf_child_deviation = iaf_child - mean(iaf across all children)
    - iaf_cg_deviation = iaf_caregiver - mean(iaf across all caregivers)

    Parameters
    ----------
    band_assignments_df : pd.DataFrame
        All band assignments across participants and ROIs.
        Must contain: participant_id, role, group, roi, fast_cf, slow_cf,
        dominant_cf, freq_gap, and dyad_id.

    Returns
    -------
    pd.DataFrame
        One row per participant: participant_id, role, group, dyad_id,
        iaf, slow_freq, freq_gap, iaf_distance, slow_distance,
        iaf_child_deviation, iaf_cg_deviation.
    """
    metrics_rows = []
    for pid, participant_rows in band_assignments_df.groupby('participant_id'):
        meta = participant_rows.iloc[0]
        iaf, slow_freq, freq_gap = _iaf_and_slow_freq(participant_rows)
        metrics_rows.append({
            'participant_id': pid,
            'role': meta['role'],
            'group': meta['group'],
            'dyad_id': meta['dyad_id'],
            'iaf': iaf,
            'slow_freq': slow_freq,
            'freq_gap': freq_gap,
        })
    metrics_df = pd.DataFrame(metrics_rows)

    child_mean_iaf = metrics_df.loc[metrics_df['role'] == 'child', 'iaf'].mean()
    caregiver_mean_iaf = metrics_df.loc[metrics_df['role'] == 'caregiver', 'iaf'].mean()
    metrics_df['iaf_child_deviation'] = np.where(
        metrics_df['role'] == 'child', metrics_df['iaf'] - child_mean_iaf, np.nan,
    )
    metrics_df['iaf_cg_deviation'] = np.where(
        metrics_df['role'] == 'caregiver', metrics_df['iaf'] - caregiver_mean_iaf, np.nan,
    )

    child = metrics_df[metrics_df['role'] == 'child'].set_index('dyad_id')
    caregiver = metrics_df[metrics_df['role'] == 'caregiver'].set_index('dyad_id')
    dyad_distances = pd.DataFrame({
        'iaf_distance': (caregiver['iaf'] - child['iaf']).abs(),
        'slow_distance': (caregiver['slow_freq'] - child['slow_freq']).abs(),
    })

    return metrics_df.merge(dyad_distances, left_on='dyad_id', right_index=True, how='left')
