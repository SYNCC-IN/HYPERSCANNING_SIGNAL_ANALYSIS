"""Slow/fast rhythm band assignment from within-ROI peak clusters, and IAF metrics."""

import numpy as np
import pandas as pd

PRIMARY_IAF_ROI = 'parietal'
FALLBACK_IAF_ROI = 'sensorimotor'


def assign_two_bands_kmeans(roi_peaks_df, slow_cf_range=(3.0, 7.5), fast_cf_range=(7.5, 13.0),
                             min_gap=1.5, max_iter=50):
    """Assign slow/fast bands via seeded, power-weighted k-means over specparam peaks.

    Operates on the raw specparam peaks for ONE participant at ONE ROI (all
    channels' peaks pooled), NOT on pre-clustered peaks. This replaces the
    greedy within-ROI clustering for the band-assignment path: greedy
    clustering (tolerance 1.0 Hz) can leave one true oscillation split
    across adjacent peaks, which a subsequent power-based "two strongest"
    selection then discards. Seeded k-means instead reassigns every peak by
    proximity to a power-weighted center of mass, merging noise-split peaks
    around the true center.

    Peaks are first restricted to ``[slow_cf_range[0], fast_cf_range[1]]``;
    peaks outside are discarded. Up to two centers are seeded, one from
    ``slow_cf_range`` and one from ``fast_cf_range``: the seed is the
    candidate peak in that range whose center_freq is closest to the
    range's midpoint (ties broken by higher power, then lower center_freq)
    — not the highest-power peak in the range. A loud peak sitting right at
    the shared boundary is plausible fast-rhythm content mislabeled by
    range membership alone, and seeding from it would anchor the slow
    cluster on the wrong peak; a midpoint-proximity seed instead picks a
    prototype location for the band and lets the nearest-center assignment
    below sort boundary-adjacent peaks into whichever cluster they actually
    belong to. Every kept peak (regardless of which range it originally
    fell in) is then assigned to its nearest center by absolute frequency
    distance, centers are recomputed as the power-weighted mean center_freq
    of their members, and this repeats until membership stops changing or
    ``max_iter`` is reached. Ties in nearest-center go to cluster index 0
    (seeded from the slow range), a fixed, deterministic rule. After
    convergence the lower of the two final centers is labelled slow, the
    higher fast; the power-weighted center is kept as-is even if it drifts
    outside its seeding range.

    If only one seed exists (no peak in the other range), or if the k-means
    partition collapses so every peak lands in one cluster, all kept peaks
    are pooled into a single band, labelled by comparing its final center
    to the shared boundary ``slow_cf_range[1]`` (== ``fast_cf_range[0]``).
    If both centers survive but their final gap is below ``min_gap``, the
    two clusters are deemed one rhythm: only the higher-summed-power
    cluster is kept (its own members' peaks), labelled the same way.

    For each returned band, the window is ``[cf - bw, cf + bw]`` where
    ``cf`` is the power-weighted mean center_freq of its members and
    ``bw = max(cf - min_edge, max_edge - cf)``, with ``min_edge`` /
    ``max_edge`` the min/max of each member's ``center_freq - bandwidth`` /
    ``center_freq + bandwidth`` — so the window covers every merged peak's
    own extent.

    Parameters
    ----------
    roi_peaks_df : pd.DataFrame
        Raw specparam peaks for one participant at one ROI (channels
        pooled). Columns: center_freq, power, bandwidth.
    slow_cf_range, fast_cf_range : tuple of float, optional
        Seeding ranges for the slow and fast centers.
    min_gap : float
        Minimum final separation (Hz) between slow and fast centers.
    max_iter : int
        k-means iteration cap.

    Returns
    -------
    dict
        slow_cf, slow_bw, slow_power, fast_cf, fast_bw, fast_power,
        freq_gap (fast_cf - slow_cf when two_rhythms else None),
        n_peaks_used, n_rhythm_assignment (0, 1, or 2), assignment_note in
        {'two_rhythms', 'single_slow', 'single_fast', 'no_peaks'}. Unused
        band fields are None.
    """
    slow_lo, slow_hi = slow_cf_range
    fast_lo, fast_hi = fast_cf_range

    result = {
        'slow_cf': None, 'slow_bw': None, 'slow_power': None,
        'fast_cf': None, 'fast_bw': None, 'fast_power': None,
        'freq_gap': None,
        'n_peaks_used': 0,
        'n_rhythm_assignment': None,
        'assignment_note': None,
    }

    in_range = roi_peaks_df[
        (roi_peaks_df['center_freq'] >= slow_lo) & (roi_peaks_df['center_freq'] <= fast_hi)
    ]
    cf = in_range['center_freq'].to_numpy()
    power = in_range['power'].to_numpy()
    bw = in_range['bandwidth'].to_numpy()
    result['n_peaks_used'] = len(cf)

    if len(cf) == 0:
        result['n_rhythm_assignment'] = 0
        result['assignment_note'] = 'no_peaks'
        return result

    def _cluster_stats(indices):
        member_cf, member_power, member_bw = cf[indices], power[indices], bw[indices]
        weighted_cf = float(np.sum(member_cf * member_power) / np.sum(member_power))
        min_edge = float(np.min(member_cf - member_bw))
        max_edge = float(np.max(member_cf + member_bw))
        half_width = max(weighted_cf - min_edge, max_edge - weighted_cf)
        return weighted_cf, half_width, float(np.sum(member_power))

    def _set_single_band(indices):
        weighted_cf, half_width, summed_power = _cluster_stats(indices)
        band = 'slow' if weighted_cf <= slow_hi else 'fast'
        result[f'{band}_cf'] = weighted_cf
        result[f'{band}_bw'] = half_width
        result[f'{band}_power'] = summed_power
        result['n_rhythm_assignment'] = 1
        result['assignment_note'] = f'single_{band}'

    def _seed(mask, range_center):
        if not mask.any():
            return None
        idx = np.where(mask)[0]
        # Primary key (last arg) = distance to range center; ties broken by
        # higher power, then by lower center_freq — all deterministic.
        order = np.lexsort((cf[idx], -power[idx], np.abs(cf[idx] - range_center)))
        return cf[idx[order[0]]]

    slow_mask = (cf >= slow_lo) & (cf <= slow_hi)
    fast_mask = (cf >= fast_lo) & (cf <= fast_hi)
    slow_seed = _seed(slow_mask, (slow_lo + slow_hi) / 2.0)
    fast_seed = _seed(fast_mask, (fast_lo + fast_hi) / 2.0)

    if slow_seed is None or fast_seed is None:
        _set_single_band(np.arange(len(cf)))
        return result

    centers = np.array([slow_seed, fast_seed], dtype=float)
    assignment = None
    for _ in range(max_iter):
        dist0 = np.abs(cf - centers[0])
        dist1 = np.abs(cf - centers[1])
        new_assignment = np.where(dist1 < dist0, 1, 0)  # ties -> cluster 0
        if assignment is not None and np.array_equal(new_assignment, assignment):
            break
        assignment = new_assignment
        for k in (0, 1):
            members = np.where(assignment == k)[0]
            if len(members) > 0:
                centers[k] = np.sum(cf[members] * power[members]) / np.sum(power[members])

    members0 = np.where(assignment == 0)[0]
    members1 = np.where(assignment == 1)[0]
    if len(members0) == 0 or len(members1) == 0:
        _set_single_band(np.arange(len(cf)))
        return result

    cf0, bw0, power0 = _cluster_stats(members0)
    cf1, bw1, power1 = _cluster_stats(members1)
    if cf0 <= cf1:
        lo_cf, lo_bw, lo_power = cf0, bw0, power0
        hi_cf, hi_bw, hi_power = cf1, bw1, power1
    else:
        lo_cf, lo_bw, lo_power = cf1, bw1, power1
        hi_cf, hi_bw, hi_power = cf0, bw0, power0

    if hi_cf - lo_cf >= min_gap:
        result['slow_cf'], result['slow_bw'], result['slow_power'] = lo_cf, lo_bw, lo_power
        result['fast_cf'], result['fast_bw'], result['fast_power'] = hi_cf, hi_bw, hi_power
        result['freq_gap'] = hi_cf - lo_cf
        result['n_rhythm_assignment'] = 2
        result['assignment_note'] = 'two_rhythms'
        return result

    # Centers collapsed together: keep only the higher-power cluster as one band.
    winner_indices = members0 if power0 >= power1 else members1
    _set_single_band(winner_indices)
    return result


def assign_bands_all_rois(participant_peaks_df, participant_id, role, roi_channels, min_gap=1.5,
                           slow_cf_range=(3.0, 7.5), fast_cf_range=(7.5, 13.0), max_iter=50):
    """Run assign_two_bands_kmeans for each ROI for one participant.

    Parameters
    ----------
    participant_peaks_df : pd.DataFrame
        Raw specparam peaks for one participant, all channels pooled.
        Columns: channel, center_freq, power, bandwidth.
    participant_id : str
        Participant identifier.
    role : str
        'child' or 'caregiver'.
    roi_channels : dict
        Mapping {roi_label: [channel_names]}.
    min_gap : float
        Passed to assign_two_bands_kmeans.
    slow_cf_range, fast_cf_range : tuple of float, optional
        Passed to assign_two_bands_kmeans.
    max_iter : int, optional
        Passed to assign_two_bands_kmeans.

    Returns
    -------
    pd.DataFrame
        One row per ROI with all fields from assign_two_bands_kmeans plus
        participant_id, role, and roi columns.
    """
    rows = []
    for roi, channels in roi_channels.items():
        roi_peaks = participant_peaks_df[participant_peaks_df['channel'].isin(channels)]
        assignment = assign_two_bands_kmeans(
            roi_peaks, slow_cf_range=slow_cf_range, fast_cf_range=fast_cf_range,
            min_gap=min_gap, max_iter=max_iter,
        )
        assignment['participant_id'] = participant_id
        assignment['role'] = role
        assignment['roi'] = roi
        rows.append(assignment)
    return pd.DataFrame(rows)


def _iaf_and_slow_freq(participant_rows):
    """Extract (iaf, slow_freq, freq_gap) for one participant from primary/fallback ROI rows.

    iaf comes directly from fast_cf (populated for two_rhythms and
    single_fast assignments); slow_freq comes directly from slow_cf
    (populated for two_rhythms and single_slow assignments). A single_slow
    assignment therefore leaves iaf undefined, and a single_fast assignment
    leaves slow_freq undefined, rather than borrowing across bands.
    """
    iaf, slow_freq, freq_gap = None, None, None
    for roi in (PRIMARY_IAF_ROI, FALLBACK_IAF_ROI):
        roi_row = participant_rows[participant_rows['roi'] == roi]
        if len(roi_row) == 0:
            continue
        roi_row = roi_row.iloc[0]
        if iaf is None and pd.notna(roi_row['fast_cf']):
            iaf = float(roi_row['fast_cf'])
        if slow_freq is None and pd.notna(roi_row['slow_cf']):
            slow_freq = float(roi_row['slow_cf'])
        if freq_gap is None and pd.notna(roi_row['freq_gap']):
            freq_gap = float(roi_row['freq_gap'])
    return iaf, slow_freq, freq_gap


def compute_iaf_metrics(band_assignments_df):
    """Compute individual alpha frequency metrics and dyadic distances.

    For each participant:
    - iaf = fast_cf at parietal ROI (primary), fallback to sensorimotor if parietal missing.
      For single_slow participants, fast_cf is unpopulated so iaf is left
      undefined (NaN) rather than borrowing the slow value.
    - slow_freq = slow_cf at parietal ROI (primary), fallback to sensorimotor.
      For single_fast participants, slow_cf is unpopulated so slow_freq is
      left undefined (NaN).
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
        freq_gap, and dyad_id.

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
