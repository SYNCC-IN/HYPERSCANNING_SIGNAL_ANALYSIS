"""Plotting and figure-generation functions shared across the exploratory spectral pipeline."""

import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

# Consistent color/linestyle conventions used across all artifacts.
COLORS = {
    'TD': '#3b6fa0',
    'ASD': '#c0392b',
    'child': '#2c3e50',
    'caregiver': '#7f8c8d',
}
ROLE_LINESTYLE = {'child': '-', 'caregiver': '--'}


def make_montage_info(channel_names, sfreq=128.0, montage_name='standard_1020'):
    """Build an MNE Info object with standard 10-20 channel positions.

    Parameters
    ----------
    channel_names : list of str
        EEG channel names to include.
    sfreq : float, optional
        Sampling frequency in Hz (arbitrary for topomap purposes).
    montage_name : str, optional
        Name of the MNE montage to use for electrode positions.

    Returns
    -------
    mne.Info
        Info object with the standard montage applied, usable with
        ``mne.viz.plot_topomap``.
    """
    info = mne.create_info(ch_names=list(channel_names), sfreq=sfreq, ch_types='eeg')
    montage = mne.channels.make_standard_montage(montage_name)
    info.set_montage(montage, on_missing='ignore')
    return info


def plot_peak_prevalence_topomap(prevalence_series, channel_names, title, vmin=0, vmax=1, ax=None):
    """Plot a topomap of peak-detection prevalence across channels.

    Parameters
    ----------
    prevalence_series : array-like, shape (n_channels,)
        Prevalence values (0-1) in the same order as ``channel_names``.
    channel_names : list of str
        Channel names matching ``prevalence_series``.
    title : str
        Axes title.
    vmin, vmax : float, optional
        Color scale limits.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw into. A new figure is created if None.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the topomap.
    """
    info = make_montage_info(channel_names)
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    else:
        fig = ax.figure
    im, _ = mne.viz.plot_topomap(
        np.asarray(prevalence_series, dtype=float), info, axes=ax, show=False,
        vlim=(vmin, vmax), cmap='viridis', contours=0,
    )
    ax.set_title(title, fontsize=9)
    return fig


def plot_peak_freq_histogram(peaks_df, channel_cluster, role, group=None, bin_width=0.5, freq_range=(3, 25), ax=None):
    """Plot a histogram of detected peak center frequencies for a channel cluster.

    Parameters
    ----------
    peaks_df : pandas.DataFrame
        Peak inventory table with columns ``channel_cluster``, ``role``, ``group``,
        ``center_freq``.
    channel_cluster : str
        Cluster label to filter on (e.g. ``'frontal'``).
    role : str
        Role to filter on (``'child'`` or ``'caregiver'``).
    group : str or None, optional
        If given, restrict to this diagnostic group only (e.g. ``'TD'``). If None,
        overlay TD and ASD groups with different colors.
    bin_width : float, optional
        Histogram bin width in Hz.
    freq_range : tuple of float, optional
        ``(low, high)`` histogram range in Hz. Should match the ``freq_range``
        used for the upstream specparam fit, so out-of-range peaks aren't
        silently clipped from the plot.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw into. A new figure is created if None.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the histogram.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))
    else:
        fig = ax.figure

    subset = peaks_df[(peaks_df['channel_cluster'] == channel_cluster) & (peaks_df['role'] == role)]
    bins = np.arange(freq_range[0], freq_range[1] + bin_width, bin_width)

    groups_to_plot = [group] if group is not None else ['TD', 'ASD']
    for grp in groups_to_plot:
        vals = subset.loc[subset['group'] == grp, 'center_freq'].dropna().values
        ax.hist(vals, bins=bins, alpha=0.5, color=COLORS.get(grp, 'gray'), label=grp, density=False)

    ax.set_xlabel('Center frequency (Hz)')
    ax.set_ylabel('Peak count')
    ax.set_title(f'{channel_cluster} / {role}', fontsize=9)
    ax.legend(fontsize=7)
    return fig


def plot_peak_freq_topomap_individual(peaks_df, channel_names, participant_id, f_range, ax=None):
    """Plot a topomap of the strongest peak's center frequency per channel.

    Channels with no detected peak are shown in gray.

    Parameters
    ----------
    peaks_df : pandas.DataFrame
        Peak inventory table for a single participant, with columns ``channel``,
        ``center_freq``, ``power``.
    channel_names : list of str
        Full channel montage to plot (defines topomap layout).
    participant_id : str
        Participant identifier, used in the title.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw into. A new figure is created if None.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the topomap.
    """
    info = make_montage_info(channel_names)
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    else:
        fig = ax.figure

    strongest = (
        peaks_df.sort_values('power', ascending=False)
        .drop_duplicates(subset='channel', keep='first')
        .set_index('channel')
    )
    values = np.array([strongest['center_freq'].get(ch, np.nan) for ch in channel_names])
    mask = ~np.isnan(values)
    plot_values = np.where(mask, values, 0.0)

    im, _ = mne.viz.plot_topomap(
        plot_values, info, axes=ax, show=False, cmap='plasma',
        vlim=f_range, contours=0, mask=mask,
        mask_params=dict(marker='o', markerfacecolor='none', markeredgecolor='k', markersize=4),
    )
    colorbar = plt.colorbar(im, ax=ax, shrink=0.7)
    colorbar.set_label('Peak center frequency (Hz)', fontsize=9)
    ax.set_title(f'{participant_id}: peak center frequency', fontsize=9)
    return fig


def plot_peak_freq_vs_age(peaks_df, roi_channels, role='child', ax=None):
    """Scatter plot of strongest peak center frequency versus age, colored by group.

    Parameters
    ----------
    peaks_df : pandas.DataFrame
        Peak inventory table with columns ``channel``, ``role``, ``group``,
        ``age_months``, ``center_freq``, ``power``, ``participant_id``.
    roi_channels : list of str
        Channels to pool over (strongest peak per participant across these channels).
    role : str, optional
        Role to filter on.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw into. A new figure is created if None.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the scatter plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.figure

    subset = peaks_df[(peaks_df['role'] == role) & (peaks_df['channel'].isin(roi_channels))]
    strongest = (
        subset.sort_values('power', ascending=False)
        .drop_duplicates(subset='participant_id', keep='first')
    )

    for grp in ['TD', 'ASD']:
        grp_data = strongest[strongest['group'] == grp]
        ax.scatter(
            grp_data['age_months'] / 12.0, grp_data['center_freq'],
            color=COLORS.get(grp, 'gray'), label=grp, alpha=0.75, s=30,
        )
        if len(grp_data) >= 3:
            x = (grp_data['age_months'] / 12.0).values
            y = grp_data['center_freq'].values
            coeffs = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 50)
            ax.plot(x_line, np.polyval(coeffs, x_line), color=COLORS.get(grp, 'gray'), lw=1.5)

    ax.set_xlabel('Age (years)')
    ax.set_ylabel('Peak center frequency (Hz)')
    ax.set_title(f'Peak frequency vs. age ({role}, {"/".join(roi_channels)})', fontsize=9)
    ax.legend(fontsize=8)
    return fig


def plot_peak_cluster_topomaps(cluster_df, peaks_df, channel_names, participant_id, role, group, max_clusters=6):
    """Plot topomaps for each peak cluster of a single participant.

    One subplot per cluster. Color = peak power (above aperiodic) at channels
    belonging to that cluster. Channels not in the cluster are masked (gray).
    Subplot title: mean center_freq, n_channels, robust/isolated label.

    Parameters
    ----------
    cluster_df : pd.DataFrame
        Output of :func:`src.peaks.cluster_peaks_across_channels` for this
        participant.
    peaks_df : pd.DataFrame
        Original peaks for this participant (to get per-channel power values).
    channel_names : list of str
        All channel names in the montage (for topomap layout).
    participant_id : str
        For figure title.
    role : str
        'child' or 'caregiver', for figure title.
    group : str
        'TD' or 'ASD', for figure title.
    max_clusters : int, optional
        Maximum number of clusters (lowest frequency first) to plot.

    Returns
    -------
    matplotlib.figure.Figure
    """
    info = make_montage_info(channel_names)
    clusters = cluster_df.sort_values('mean_center_freq').head(max_clusters)
    n_panels = max(len(clusters), 1)

    fig, axes = plt.subplots(1, n_panels, figsize=(3 * n_panels, 3.5))
    axes = np.atleast_1d(axes)

    if len(clusters) == 0:
        axes[0].text(0.5, 0.5, 'No peaks detected', ha='center', va='center', fontsize=9)
        axes[0].axis('off')

    for ax, (_, cluster) in zip(axes, clusters.iterrows()):
        cluster_channels = cluster['channel_list'].split(',')
        power_by_channel = peaks_df[peaks_df['channel'].isin(cluster_channels)].groupby('channel')['power'].max()
        values = np.array([power_by_channel.get(ch, 0.0) for ch in channel_names])
        mask = np.array([ch in cluster_channels for ch in channel_names])

        mne.viz.plot_topomap(
            values, info, axes=ax, show=False, cmap='YlOrRd', contours=0, mask=mask,
            mask_params=dict(marker='o', markerfacecolor='none', markeredgecolor='k', markersize=4),
        )
        robust_label = 'robust' if cluster['is_robust'] else 'isolated'
        ax.set_title(f"{cluster['mean_center_freq']:.1f} Hz, n_ch={cluster['n_channels']} ({robust_label})", fontsize=8)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('black' if cluster['is_robust'] else 'gray')
            spine.set_linestyle('-' if cluster['is_robust'] else '--')
            spine.set_linewidth(1.5)

    fig.suptitle(f'{participant_id} ({role}, {group}) — peak clusters', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    return fig


ROI_ORDER_DEFAULT = ['frontal-midline', 'frontal-lateral', 'central', 'parietal', 'occipital', 'temporal']


def plot_roi_cluster_scatter(clusters_df, group_colors, roi_order=None, freq_range=(3, 14)):
    """Beeswarm-style scatter of within-ROI peak clusters.

    Grid layout: columns = ROI, rows = role (child, caregiver). Each point
    is one cluster from one participant. X-axis is never jittered; a small
    random y-jitter (~2% of each row's power range) is added to reduce
    overplotting.

    Parameters
    ----------
    clusters_df : pd.DataFrame
        All clusters across all participants. Must contain columns:
        participant_id, role, group, roi, weighted_center_freq,
        summed_power, n_channels.
    group_colors : dict
        Mapping {group_label: color}, e.g. {'TD': '#4477AA', 'ASD': '#CC6677'}.
    roi_order : list of str or None, optional
        Column order for ROIs. Defaults to ROI_ORDER_DEFAULT.
    freq_range : tuple of float, optional
        Shared x-axis limits (Hz) across all panels.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if roi_order is None:
        roi_order = ROI_ORDER_DEFAULT
    roles = ['child', 'caregiver']
    rng = np.random.default_rng(42)

    fig, axes = plt.subplots(len(roles), len(roi_order), figsize=(2.8 * len(roi_order), 5.5), sharex=True)

    for row, role in enumerate(roles):
        role_clusters = clusters_df[clusters_df['role'] == role]
        power = role_clusters['summed_power']
        y_min, y_max = (power.min(), power.max()) if len(power) else (0.0, 1.0)
        y_span = y_max - y_min if y_max > y_min else 1.0
        jitter_scale = 0.02 * y_span
        y_pad = 0.05 * y_span

        for col, roi in enumerate(roi_order):
            ax = axes[row, col]
            roi_clusters = role_clusters[role_clusters['roi'] == roi]
            for grp, color in group_colors.items():
                grp_clusters = roi_clusters[roi_clusters['group'] == grp]
                y_jitter = rng.uniform(-jitter_scale, jitter_scale, size=len(grp_clusters))
                ax.scatter(
                    grp_clusters['weighted_center_freq'], grp_clusters['summed_power'] + y_jitter,
                    color=color, alpha=0.6, s=25, edgecolor='none',
                )
            ax.set_ylim(y_min - y_pad, y_max + y_pad)
            if row == 0:
                ax.set_title(roi, fontsize=9)
            if col == 0:
                ax.set_ylabel(f'{role}\nsummed power', fontsize=8)
            if row == len(roles) - 1:
                ax.set_xlabel('Weighted CF (Hz)', fontsize=7)

    axes[0, 0].set_xlim(freq_range)
    legend_handles = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor=color, markersize=6, label=grp)
        for grp, color in group_colors.items()
    ]
    fig.legend(handles=legend_handles, loc='upper right', fontsize=8)
    fig.suptitle('Within-ROI peak clusters: frequency vs. summed power')
    fig.tight_layout(rect=[0, 0, 0.95, 0.93])
    return fig


def plot_peak_count_histogram(peak_counts_df, rois):
    """Histogram of number of peaks per participant per ROI.

    Grid: columns = ROIs (sensorimotor, parietal — or whichever are passed),
    rows = role (child, caregiver).
    X-axis: number of peaks (0, 1, 2, 3, 4+), discrete bins.
    Bars colored by group (TD / ASD), side by side.

    Parameters
    ----------
    peak_counts_df : pd.DataFrame
        Columns: participant_id, role, group, roi, n_peaks.
    rois : list of str
        ROI labels to include as columns.

    Returns
    -------
    matplotlib.figure.Figure
    """
    roles = ['child', 'caregiver']
    groups = ['TD', 'ASD']
    bin_labels = ['0', '1', '2', '3', '4+']
    x = np.arange(len(bin_labels))
    width = 0.35

    fig, axes = plt.subplots(len(roles), len(rois), figsize=(4 * len(rois), 6), sharey=True)
    axes = np.atleast_2d(axes)

    for row, role in enumerate(roles):
        for col, roi in enumerate(rois):
            ax = axes[row, col]
            subset = peak_counts_df[(peak_counts_df['role'] == role) & (peak_counts_df['roi'] == roi)]
            for i, grp in enumerate(groups):
                clipped = subset.loc[subset['group'] == grp, 'n_peaks'].clip(upper=4)
                counts = clipped.value_counts().reindex(range(5), fill_value=0).sort_index()
                ax.bar(x + (i - 0.5) * width, counts.values, width=width, color=COLORS.get(grp, 'gray'), label=grp)
            ax.set_xticks(x)
            ax.set_xticklabels(bin_labels)
            if row == 0:
                ax.set_title(roi, fontsize=10)
            if col == 0:
                ax.set_ylabel(f'{role}\nparticipant count', fontsize=8)
            if row == len(roles) - 1:
                ax.set_xlabel('Number of peaks', fontsize=8)
            ax.legend(fontsize=7)

    fig.suptitle('Peak count per participant by ROI, role, and group')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def plot_two_peak_scatter(two_peak_df):
    """Scatter of peak 1 freq vs peak 2 freq for participants with exactly 2 peaks in a ROI.

    Grid: columns = ROIs, rows = role.
    X-axis: center_freq of the lower-frequency peak (peak 1).
    Y-axis: center_freq of the higher-frequency peak (peak 2).
    Color: group (TD / ASD).
    Diagonal line y=x for reference.
    Each point is annotated with its frequency gap (peak2 - peak1) in Hz.

    Only include participants who have exactly 2 peaks within the ROI.
    If a participant has 3+ peaks, take the two strongest by power.

    Parameters
    ----------
    two_peak_df : pd.DataFrame
        Columns: participant_id, role, group, roi, peak1_freq, peak2_freq,
        peak1_power, peak2_power, freq_gap.

    Returns
    -------
    matplotlib.figure.Figure
    """
    roles = ['child', 'caregiver']
    rois = sorted(two_peak_df['roi'].unique()) if len(two_peak_df) else ['roi']

    all_freqs = pd.concat([two_peak_df['peak1_freq'], two_peak_df['peak2_freq']]) if len(two_peak_df) else None
    lo, hi = (all_freqs.min(), all_freqs.max()) if all_freqs is not None and len(all_freqs) else (3, 14)
    pad = 0.05 * (hi - lo) if hi > lo else 1.0

    fig, axes = plt.subplots(len(roles), len(rois), figsize=(4 * len(rois), 8), squeeze=False)

    for row, role in enumerate(roles):
        for col, roi in enumerate(rois):
            ax = axes[row, col]
            subset = two_peak_df[(two_peak_df['role'] == role) & (two_peak_df['roi'] == roi)]
            for grp in ['TD', 'ASD']:
                grp_subset = subset[subset['group'] == grp]
                ax.scatter(
                    grp_subset['peak1_freq'], grp_subset['peak2_freq'],
                    color=COLORS.get(grp, 'gray'), label=grp, alpha=0.7, s=30, edgecolor='none',
                )
                for _, prow in grp_subset.iterrows():
                    ax.annotate(
                        f"{prow['freq_gap']:.1f}", (prow['peak1_freq'], prow['peak2_freq']),
                        fontsize=6, xytext=(3, 3), textcoords='offset points', color='gray',
                    )
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color='black', linestyle='--', lw=1, alpha=0.5)
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
            if row == 0:
                ax.set_title(roi, fontsize=10)
            if col == 0:
                ax.set_ylabel(f'{role}\npeak 2 freq (Hz)', fontsize=8)
            if row == len(roles) - 1:
                ax.set_xlabel('peak 1 freq (Hz)', fontsize=8)
            ax.legend(fontsize=7)

    fig.suptitle('Two-peak participants: peak 1 vs. peak 2 frequency (label = gap in Hz)')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def plot_individual_peak_profiles(peaks_df, roi_channels, roi_label, role='child'):
    """Strip plot showing each participant's detected peaks within one ROI.

    Y-axis: participant_id (sorted by lowest peak frequency).
    X-axis: center frequency (Hz).
    Each peak is a dot at its center_freq. Dot size proportional to power.
    Color by group (TD / ASD).

    This directly visualizes whether bimodality is within- or between-subject:
    - If between-subject: each row has one dot, but dots cluster at different
      x positions for different participants.
    - If within-subject: rows have two dots — one low, one high.

    Parameters
    ----------
    peaks_df : pd.DataFrame
        Peaks for the relevant ROI and role, with columns ``participant_id``,
        ``group``, ``channel``, ``center_freq``, ``power``. To avoid
        double-counting the same oscillation seen at multiple electrodes,
        callers should pass within-ROI clustered peaks (one row per
        oscillation) rather than raw per-channel peaks.
    roi_channels : list of str
        Channels in this ROI.
    roi_label : str
        For title.
    role : str
        'child' or 'caregiver'.

    Returns
    -------
    matplotlib.figure.Figure
    """
    subset = peaks_df[peaks_df['channel'].isin(roi_channels)]

    lowest_freq = subset.groupby('participant_id')['center_freq'].min().sort_values()
    pid_order = lowest_freq.index.tolist()
    y_pos = {pid: i for i, pid in enumerate(pid_order)}

    fig, ax = plt.subplots(figsize=(6, max(4, 0.18 * len(pid_order))))
    for grp in ['TD', 'ASD']:
        grp_subset = subset[subset['group'] == grp]
        y = grp_subset['participant_id'].map(y_pos)
        sizes = grp_subset['power'] * 60 + 10
        ax.scatter(
            grp_subset['center_freq'], y, s=sizes,
            color=COLORS.get(grp, 'gray'), label=grp, alpha=0.7, edgecolor='none',
        )

    ax.set_yticks(range(len(pid_order)))
    ax.set_yticklabels(pid_order, fontsize=5)
    ax.set_xlabel('Center frequency (Hz)')
    ax.set_xlim((3,15))
    ax.set_title(f'{roi_label} ({role}): individual peak profiles', fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_band_assignment_strips(band_assignments_df, rois):
    """Strip plot of each participant's slow/fast (or dominant) rhythm assignment.

    Grid layout: columns = ROI, rows = role (child, caregiver). For each
    participant: slow_cf is plotted as a triangle, fast_cf as a circle,
    joined by a horizontal line segment; single-dominant participants are
    plotted as a diamond at dominant_cf. Y-axis is participant_id, sorted by
    fast_cf (falling back to dominant_cf for single-dominant participants).
    Participants with no peaks are omitted.

    Only participants in the TD or ASD groups are plotted; other group
    labels (e.g. secondary cohorts not part of the TD/ASD comparison) are
    excluded since they have no defined color in COLORS.

    Parameters
    ----------
    band_assignments_df : pd.DataFrame
        Output of assign_bands_all_rois concatenated across participants.
        Must contain columns: participant_id, role, group, roi, slow_cf,
        fast_cf, dominant_cf, assignment_note.
    rois : list of str
        ROI labels to include as columns.

    Returns
    -------
    matplotlib.figure.Figure
    """
    band_assignments_df = band_assignments_df[band_assignments_df['group'].isin(['TD', 'ASD'])]
    roles = ['child', 'caregiver']
    max_rows = band_assignments_df.groupby(['role', 'roi'])['participant_id'].nunique().max()
    fig_height = max(4, 0.13 * max_rows) * len(roles)

    fig, axes = plt.subplots(len(roles), len(rois), figsize=(5 * len(rois), fig_height), squeeze=False)

    for row, role in enumerate(roles):
        for col, roi in enumerate(rois):
            ax = axes[row, col]
            subset = band_assignments_df[
                (band_assignments_df['role'] == role) & (band_assignments_df['roi'] == roi)
            ].copy()
            subset['sort_freq'] = subset['fast_cf'].fillna(subset['dominant_cf'])
            subset = subset.dropna(subset=['sort_freq']).sort_values('sort_freq')
            pid_order = subset['participant_id'].tolist()
            y_pos = {pid: i for i, pid in enumerate(pid_order)}

            for _, prow in subset.iterrows():
                y = y_pos[prow['participant_id']]
                color = COLORS.get(prow['group'], 'gray')
                if prow['assignment_note'] == 'two_rhythms':
                    ax.plot([prow['slow_cf'], prow['fast_cf']], [y, y], color=color, lw=1, alpha=0.5, zorder=1)
                    ax.scatter(prow['slow_cf'], y, marker='^', color=color, s=25, zorder=2)
                    ax.scatter(prow['fast_cf'], y, marker='o', color=color, s=25, zorder=2)
                elif prow['assignment_note'] == 'single_dominant':
                    ax.scatter(prow['dominant_cf'], y, marker='D', color=color, s=25, zorder=2)

            ax.set_yticks(range(len(pid_order)))
            ax.set_yticklabels(pid_order, fontsize=4)
            if row == 0:
                ax.set_title(roi, fontsize=10)
            if col == 0:
                ax.set_ylabel(role, fontsize=9)
            if row == len(roles) - 1:
                ax.set_xlabel('Center frequency (Hz)', fontsize=8)

    marker_handles = [
        Line2D([0], [0], marker='^', color='gray', linestyle='none', markersize=6, label='slow'),
        Line2D([0], [0], marker='o', color='gray', linestyle='none', markersize=6, label='fast'),
        Line2D([0], [0], marker='D', color='gray', linestyle='none', markersize=6, label='dominant'),
    ]
    group_handles = [
        Line2D([0], [0], marker='s', color='none', markerfacecolor=color, markersize=8, label=grp)
        for grp in ('TD', 'ASD') for color in [COLORS.get(grp, 'gray')]
    ]
    fig.legend(handles=marker_handles + group_handles, loc='upper right', fontsize=7)
    fig.suptitle('Slow/fast rhythm assignment by ROI, role, and group')
    fig.tight_layout(rect=[0, 0, 0.9, 0.96])
    return fig


def plot_iaf_distance_by_group(iaf_metrics_df):
    """Boxplot + strip overlay of dyadic IAF and slow-rhythm distance by group.

    Left panel: iaf_distance (|fast caregiver - fast child|) by group.
    Right panel: slow_distance (|slow caregiver - slow child|) by group.

    Parameters
    ----------
    iaf_metrics_df : pd.DataFrame
        Output of compute_iaf_metrics. One row per participant; iaf_distance
        and slow_distance are dyad-level values duplicated across both
        members of a dyad, so this function deduplicates by dyad_id first.

    Returns
    -------
    matplotlib.figure.Figure
    """
    dyad_df = iaf_metrics_df.drop_duplicates(subset='dyad_id')
    groups = ['TD', 'ASD']
    rng = np.random.default_rng(42)

    fig, axes = plt.subplots(1, 2, figsize=(9, 5))
    metrics = [('iaf_distance', 'Fast-rhythm (IAF) distance'), ('slow_distance', 'Slow-rhythm distance')]
    for ax, (metric, title) in zip(axes, metrics):
        data = [dyad_df.loc[dyad_df['group'] == grp, metric].dropna().values for grp in groups]
        bp = ax.boxplot(data, tick_labels=groups, showfliers=False, patch_artist=True)
        for patch, grp in zip(bp['boxes'], groups):
            patch.set_facecolor(COLORS.get(grp, 'gray'))
            patch.set_alpha(0.4)
        for i, (grp, vals) in enumerate(zip(groups, data), start=1):
            jitter = rng.uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals, color=COLORS.get(grp, 'gray'), alpha=0.7, s=20, zorder=3)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel('Absolute frequency distance (Hz)')

    fig.suptitle('Dyadic child-caregiver frequency distance by group')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


def plot_gap_distribution(band_assignments_df, min_gap):
    """Histogram of slow/fast frequency gap for two-rhythm participants.

    Parameters
    ----------
    band_assignments_df : pd.DataFrame
        Output of assign_bands_all_rois concatenated across participants and
        ROIs. Must contain columns: role, group, freq_gap, assignment_note.
    min_gap : float
        The min_gap threshold used for two-rhythm assignment, marked as a
        vertical reference line.

    Returns
    -------
    matplotlib.figure.Figure
    """
    two_rhythms = band_assignments_df[band_assignments_df['assignment_note'] == 'two_rhythms']
    roles = ['child', 'caregiver']
    max_gap = two_rhythms['freq_gap'].max() if len(two_rhythms) else min_gap * 2
    bins = np.arange(0, max_gap + 0.5, 0.25)

    fig, axes = plt.subplots(1, len(roles), figsize=(5 * len(roles), 4), sharex=True, sharey=True)
    for ax, role in zip(axes, roles):
        subset = two_rhythms[two_rhythms['role'] == role]
        for grp in ['TD', 'ASD']:
            vals = subset.loc[subset['group'] == grp, 'freq_gap'].dropna()
            ax.hist(vals, bins=bins, alpha=0.55, color=COLORS.get(grp, 'gray'), label=grp)
        ax.axvline(min_gap, color='black', linestyle='--', lw=1.2, label=f'min_gap = {min_gap}')
        ax.set_title(role, fontsize=10)
        ax.set_xlabel('Frequency gap (Hz)')
        ax.legend(fontsize=7)

    axes[0].set_ylabel('Count')
    fig.suptitle('Slow-fast frequency gap distribution (two-rhythm participants)')
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


def plot_dyad_gap_scatter(iaf_metrics_df):
    """Scatter of each dyad's slow-fast frequency gap: caregiver vs. child.

    X-axis: caregiver's freq_gap (Hz). Y-axis: child's freq_gap (Hz).
    Color: dyad group (TD / ASD). Diagonal line y=x for reference.
    Only dyads where both members have a defined freq_gap (i.e. a
    two_rhythms assignment) are plotted.

    Parameters
    ----------
    iaf_metrics_df : pd.DataFrame
        Output of compute_iaf_metrics. Must contain columns: dyad_id, role,
        group, freq_gap.

    Returns
    -------
    matplotlib.figure.Figure
    """
    child = iaf_metrics_df[iaf_metrics_df['role'] == 'child'].set_index('dyad_id')
    caregiver = iaf_metrics_df[iaf_metrics_df['role'] == 'caregiver'].set_index('dyad_id')

    dyads = pd.DataFrame({
        'child_gap': child['freq_gap'],
        'caregiver_gap': caregiver['freq_gap'],
        'group': child['group'],
    }).dropna(subset=['child_gap', 'caregiver_gap'])
    dyads = dyads[dyads['group'].isin(['TD', 'ASD'])]

    lo = min(dyads['child_gap'].min(), dyads['caregiver_gap'].min())
    hi = max(dyads['child_gap'].max(), dyads['caregiver_gap'].max())
    pad = 0.05 * (hi - lo) if hi > lo else 1.0

    fig, ax = plt.subplots(figsize=(6, 6))
    for grp in ['TD', 'ASD']:
        grp_dyads = dyads[dyads['group'] == grp]
        ax.scatter(
            grp_dyads['caregiver_gap'], grp_dyads['child_gap'],
            color=COLORS.get(grp, 'gray'), label=grp, alpha=0.7, s=30, edgecolor='none',
        )
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color='black', linestyle='--', lw=1, alpha=0.5)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel('Caregiver slow-fast gap (Hz)')
    ax.set_ylabel('Child slow-fast gap (Hz)')
    ax.set_title('Dyadic slow-fast frequency gap: caregiver vs. child')
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_roi_validation_heatmap(roi_validation_df, min_prevalence=0.5):
    """Heatmap of ROI peak-prevalence validation, per band and subgroup.

    Rows = ROI, columns = band x role x group prevalence values. Cells are
    colored by prevalence (0-1); cells below min_prevalence are outlined in
    red.

    Parameters
    ----------
    roi_validation_df : pd.DataFrame
        Output of the ROI validation loop. Must contain a 'ROI' column plus
        prevalence columns named '{band}_prev_{role}_{group}' for
        band in ('slow', 'fast'), role in ('child', 'cg'), group in ('TD', 'ASD').
    min_prevalence : float, optional
        Threshold marked with a red cell outline.

    Returns
    -------
    matplotlib.figure.Figure
    """
    value_cols = [
        'slow_prev_child_TD', 'slow_prev_child_ASD', 'slow_prev_cg_TD', 'slow_prev_cg_ASD',
        'fast_prev_child_TD', 'fast_prev_child_ASD', 'fast_prev_cg_TD', 'fast_prev_cg_ASD',
    ]
    data = roi_validation_df.set_index('ROI')[value_cols]

    fig, ax = plt.subplots(figsize=(1.3 * len(value_cols), 0.6 * len(data) + 2))
    im = ax.imshow(data.values, cmap='viridis', vmin=0, vmax=1, aspect='auto')

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data.values[i, j]
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8,
                    color='white' if val < 0.6 else 'black')
            if val < min_prevalence:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor='red', lw=2))

    ax.set_xticks(range(len(value_cols)))
    ax.set_xticklabels(value_cols, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(data.index, fontsize=9)
    fig.colorbar(im, ax=ax, label='Prevalence', shrink=0.8)
    ax.set_title(f'ROI peak prevalence by band, role, and group (red = below {min_prevalence:.0%})', fontsize=10)
    fig.tight_layout()
    return fig


def plot_roi_validation_heatmap_individual(prevalence_df, min_prevalence=0.5, title_suffix='(individualized bands)'):
    """Heatmap of ROI peak-prevalence validation using individualized band windows.

    Same layout as :func:`plot_roi_validation_heatmap`: rows = ROI, columns =
    band x role x group prevalence values. Cells are colored by prevalence
    (0-1); cells below ``min_prevalence`` are outlined in red.

    Parameters
    ----------
    prevalence_df : pd.DataFrame
        Columns: roi, slow_prev_child_TD, slow_prev_child_ASD,
        slow_prev_cg_TD, slow_prev_cg_ASD, fast_prev_child_TD,
        fast_prev_child_ASD, fast_prev_cg_TD, fast_prev_cg_ASD.
    min_prevalence : float, optional
        Threshold marked with a red cell outline.
    title_suffix : str, optional
        Appended to the figure title to distinguish from the fixed-window version.

    Returns
    -------
    matplotlib.figure.Figure
    """
    value_cols = [
        'slow_prev_child_TD', 'slow_prev_child_ASD', 'slow_prev_cg_TD', 'slow_prev_cg_ASD',
        'fast_prev_child_TD', 'fast_prev_child_ASD', 'fast_prev_cg_TD', 'fast_prev_cg_ASD',
    ]
    data = prevalence_df.set_index('roi')[value_cols]

    fig, ax = plt.subplots(figsize=(1.3 * len(value_cols), 0.6 * len(data) + 2))
    im = ax.imshow(data.values, cmap='viridis', vmin=0, vmax=1, aspect='auto')

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data.values[i, j]
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8,
                    color='white' if val < 0.6 else 'black')
            if val < min_prevalence:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor='red', lw=2))

    ax.set_xticks(range(len(value_cols)))
    ax.set_xticklabels(value_cols, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(data.index, fontsize=9)
    fig.colorbar(im, ax=ax, label='Prevalence', shrink=0.8)
    ax.set_title(
        f'ROI peak prevalence by band, role, and group {title_suffix} (red = below {min_prevalence:.0%})',
        fontsize=10,
    )
    fig.tight_layout()
    return fig


def plot_survival_rate_bars(summary_df, threshold=0.6):
    """Bar chart of ROI x band peak-survival rate, by role and group.

    Two-panel figure (children | caregivers), stacked vertically. X-axis:
    ROI x band category. Bars grouped by group (TD / ASD), side by side.
    Horizontal line at threshold marks the reliability cutoff for DTF.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Output of the peak-survival summary aggregation. Must contain
        columns: roi, band, role, group, survival_rate.
    threshold : float, optional
        Reliability cutoff, drawn as a horizontal reference line.

    Returns
    -------
    matplotlib.figure.Figure
    """
    roles = ['child', 'caregiver']
    categories = summary_df[['roi', 'band']].drop_duplicates().sort_values(['roi', 'band'])
    cat_labels = [f'{roi}\n{band}' for roi, band in categories.itertuples(index=False)]
    x = np.arange(len(cat_labels))
    width = 0.35

    fig, axes = plt.subplots(len(roles), 1, figsize=(1.6 * len(cat_labels), 8), sharex=True)
    for ax, role in zip(axes, roles):
        for i, grp in enumerate(['TD', 'ASD']):
            vals = []
            for roi, band in categories.itertuples(index=False):
                row = summary_df[
                    (summary_df['roi'] == roi) & (summary_df['band'] == band)
                    & (summary_df['role'] == role) & (summary_df['group'] == grp)
                ]
                vals.append(row['survival_rate'].iloc[0] if len(row) else np.nan)
            ax.bar(x + (i - 0.5) * width, vals, width=width, color=COLORS.get(grp, 'gray'), label=grp)
        ax.axhline(threshold, color='black', linestyle='--', lw=1, label=f'threshold = {threshold:.0%}')
        ax.set_ylabel(f'{role}\nsurvival rate')
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(cat_labels, fontsize=8)
    fig.suptitle('Peak survival rate by ROI, band, role, and group')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_stability_heatmap(summary_df, threshold=60.0, title_suffix=''):
    """Heatmap of the percentage of participants with a peak detected in all 3 movies.

    Rows = ROI x band. Columns = role x group. Cells below threshold are
    outlined in red.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Output of the movie-stability summary aggregation. Must contain
        columns: roi, band, role, group, pct_detected_all_3.
    threshold : float, optional
        Percentage cutoff, drawn as a red cell outline below this value.
    title_suffix : str, optional
        Appended to the figure title, e.g. to distinguish an individualized-
        band version from the fixed-window version.

    Returns
    -------
    matplotlib.figure.Figure
    """
    summary_df = summary_df.copy()
    summary_df['roi_band'] = summary_df['roi'] + ' / ' + summary_df['band']
    summary_df['role_group'] = summary_df['role'] + '\n' + summary_df['group']
    pivot = summary_df.pivot_table(index='roi_band', columns='role_group', values='pct_detected_all_3')

    fig, ax = plt.subplots(figsize=(1.6 * pivot.shape[1] + 2, 0.6 * pivot.shape[0] + 2))
    im = ax.imshow(pivot.values, cmap='viridis', vmin=0, vmax=100, aspect='auto')

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if np.isnan(val):
                continue
            ax.text(j, i, f'{val:.0f}%', ha='center', va='center', fontsize=8,
                    color='white' if val < 60 else 'black')
            if val < threshold:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor='red', lw=2))

    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, fontsize=8)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=9)
    fig.colorbar(im, ax=ax, label='% detected in all 3 movies', shrink=0.8)
    ax.set_title(
        f'Peak detection consistency across movies{title_suffix} (red = below {threshold:.0f}%)', fontsize=10,
    )
    fig.tight_layout()
    return fig


def plot_detection_consistency_heatmap(movie_band_peaks_df, movies, bands=('slow', 'fast'), title=None):
    """Per-participant heatmap of peak detection across movies and bands.

    Rows = participants (sorted by role, then group). Columns = movie x band
    (e.g. 'Peppa/slow', 'Peppa/fast', ...). Cell is green if a peak was
    detected, gray otherwise. Row labels colored by group.

    Parameters
    ----------
    movie_band_peaks_df : pd.DataFrame
        Long-format per participant x band x movie peak table, pre-filtered
        to a single ROI. Must contain columns: participant_id, role, group,
        band, movie, center_freq.
    movies : list of str
        Movie names, in column display order.
    bands : tuple of str, optional
        Band names, in column display order.
    title : str or None, optional
        Axes title. Defaults to a generic label if not given.

    Returns
    -------
    matplotlib.figure.Figure
    """
    participants = (
        movie_band_peaks_df[['participant_id', 'role', 'group']]
        .drop_duplicates()
        .sort_values(['role', 'group', 'participant_id'])
    )
    columns = [(movie, band) for movie in movies for band in bands]
    col_labels = [f'{movie}\n{band}' for movie, band in columns]

    detected = movie_band_peaks_df.set_index(['participant_id', 'movie', 'band'])['center_freq'].notna()

    grid = np.zeros((len(participants), len(columns)))
    for row, pid in enumerate(participants['participant_id']):
        for col, (movie, band) in enumerate(columns):
            grid[row, col] = float(detected.get((pid, movie, band), False))

    fig, ax = plt.subplots(figsize=(1.2 * len(columns) + 2, max(6, 0.12 * len(participants))))
    ax.imshow(grid, cmap=ListedColormap(['#bbbbbb', '#2ca02c']), vmin=0, vmax=1, aspect='auto')

    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(col_labels, fontsize=8)
    ax.set_yticks(range(len(participants)))
    ax.set_yticklabels(participants['participant_id'], fontsize=4)
    for tick_label, group in zip(ax.get_yticklabels(), participants['group']):
        tick_label.set_color(COLORS.get(group, 'gray'))

    ax.set_title(title or 'Peak detection consistency: movie x band, per participant', fontsize=10)
    fig.tight_layout()
    return fig
