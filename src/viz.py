"""Plotting and figure-generation functions shared across the exploratory spectral pipeline."""

import mne
import numpy as np
import matplotlib.pyplot as plt

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
