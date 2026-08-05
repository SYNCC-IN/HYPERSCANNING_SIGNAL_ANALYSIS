import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from scipy import signal


def plot_xarray_signals(
    data_xr: "xr.DataArray",
    regions: "list[dict] | None" = None,
    stacked: "bool | None" = None,
    max_channels: int = 30,
    spacing: float = 8.0,
    normalize: bool = True,
    figsize: "tuple[float, float]" = (16.0, 9.0),
    event_duration: "float | None" = None,
    time_margin_s: "float | None" = None,
    title: str = "",
    xlabel: str = "Time (s)  (0 = event start)",
    ylabel: "str | None" = None,
    line_color: str = "#1f4f8b",
    linewidth: float = 0.6,
) -> "tuple[plt.Figure, plt.Axes]":
    """Plot an xarray DataArray as a time series with optional highlighted regions.

    Works for any modality stored as a DataArray with a ``time`` dimension:
    EEG (stacked multi-channel), ECG, IBI, RMSSD (single-channel line plots),
    ET gaze / pupil data, or any other time-indexed signal.

    Stacking mode is selected automatically based on the number of channels,
    but can be forced with the ``stacked`` parameter.

    Parameters
    ----------
    data_xr : xr.DataArray
        Input data.  Must have a ``time`` dimension.  Optional ``channel``
        dimension is used for labels and stacking.
    regions : list of dict, optional
        Highlighted time spans.  Each dict may contain:

        ``span``  : (float, float)   start and end time in seconds — required
        ``name``  : str              legend label — optional
        ``color`` : str              matplotlib color — default '#d62728' (red)
        ``alpha`` : float            transparency — default 0.18 for stacked,
                                     0.30 for single-channel

    stacked : bool | None
        True  → vertically stacked traces, one per channel (EEG style).
        False → all channels overlaid on the same y-axis (ECG / IBI style).
        None  → auto: stacked when n_channels > 1, overlaid when n_channels == 1.

    max_channels : int
        Maximum number of channels / series to display.
    spacing : float
        Vertical offset between stacked traces (in normalised units).
        Ignored when stacked=False.
    normalize : bool
        Divide every channel by the **median per-channel standard deviation**
        across all displayed channels.  This preserves relative amplitude
        differences between channels while bringing the overall scale into a
        convenient range.  Set to False for IBI / RMSSD where the absolute
        scale is meaningful.
    figsize : (float, float)
        Matplotlib figure size in inches.
    event_duration : float, optional
        Duration of the core event in seconds.  If given, dashed vertical
        lines are drawn at t = 0 and t = event_duration, and the margin
        regions (if time_margin_s is set) are shaded light gray.
    time_margin_s : float, optional
        Length of the pre/post margin.  Enables light-gray shading outside
        [0, event_duration].  Requires event_duration to be set as well.
    title : str
        Figure title.
    xlabel : str
        X-axis label.
    ylabel : str | None
        Y-axis label.  Auto-derived from DataArray.name or channel names
        when None.
    line_color : str
        Matplotlib color for signal traces.
    linewidth : float
        Line width for signal traces.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes

    Examples
    --------
    # EEG — stacked, with rejected windows and event boundaries
    plot_xarray_signals(
        eeg_xr,
        regions=[{'span': (12.3, 15.1), 'name': 'rejected', 'color': '#d62728'}],
        event_duration=180.0,
        time_margin_s=5.0,
        title="EEG passive movies — child",
    )

    # ECG — single channel, no stacking
    plot_xarray_signals(ecg_xr, stacked=False, normalize=False, title="ECG")

    # IBI — absolute values matter, no normalisation
    plot_xarray_signals(
        ibi_xr,
        stacked=False,
        normalize=False,
        ylabel="IBI (ms)",
        title="Inter-beat intervals",
    )
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import xarray as xr

    if "time" not in data_xr.coords:
        raise ValueError("data_xr must have a 'time' coordinate.")

    time = np.asarray(data_xr.coords["time"].values, dtype=float)
    units = data_xr.attrs.get("units", "unknown")
    modality = data_xr.attrs.get("modality", "Signal")

    # ── Reshape to (n_channels, n_times) ─────────────────────────────────────
    values = np.asarray(data_xr.values)
    if values.ndim == 1:
        values = values[np.newaxis, :]     # (1, n_times)
        ch_names = [str(data_xr.name or "Signal")]
    elif values.ndim == 2:
        # Ensure shape is (n_channels, n_times)
        if values.shape[1] == len(time):
            pass                           # already (n_ch, n_times)
        elif values.shape[0] == len(time):
            values = values.T             # was (n_times, n_ch)
        else:
            raise ValueError(
                f"Cannot align data shape {values.shape} with time axis "
                f"of length {len(time)}."
            )
        if "channel" in data_xr.coords:
            ch_names = [str(ch) for ch in data_xr.coords["channel"].values]
        else:
            ch_names = [f"ch{i}" for i in range(values.shape[0])]
    else:
        raise ValueError(
            f"Expected 1-D or 2-D DataArray, got {values.ndim}-D."
        )

    n_channels = min(max_channels, values.shape[0])
    values     = values[:n_channels]
    ch_names   = ch_names[:n_channels]

    # ── Auto-detect stacking mode ─────────────────────────────────────────────
    if stacked is None:
        stacked = n_channels > 1

    # ── Normalise ─────────────────────────────────────────────────────────────
    # All channels are divided by the same factor (median per-channel std)
    # so that relative amplitudes between channels are preserved.
    if normalize:
        per_ch_stds = np.std(values, axis=1)          # (n_channels,)
        valid = per_ch_stds[per_ch_stds > 0]
        global_scale = float(np.median(valid)) if valid.size else 1.0
        plot_values = values / global_scale
        units = "normalized"
    else:
        plot_values = values.copy()

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)

    # ── Light-gray margin shading ─────────────────────────────────────────────
    if time_margin_s is not None and time_margin_s > 0 and event_duration is not None:
        t_start = float(time[0])
        t_end   = float(time[-1])
        if t_start < 0.0:
            ax.axvspan(t_start, 0.0,
                       color="#cccccc", alpha=0.45, zorder=0, label="pre-event margin")
        if np.isfinite(event_duration) and t_end > event_duration:
            ax.axvspan(event_duration, t_end,
                       color="#cccccc", alpha=0.45, zorder=0, label="post-event margin")

    # ── Highlighted regions ───────────────────────────────────────────────────
    if regions:
        seen_labels: set[str] = set()
        default_alpha = 0.18 if stacked else 0.30
        for region in regions:
            start, end = region["span"]
            name       = region.get("name", "")
            color      = region.get("color", "#d62728")
            alpha      = region.get("alpha", default_alpha)
            # Only add each label once to the legend
            label_arg  = name if name not in seen_labels else None
            ax.axvspan(start, end, color=color, alpha=alpha,
                       zorder=2, label=label_arg)
            if name:
                seen_labels.add(name)
            if name:
                ax.text(
                    (start + end) / 2,
                    0.97,
                    name,
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="top",
                    fontsize=8, fontweight="bold",
                    color=color,
                    clip_on=True,
                    zorder=6,
                )

    # ── Event boundary lines ──────────────────────────────────────────────────
    if event_duration is not None and np.isfinite(event_duration):
        ax.axvline(0.0,             color="#444444", linewidth=0.8,
                   linestyle="--", alpha=0.6, zorder=3)
        ax.axvline(event_duration,  color="#444444", linewidth=0.8,
                   linestyle="--", alpha=0.6, zorder=3)

    # ── Signal traces ─────────────────────────────────────────────────────────
    if stacked:
        offsets = np.arange(n_channels) * spacing
        for idx in range(n_channels):
            ax.plot(time, plot_values[idx] + offsets[idx],
                    linewidth=linewidth, color=line_color, zorder=1)
        ax.set_yticks(offsets)
        ax.set_yticklabels(ch_names)

        # ── 50 μV scale bar ───────────────────────────────────────────────────
        # Height is computed from the same global_scale used for normalisation,
        # so the bar correctly represents 50 μV regardless of which channel
        # sits at the bottom.
        from matplotlib.transforms import blended_transform_factory

        scale_uv = 50.0
        if normalize:
            # global_scale has units of μV → divide to get normalised height
            scale_height = scale_uv / global_scale
        else:
            scale_height = scale_uv          # raw μV space — direct mapping

        trans  = blended_transform_factory(ax.transAxes, ax.transData)
        bar_x  = 0.005                       # 0.5 % from the left wall
        bar_y0 = offsets[0] - scale_height / 2
        bar_y1 = offsets[0] + scale_height / 2

        ax.plot(
            [bar_x, bar_x], [bar_y0, bar_y1],
            transform=trans,
            color='magenta', linewidth=2.5,
            solid_capstyle='butt',
            clip_on=False, zorder=5,
        )
        ax.text(
            bar_x, bar_y1,
            f'{scale_uv:.0f} μV',
            transform=trans,
            color='magenta', fontsize=7,
            va='bottom', ha='center',
            clip_on=False,
        )

        _ylabel = ylabel or "Channel"
    else:
        for idx, name in enumerate(ch_names):
            ax.plot(time, plot_values[idx],
                    linewidth=linewidth, label=name, zorder=1)
        if n_channels > 1:
            ax.legend(fontsize=8, loc="upper right")
        _ylabel = ylabel or f"{modality} ({units})"

    # ── Axes labels ───────────────────────────────────────────────────────────
    ax.set_xlabel(xlabel)
    ax.set_ylabel(_ylabel)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    return fig, ax
# ==================================

def plot_filter_characteristics(b, a, f, T, Fs, f_lim=None, db_lim=None):
    """
    Plot comprehensive filter characteristics including magnitude response, 
    group delay, impulse response, and step response.
    
    Parameters
    ----------
    b : array_like
        Numerator polynomial coefficients of the filter
    a : array_like
        Denominator polynomial coefficients of the filter
    f : array_like
        Frequency vector for frequency response computation (Hz)
    T : float
        Time duration for impulse and step response plots (seconds)
    Fs : float
        Sampling frequency (Hz)
    f_lim : tuple of float, optional
        Frequency axis limits (f_min, f_max) in Hz. Default is (0, Fs/2)
    db_lim : tuple of float, optional
        Magnitude axis limits (db_min, db_max) in dB. Default is auto-scaled
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object
    """
    # Set default frequency limits
    if f_lim is None:
        f_lim = (0, Fs / 2)
    else:
        f_lim = (max(f_lim[0], 0), min(f_lim[1], Fs / 2))   
        f = f[np.logical_and(f >= f_lim[0], f <= f_lim[1])]
    
    # Create time vector
    time_vector = np.arange(-T, T, 1 / Fs)

    # Compute frequency response
    frequencies, frequency_response = signal.freqz(b, a, f, fs=Fs)
    magnitude = np.abs(frequency_response)
    magnitude_db = 20 * np.log10(magnitude)

    # Compute group delay
    _, group_delay = signal.group_delay((b, a), f, fs=Fs)

    # Compute impulse response
    impulse_signal = np.zeros(len(time_vector))
    impulse_signal[len(time_vector) // 2] = 1
    impulse_response = signal.lfilter(b, a, impulse_signal)

    # Compute step response
    step_signal = np.zeros(len(time_vector))
    step_signal[len(time_vector) // 2:] = 1
    step_response = signal.lfilter(b, a, step_signal)

    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Plot magnitude response (top left)
    _plot_magnitude_response(frequencies, magnitude_db, f_lim, db_lim)
    
    # Plot group delay (bottom left)
    _plot_group_delay(frequencies, group_delay, f_lim)
    
    # Plot impulse response (top right)
    _plot_impulse_response(time_vector, impulse_signal, impulse_response, T)
    
    # Plot step response (bottom right)
    _plot_step_response(time_vector, step_signal, step_response, T)
    
    fig.subplots_adjust(hspace=0.5)
    plt.show()
    
    return fig


def _find_db_crossings(frequencies, magnitude_db, target_db):
    """
    Find frequencies where magnitude response crosses a target dB value.
    
    Parameters
    ----------
    frequencies : array_like
        Frequency vector (Hz)
    magnitude_db : array_like
        Magnitude response in dB
    target_db : float
        Target dB value to find crossings
    
    Returns
    -------
    list
        List of frequencies where crossings occur
    """
    crossings = []
    for i in range(len(magnitude_db) - 1):
        # Check if the line segment crosses the target
        if (magnitude_db[i] >= target_db and magnitude_db[i + 1] < target_db) or \
           (magnitude_db[i] < target_db and magnitude_db[i + 1] >= target_db):
            # Linear interpolation to find exact crossing frequency
            freq_crossing = frequencies[i] + (target_db - magnitude_db[i]) * \
                           (frequencies[i + 1] - frequencies[i]) / \
                           (magnitude_db[i + 1] - magnitude_db[i])
            crossings.append(freq_crossing)
    return crossings


def _plot_magnitude_response(frequencies, magnitude_db, f_lim, db_lim):
    """
    Plot magnitude response of the filter in dB.
    
    Parameters
    ----------
    frequencies : array_like
        Frequency vector (Hz)
    magnitude_db : array_like
        Magnitude response in dB
    f_lim : tuple of float
        Frequency axis limits (f_min, f_max)
    db_lim : tuple of float or None
        Magnitude axis limits (db_min, db_max)
    """
    plt.subplot(2, 2, 1)
    plt.title('Magnitude Response')
    plt.plot(frequencies, magnitude_db)
    plt.ylabel('Magnitude [dB]')
    plt.grid(True)
    plt.xlim(f_lim)
    
    # Find and mark -6dB crossing points
    crossing_freqs = _find_db_crossings(frequencies, magnitude_db, -6.0)
    for freq in crossing_freqs:
        plt.axvline(x=freq, color='r', linestyle='--', linewidth=1.5, alpha=0.7)
    
    if db_lim is None:
        # Auto-scale to show only the zoomed frequency range
        magnitude_zoom = magnitude_db[np.logical_and(f_lim[0] < frequencies, frequencies < f_lim[1])]
        plt.ylim((np.min(magnitude_zoom), np.max(magnitude_zoom)))
    else:
        plt.ylim((db_lim[0], db_lim[1]))


def _plot_group_delay(frequencies, group_delay, f_lim):
    """
    Plot group delay of the filter.
    
    Parameters
    ----------
    frequencies : array_like
        Frequency vector (Hz)
    group_delay : array_like
        Group delay in samples
    f_lim : tuple of float
        Frequency axis limits (f_min, f_max)
    """
    plt.subplot(2, 2, 3)
    plt.title('Group Delay')
    plt.plot(frequencies, group_delay)
    plt.ylabel('Samples')
    plt.xlabel('Frequency [Hz]')
    plt.grid(True)
    plt.xlim(f_lim)
    plt.ylim([np.min(group_delay) - 1, np.max(group_delay) + 1])


def _plot_impulse_response(time_vector, impulse_signal, impulse_response, T):
    """
    Plot impulse response of the filter.
    
    Parameters
    ----------
    time_vector : array_like
        Time vector (seconds)
    impulse_signal : array_like
        Input impulse signal
    impulse_response : array_like
        Filter's impulse response
    T : float
        Time duration (seconds)
    """
    plt.subplot(2, 2, 2)
    plt.title('Impulse Response')
    plt.stem(time_vector, impulse_signal)
    plt.plot(time_vector, impulse_response, 'r')
    plt.xlim([-T / 4, T])
    plt.grid(True)


def _plot_step_response(time_vector, step_signal, step_response, T):
    """
    Plot step response of the filter.
    
    Parameters
    ----------
    time_vector : array_like
        Time vector (seconds)
    step_signal : array_like
        Input step signal
    step_response : array_like
        Filter's step response
    T : float
        Time duration (seconds)
    """
    plt.subplot(2, 2, 4)
    plt.title('Step Response')
    plt.plot(time_vector, step_signal)
    plt.plot(time_vector, step_response, 'r')
    plt.xlim([-T / 4, T])
    plt.xlabel('Time [s]')
    plt.grid(True)




# def plot_eeg_with_rejected_segments(
#     raw: "mne.io.BaseRaw",
#     rejected_windows: Optional["pd.DataFrame"] = None,
#     max_channels: int = 19,
#     spacing: float = 8.0,
#     figsize: tuple[float, float] = (16.0, 9.0),
#     time_offset: float = 0.0,
#     event_duration: Optional[float] = None,
#     time_margin_s: Optional[float] = None,
# ):
#     """Plot stacked EEG traces and highlight rejected windows.

#     Args:
#         raw: MNE Raw object with EEG channels.
#         rejected_windows: DataFrame with columns ``start_s`` and ``end_s`` in NCDF time coords.
#         max_channels: Maximum number of EEG channels to display.
#         spacing: Vertical distance between channel traces.
#         figsize: Matplotlib figure size.
#         time_offset: First sample time from the NCDF time coordinate (typically negative,
#             equal to -time_margin_s). Used to shift MNE's 0-based time axis to match the
#             original NCDF time axis where 0 = event start.
#         event_duration: Duration of the event in seconds; used to shade the post-event margin.
#         time_margin_s: Margin length in seconds; enables light-gray shading of pre/post margins.

#     Returns:
#         tuple: (figure, axis)
#     """
#     import matplotlib.pyplot as plt
#     import numpy as np

#     picks = raw.copy().pick("eeg")
#     data = picks.get_data()
#     # Shift MNE's 0-based time axis to match the NCDF time coordinate.
#     times = picks.times + time_offset
#     ch_names = list(picks.ch_names)

#     if data.size == 0:
#         raise ValueError("No EEG channels available to plot.")

#     n_channels = min(max_channels, data.shape[0])
#     data = data[:n_channels]
#     ch_names = ch_names[:n_channels]

#     stds = np.std(data,  keepdims=True) # axis=1,
#     # stds[stds == 0] = 1.0
#     normalized = data / stds # normalized to unit variance of all channels 
#                              # for better visual comparison across channels

#     fig, ax = plt.subplots(figsize=figsize)
#     offsets = np.arange(n_channels) * spacing

#     # Light-gray shading for margin regions (drawn first, behind traces).
#     if time_margin_s is not None and time_margin_s > 0:
#         t_start = float(times[0])
#         t_end = float(times[-1])
#         if t_start < 0.0:
#             ax.axvspan(t_start, 0.0, color="#cccccc", alpha=0.45, zorder=0, label="margin")
#         if event_duration is not None and np.isfinite(event_duration) and t_end > event_duration:
#             ax.axvspan(event_duration, t_end, color="#cccccc", alpha=0.45, zorder=0)

#     for idx in range(n_channels):
#         ax.plot(times, normalized[idx] + offsets[idx], linewidth=0.6, color="#1f4f8b", zorder=1)

#     if rejected_windows is not None and not rejected_windows.empty:
#         for _, row in rejected_windows.iterrows():
#             ax.axvspan(float(row["start_s"]), float(row["end_s"]), color="#d62728", alpha=0.18, zorder=2)

#     # Dashed vertical lines at event boundaries.
#     if event_duration is not None and np.isfinite(event_duration):
#         ax.axvline(0.0, color="#444444", linewidth=0.8, linestyle="--", alpha=0.6)
#         ax.axvline(event_duration, color="#444444", linewidth=0.8, linestyle="--", alpha=0.6)

#     ax.set_yticks(offsets)
#     ax.set_yticklabels(ch_names)
#     ax.set_xlabel("Time [s]  (0 = event start)")
#     ax.set_ylabel("EEG channel")
#     if "temp" in raw.info:
#         note = str(raw.info["temp"])
#     else:
#         note = ""
#     ax.set_title(f"AutoReject suggested rejections. {note}")
#     ax.grid(axis="x", alpha=0.2)
#     fig.tight_layout()
#     return fig, ax


# def plot_loaded_eeg_signals(
#     time_s: "np.ndarray",
#     signals: "np.ndarray",
#     channel_names: list,
#     max_channels: int = 19,
#     spacing: float = 8.0,
#     figsize: tuple = (16.0, 9.0),
#     event_duration_s: Optional[float] = None,
#     title: str = "Loaded EEG signal (stacked channels)",
# ):
#     """Plot loaded EEG traces in a stacked view analogous to
#     :func:`plot_eeg_with_rejected_segments`.

#     Args:
#         time_s: 1-D time axis in seconds (0 = event start).
#         signals: EEG data array of shape ``(n_chan, n_samp)``.
#         channel_names: Ordered channel labels matching ``signals`` rows.
#         max_channels: Maximum number of channels to display.
#         spacing: Vertical distance between channel traces.
#         figsize: Matplotlib figure size ``(width, height)`` in inches.
#         event_duration_s: If given, dashed vertical lines are drawn at
#             ``t = 0`` and ``t = event_duration_s``.
#         title: Figure title.

#     Returns:
#         tuple: ``(figure, axis)``
#     """
#     import numpy as np
#     import matplotlib.pyplot as plt

#     n_channels = min(max_channels, signals.shape[0])
#     if n_channels <= 0:
#         raise ValueError("No EEG channels available to plot.")

#     data = signals[:n_channels]
#     ch_names = list(channel_names[:n_channels])

#     stds = np.std(data, axis=1, keepdims=True)
#     stds[stds == 0] = 1.0
#     normalized = data / stds

#     fig, ax = plt.subplots(figsize=figsize)
#     offsets = np.arange(n_channels) * spacing

#     for idx in range(n_channels):
#         ax.plot(time_s, normalized[idx] + offsets[idx], linewidth=0.6, color="#1f4f8b", zorder=1)

#     if event_duration_s is not None and np.isfinite(event_duration_s):
#         ax.axvline(0.0, color="#444444", linewidth=0.8, linestyle="--", alpha=0.6)
#         ax.axvline(event_duration_s, color="#444444", linewidth=0.8, linestyle="--", alpha=0.6)

#     ax.set_yticks(offsets)
#     ax.set_yticklabels(ch_names)
#     ax.set_xlabel("Time [s]  (0 = event start)")
#     ax.set_ylabel("EEG channel")
#     ax.set_title(title)
#     ax.grid(axis="x", alpha=0.2)
#     fig.tight_layout()
#     return fig, ax

# def plot_xarray_with_regions(xr_dataarray, regions):
#     """
#     Plot a DataArray with events highlighted as background colors.

#     Parameters
#     ----------
#     xr_dataarray : xarray.DataArray
#         The data array to plot. The function accepts either a 1D signal or a
#         2D array with a ``time`` dimension and an optional ``channel`` dimension.
#     regions : list of dict
#         List of dictionaries, each containing 'span' (tuple of start_time, end_time)
#         and 'name' (str) for each event region.

#     Returns
#     -------
#     None
#         Displays the plot using matplotlib.
#     """
#     if "time" not in xr_dataarray.coords:
#         raise ValueError("xr_dataarray must have a 'time' coordinate.")

#     time = np.asarray(xr_dataarray.coords["time"].values, dtype=float)
#     values = np.asarray(xr_dataarray.values)

#     if values.ndim == 1:
#         series_list = [values]
#         labels = ["Signal"]
#     elif values.ndim == 2:
#         if values.shape[0] == len(time):
#             series_list = [values[:, idx] for idx in range(values.shape[1])]
#             if "channel" in xr_dataarray.coords:
#                 labels = [str(ch) for ch in xr_dataarray.coords["channel"].values]
#             else:
#                 labels = [f"Signal {idx + 1}" for idx in range(values.shape[1])]
#         elif values.shape[1] == len(time):
#             series_list = [values[idx, :] for idx in range(values.shape[0])]
#             if "channel" in xr_dataarray.coords:
#                 labels = [str(ch) for ch in xr_dataarray.coords["channel"].values]
#             else:
#                 labels = [f"Signal {idx + 1}" for idx in range(values.shape[0])]
#         else:
#             raise ValueError("Unable to align 2D data with the time coordinate.")
#     else:
#         raise ValueError("Expected a 1D or 2D DataArray for plotting.")

#     plt.figure(figsize=(12, 6))
#     for series, label in zip(series_list, labels):
#         plt.plot(time, series, label=label)

#     for region in regions:
#         start, end = region["span"]
#         name = region["name"]
#         plt.axvspan(start, end, alpha=0.3, label=name)

#     plt.xlabel("Time (s)")
#     plt.ylabel("Signal")
#     plt.legend()
#     plt.show()

# def plot_signal_with_events(time, data, channels, marker_channel, event_to_marker, selected_time):
#     """
#     Plot signal data with background colors indicating different events.
    
#     This function creates a time-series plot with colored background regions corresponding
#     to different events. Each event type is assigned a unique color and plotted as a 
#     semi-transparent background. Multiple signal channels can be overlaid on the same plot.
    
#     Parameters
#     ----------
#     time : array-like
#         Time vector in seconds, matching the length of data columns
#     data : numpy.ndarray
#         Signal data array with shape (n_channels, n_samples)
#     channels : list of str
#         List of channel names corresponding to rows in data
#     marker_channel : array-like
#         Array of marker values indicating which event is active at each time point.
#         0 indicates no event, other values map to events via event_to_marker dict.
#         Should have same length as time vector.
#     event_to_marker : dict
#         Dictionary mapping event names (str) to marker values (int).
#         Example: {'Peppa': 1, 'Incredibles': 2, 'Brave': 3}
#     selected_time : list of float
#         [start_time, end_time] in seconds, used for plot title
    
#     Returns
#     -------
#     None
#         Displays the plot using matplotlib
    
#     Notes
#     -----
#     - Event regions are plotted as semi-transparent (alpha=0.3) background spans
#     - Up to 6 default colors are cycled for different events
#     - Each event is only added to the legend once, even if it appears multiple times
#     - Grid is enabled with alpha=0.3 for better readability
    
#     Examples
#     --------
#     >>> time, channels, data = multimodal_data.get_signals(
#     ...     mode='EEG', member='ch', 
#     ...     selected_channels=['Fz', 'Cz'], 
#     ...     selected_times=[60, 120]
#     ... )
#     >>> time, marker_channel, event_to_marker = multimodal_data.get_events_as_marker_channel(
#     ...     selected_times=[60, 120]
#     ... )
#     >>> plot_signal_with_events(time, data, channels, marker_channel, 
#     ...                          event_to_marker, [60, 120])
#     """
#     # Create reverse mapping from marker value to event name
#     marker_to_event = {v: k for k, v in event_to_marker.items()}

#     # Create color map for events
#     colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral', 'lightpink', 'lavender']
#     event_colors = {}
#     for i, event_name in enumerate(marker_to_event.values()):
#         if event_name != '':
#             event_colors[event_name] = colors[i % len(colors)]

#     plt.figure(figsize=(15, 6), dpi=100)

#     # Plot background colors for each event
#     current_marker = marker_channel[0]
#     segment_start = time[0]

#     for i in range(1, len(marker_channel)):
#         if marker_channel[i] != current_marker:
#             # End of current segment
#             if current_marker > 0 and current_marker in marker_to_event:
#                 event_name = marker_to_event[current_marker]
#                 if event_name in event_colors:
#                     plt.axvspan(segment_start, time[i-1], 
#                             alpha=0.6, color=event_colors[event_name], 
#                             label=event_name if segment_start == time[0] or event_name not in plt.gca().get_legend_handles_labels()[1] else '')
#             # Start new segment
#             current_marker = marker_channel[i]
#             segment_start = time[i]

#     # Handle last segment
#     if current_marker > 0 and current_marker in marker_to_event:
#         event_name = marker_to_event[current_marker]
#         if event_name in event_colors:
#             plt.axvspan(segment_start, time[-1], 
#                     alpha=0.6, color=event_colors[event_name],
#                     label=event_name if event_name not in plt.gca().get_legend_handles_labels()[1] else '')

#     # Plot  data
#     for idxch, ch in enumerate(channels):
#             plt.plot(time, data[:, idxch], label=ch, linewidth=1.0)

#     plt.xlabel('Time (s)')
#     plt.ylabel('Amplitude (µV)')
#     plt.title(f'EEG Signal ({", ".join(channels)}) and Event Markers between {selected_time[0]}s and {selected_time[1]}s')
#     plt.legend(loc='upper right')
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.show()