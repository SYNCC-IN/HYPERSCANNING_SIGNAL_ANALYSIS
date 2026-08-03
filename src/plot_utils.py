import numpy as np
import matplotlib.pyplot as plt
import xarray as xr


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
        Divide each channel by its standard deviation before plotting.
        Useful for stacked EEG; set to False for IBI / RMSSD where the
        absolute scale is meaningful.
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
        ylabel="IBI (s)",
        title="Inter-beat intervals",
    )
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import xarray as xr

    if "time" not in data_xr.coords:
        raise ValueError("data_xr must have a 'time' coordinate.")

    time = np.asarray(data_xr.coords["time"].values, dtype=float)

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
    if normalize:
        stds = np.std(values, axis=1, keepdims=True)
        stds[stds == 0] = 1.0
        plot_values = values / stds
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

        # ── 50 μV scale bar — left edge of the bottom trace ──────────────────
        # Blended transform: x in axes-fraction coords (pinned to left wall),
        # y in data coords (so the height correctly represents 50 μV).
        from matplotlib.transforms import blended_transform_factory

        scale_uv = 50.0
        if normalize:
            # Signal was divided by its std, so 50 μV maps to 50/std plot units.
            std_bottom = float(np.std(values[0]))
            scale_height = scale_uv / std_bottom if std_bottom > 0 else scale_uv
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
        _ylabel = ylabel or (str(data_xr.name) if data_xr.name else "Value")

    # ── Axes labels ───────────────────────────────────────────────────────────
    ax.set_xlabel(xlabel)
    ax.set_ylabel(_ylabel)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    return fig, ax


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

