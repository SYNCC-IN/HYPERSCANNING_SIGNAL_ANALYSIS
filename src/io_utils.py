"""Per-modality NetCDF readers, file discovery, and small path/array utilities.

Thin wrappers over the generic core in `src.netcdf_io`: EEG and IBI files
share the by-task export's file-naming/path conventions and modality-agnostic
attributes (`src.netcdf_io.read_core_attrs`, `parse_task_events`), so
`get_participant_files`/`load_eeg_nc`/`load_ibi_nc` only add what is specific
to their own modality (channel data, or the raw 1-D signal).

`load_eeg_nc` reads ICA-cleaned EEG NetCDF files produced by
``scripts/EEG_ICA_clean.py`` / ``src/ica_preprocessing.py``. Each file covers
one participant's whole ``passive_movies`` task (all movies back to back, CAR
referenced, mastoids already dropped), named
``<dyad_id>_EEG_<ch|cg>_passive_movies_cleaned.nc`` under
``<data_dir>/<dyad_id>/``. `load_ibi_nc` reads the matching per-task IBI
export, already interpolated onto the EEG time grid. Individual movie
boundaries within either recording are given by the shared
``task_events_structure`` attribute (`src.netcdf_io.parse_task_events`).
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .netcdf_io import load_xarray_from_netcdf, parse_task_events, read_core_attrs
except ImportError:  # pragma: no cover - fallback for direct script execution
    from src.netcdf_io import load_xarray_from_netcdf, parse_task_events, read_core_attrs

ROLE_FROM_CODE = {'ch': 'child', 'cg': 'caregiver'}

_EEG_GLOB = '*_passive_movies_cleaned.nc'
_EEG_STEM_REGEX = re.compile(r'^(?P<dyad_id>[A-Za-z]+_\d+)_EEG_(?P<role_code>ch|cg)_passive_movies_cleaned$')


def ensure_dir(path):
    """Create a directory (and parents) if it does not already exist.

    Parameters
    ----------
    path : str or pathlib.Path
        Directory path to create.

    Returns
    -------
    pathlib.Path
        The same path, as a ``Path`` object.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def discover_participant_files(data_dir, glob_pattern, stem_regex, role_from_code):
    """Scan a directory tree for per-participant files and organize them by participant.

    Generic core behind `get_participant_files`: recursively glob candidate
    files, then match each file's stem against a regex with named groups
    ``dyad_id`` and ``role_code``.

    Parameters
    ----------
    data_dir : str or pathlib.Path
        Root directory to scan recursively.
    glob_pattern : str
        Recursive glob pattern selecting candidate files (e.g.
        ``'*_passive_movies_cleaned.nc'``).
    stem_regex : str or re.Pattern
        Regex matched against each candidate file's stem (``Path.stem``); must
        define named groups ``dyad_id`` and ``role_code``. A stem that does not
        match is silently skipped, not an error.
    role_from_code : dict
        Maps a matched ``role_code`` to a role name (e.g.
        ``{'ch': 'child', 'cg': 'caregiver'}``).

    Returns
    -------
    pandas.DataFrame
        One row per matched file, with columns: ``filepath``, ``dyad_id``,
        ``role_code``, ``role``.
    """
    data_dir = Path(data_dir)
    pattern = re.compile(stem_regex) if isinstance(stem_regex, str) else stem_regex

    rows = []
    for filepath in sorted(data_dir.rglob(glob_pattern)):
        m = pattern.match(filepath.stem)
        if m is None:
            continue
        rows.append({
            'filepath': filepath,
            'dyad_id': m.group('dyad_id'),
            'role_code': m.group('role_code'),
            'role': role_from_code[m.group('role_code')],
        })
    return pd.DataFrame(rows)


def get_participant_files(data_dir):
    """Scan a directory of cleaned EEG NetCDF files and organize them by participant.

    Expects files named ``<dyad_id>_EEG_<ch|cg>_passive_movies_cleaned.nc`` under
    ``<data_dir>/<dyad_id>/``, one file per participant covering all movies.

    Parameters
    ----------
    data_dir : str or pathlib.Path
        Root directory containing one subfolder per dyad with cleaned EEG NetCDF files.

    Returns
    -------
    pandas.DataFrame
        One row per participant, with columns: ``filepath``, ``dyad_id``,
        ``role_code`` (``ch``/``cg``), ``role`` (``child``/``caregiver``).
    """
    return discover_participant_files(data_dir, _EEG_GLOB, _EEG_STEM_REGEX, ROLE_FROM_CODE)


def load_eeg_nc(filepath):
    """Load a cleaned, CAR-referenced EEG NetCDF file covering the whole passive-movies task.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Path to a ``*_passive_movies_cleaned.nc`` file.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``data`` : ndarray, shape (n_channels, n_times) -- EEG signal in microvolts,
          the full recording including pre/post margins and inter-movie gaps
        - ``channel_names`` : list of str -- 19-channel scalp montage (CAR referenced)
        - ``sfreq`` : float -- sampling frequency in Hz
        - ``time`` : ndarray, shape (n_times,) -- time axis in seconds (0 = task onset)
        - ``dyad_id`` : str
        - ``role_code`` : str -- ``ch`` or ``cg``
        - ``role`` : str -- ``child`` or ``caregiver``
        - ``movies`` : list of dict -- ``{'name', 'start_s', 'duration_s'}`` per movie,
          chunk-relative (matches ``time``), giving each movie's boundaries within it
        - ``age_months`` : float or None -- child age in months
        - ``group`` : str or None -- diagnostic group (e.g. ``TD``, ``ASD``)
        - ``sex`` : str or None
    """
    data_xr = load_xarray_from_netcdf(str(filepath))
    core = read_core_attrs(data_xr)
    movies = parse_task_events(data_xr, reference="relative")
    role_code = str(core["who"])

    return {
        'data': data_xr.transpose('channel', 'time').values.astype(float),
        'channel_names': list(data_xr.coords['channel'].values),
        'sfreq': core["sfreq"],
        'time': np.asarray(data_xr.coords['time'].values, dtype=float),
        'dyad_id': str(core["dyad_id"]),
        'role_code': role_code,
        'role': ROLE_FROM_CODE.get(role_code, role_code),
        'movies': movies,
        'age_months': core["age_months"],
        'group': core["group"],
        'sex': core["sex"],
    }


def load_ibi_nc(filepath):
    """Load a per-task IBI NetCDF file, already interpolated onto the EEG time grid.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Path to a ``<dyad_id>_IBI_<ch|cg>_passive_movies.nc`` file.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``data`` : ndarray, shape (n_times,) -- interbeat-interval signal
        - ``sfreq`` : float -- sampling frequency in Hz
        - ``time`` : ndarray, shape (n_times,) -- time axis in seconds
        - ``dyad_id`` : str
        - ``role_code`` : str -- ``ch`` or ``cg``
        - ``role`` : str -- ``child`` or ``caregiver``
    """
    data_xr = load_xarray_from_netcdf(str(filepath))
    core = read_core_attrs(data_xr)
    role_code = str(core["who"])

    return {
        'data': np.asarray(data_xr.values, dtype=float).reshape(-1),
        'sfreq': core["sfreq"],
        'time': np.asarray(data_xr.coords['time'].values, dtype=float),
        'dyad_id': str(core["dyad_id"]),
        'role_code': role_code,
        'role': ROLE_FROM_CODE.get(role_code, role_code),
    }


def trim_to_event_window(data, time, duration, start=0.0):
    """Slice signal data to a time window, e.g. one movie within a longer recording.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        EEG signal, matching ``time`` along the last axis.
    time : ndarray, shape (n_times,)
        Time axis in seconds.
    duration : float
        Window duration in seconds.
    start : float, optional
        Window start time in seconds (matching ``time``'s reference). Defaults to 0.

    Returns
    -------
    tuple
        ``(data_trimmed, time_trimmed)`` restricted to ``start <= time <= start + duration``.
    """
    mask = (time >= start) & (time <= start + duration)
    return data[:, mask], time[mask]
