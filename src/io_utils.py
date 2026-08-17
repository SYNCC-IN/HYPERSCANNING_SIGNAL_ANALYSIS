"""Data loading, path management, and results I/O for the exploratory spectral pipeline.

Reads ICA-cleaned EEG NetCDF files produced by ``scripts/EEG_ICA_clean.py`` /
``src/ica_preprocessing.py``. Each file covers one participant's whole
``passive_movies`` task (all movies back to back, CAR referenced, mastoids
already dropped), named ``<dyad_id>_EEG_<ch|cg>_passive_movies_cleaned.nc``
under ``<data_dir>/<dyad_id>/``. Individual movie boundaries within the
recording are given by the ``task_events_structure`` attribute.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .ncdf import load_xarray_from_netcdf, get_export_metadata
except ImportError:  # pragma: no cover - fallback for direct script execution
    from src.ncdf import load_xarray_from_netcdf, get_export_metadata

ROLE_FROM_CODE = {'ch': 'child', 'cg': 'caregiver'}


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
    data_dir = Path(data_dir)
    pattern = re.compile(r'^(?P<dyad_id>[A-Za-z]+_\d+)_EEG_(?P<role_code>ch|cg)_passive_movies_cleaned$')

    rows = []
    for filepath in sorted(data_dir.rglob('*_passive_movies_cleaned.nc')):
        m = pattern.match(filepath.stem)
        if m is None:
            continue
        rows.append({
            'filepath': filepath,
            'dyad_id': m.group('dyad_id'),
            'role_code': m.group('role_code'),
            'role': ROLE_FROM_CODE[m.group('role_code')],
        })
    return pd.DataFrame(rows)


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
          giving each movie's boundaries within ``time``
        - ``age_months`` : float or None -- child age in months
        - ``group`` : str or None -- diagnostic group (e.g. ``TD``, ``ASD``)
        - ``sex`` : str or None
    """
    data_xr = load_xarray_from_netcdf(str(filepath))
    metadata = get_export_metadata(data_xr)
    child_info = metadata.get('child_info', {})
    if not isinstance(child_info, dict):
        child_info = {}

    movies = [
        {'name': ev['name'], 'start_s': ev['start_rel_s'], 'duration_s': ev['duration_s']}
        for ev in data_xr.attrs.get('task_events_structure', [])
    ]

    role_code = str(data_xr.attrs['who'])
    return {
        'data': data_xr.transpose('channel', 'time').values.astype(float),
        'channel_names': list(data_xr.coords['channel'].values),
        'sfreq': float(data_xr.attrs['sampling_freq']),
        'time': np.asarray(data_xr.coords['time'].values, dtype=float),
        'dyad_id': str(data_xr.attrs['dyad_id']),
        'role_code': role_code,
        'role': ROLE_FROM_CODE.get(role_code, role_code),
        'movies': movies,
        'age_months': child_info.get('age_months'),
        'group': child_info.get('group'),
        'sex': child_info.get('sex'),
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
