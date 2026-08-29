"""Generic NetCDF <-> xarray core, shared by every modality-specific loader.

Depends only on ``xarray``/``json``/``numbers`` -- no ``pandas``, no MNE, no
EEG/IBI/ET specifics. Per-modality readers (`src.io_utils.load_eeg_nc`,
`src.io_utils.load_ibi_nc`, ...) are thin wrappers built on top of the
functions here: `load_xarray_from_netcdf` to read the file,
`read_core_attrs`/`get_export_metadata` to pull the modality-agnostic
attributes every loader needs, and `parse_task_events`/`task_regions` as the
single source of truth for the embedded task-event metadata.
"""

import json
import numbers

import xarray as xr


def load_xarray_from_netcdf(filename: str, decode_json_attrs: bool = True) -> xr.DataArray:
    """Load DataArray from a NetCDF file with optional JSON attribute decoding.

    Args:
        filename: Path to the NetCDF file.
        decode_json_attrs: If True, decode JSON-serialized attribute strings
            (typically dict/list values serialized during export).

    Returns:
        xarray.DataArray: Loaded DataArray.
    """
    data_xr = xr.load_dataarray(filename)
    if decode_json_attrs:
        decode_json_attrs_inplace(data_xr.attrs)
    return data_xr


def load_ncdf(path) -> xr.DataArray:
    """Load a NetCDF file into an xarray.DataArray (one-line convenience wrapper).

    Args:
        path: Path to the NetCDF file.

    Returns:
        xarray.DataArray: Loaded DataArray, with JSON-serialized attrs decoded.
    """
    return load_xarray_from_netcdf(str(path))


def get_export_metadata(data_xr: xr.DataArray) -> dict:
    """Get structured export metadata from a DataArray attrs payload.

    Args:
        data_xr: Exported DataArray that may contain ``metadata_json`` attr.

    Returns:
        dict: Parsed metadata dictionary, or empty dict if unavailable/invalid.
    """
    raw_metadata = data_xr.attrs.get("metadata_json")
    decoded_metadata = _try_decode_json_attr_value(raw_metadata)
    if isinstance(decoded_metadata, dict):
        return decoded_metadata
    return {}


def read_core_attrs(data_xr: xr.DataArray) -> dict:
    """Read the modality-agnostic attributes every per-modality loader needs.

    Args:
        data_xr: Exported DataArray (EEG, IBI, ET, ...).

    Returns:
        dict: ``{'dyad_id', 'who', 'sfreq', 'age_months', 'group', 'sex'}``.
        ``who`` is the raw role code (e.g. ``'ch'``/``'cg'``) as exported --
        callers map it to a role name themselves. ``age_months``/``group``/
        ``sex`` come from the ``child_info`` block of `get_export_metadata`
        and are ``None`` if that block is absent or not a dict.
    """
    child_info = get_export_metadata(data_xr).get("child_info", {})
    if not isinstance(child_info, dict):
        child_info = {}
    return {
        "dyad_id": data_xr.attrs.get("dyad_id"),
        "who": data_xr.attrs.get("who"),
        "sfreq": float(data_xr.attrs["sampling_freq"]),
        "age_months": child_info.get("age_months"),
        "group": child_info.get("group"),
        "sex": child_info.get("sex"),
    }


def parse_task_events(data_xr: xr.DataArray, reference: str = "relative") -> list[dict]:
    """Parse the embedded task-event metadata into a normalized event list.

    The export pipeline (`src.export._build_events_structure`) writes each
    event's start in two time origins: ``start_s`` (absolute, in the original
    recording's time base) and ``start_rel_s`` (relative to the exported
    chunk's own ``time`` coordinate, i.e. ``start_s - chunk_start``). Which one
    a consumer needs depends on what its own time axis is anchored to.

    Args:
        data_xr: Exported DataArray with a ``task_events_structure`` attr.
        reference: ``'relative'`` (default) reads ``start_rel_s``, matching the
            chunk-relative ``time`` coordinate that `src.io_utils.load_eeg_nc`
            and every downstream Stage 1/2 consumer slice against.
            ``'absolute'`` reads ``start_s``, the original recording's time
            base -- only meaningful if the caller's own time axis is also in
            that base.

    Returns:
        list of dict: ``[{'name', 'start_s', 'duration_s'}, ...]`` in event
        order, ``start_s`` reported in the selected ``reference`` origin.
    """
    if reference not in ("relative", "absolute"):
        raise ValueError(f"Unknown reference {reference!r}; expected 'relative' or 'absolute'")
    start_key = "start_rel_s" if reference == "relative" else "start_s"

    task_events = data_xr.attrs.get("task_events_structure", [])
    if isinstance(task_events, str):
        task_events = json.loads(task_events)

    events = []
    for idx, event in enumerate(task_events or []):
        if not isinstance(event, dict):
            continue
        events.append({
            "name": str(event.get("name", f"event_{idx + 1}")),
            "start_s": float(event.get(start_key, 0.0)),
            "duration_s": float(event.get("duration_s", 0.0)),
        })
    return events


def task_regions(data_xr: xr.DataArray, reference: str = "relative") -> list[dict]:
    """Build plotting regions from the embedded task-event metadata.

    Thin wrapper over `parse_task_events` for callers that want the
    ``{"span": (start, end), "name": ...}`` shape `src.plot_utils.plot_xarray_signals`
    expects.

    Args:
        data_xr: Exported DataArray with a ``task_events_structure`` attr.
        reference: See `parse_task_events`.

    Returns:
        list of dict: ``[{'span': (start_s, start_s + duration_s), 'name': ...}, ...]``.
    """
    return [
        {"span": (event["start_s"], event["start_s"] + event["duration_s"]), "name": event["name"]}
        for event in parse_task_events(data_xr, reference=reference)
    ]


def _sanitize_netcdf_attr_value(value):
    if value is None:
        return ""

    if isinstance(value, (str, bytes, bool, int, float, numbers.Number)):
        return value

    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, default=str)

    if isinstance(value, (list, tuple)):
        has_nested = any(isinstance(v, (dict, list, tuple)) for v in value)
        if has_nested:
            return json.dumps(value, ensure_ascii=False, default=str)
        return ["" if v is None else v for v in value]

    if hasattr(value, "tolist"):
        converted = value.tolist()
        return _sanitize_netcdf_attr_value(converted)

    return str(value)


def sanitize_netcdf_attrs_inplace(attrs: dict) -> None:
    """Sanitize a DataArray's attrs dict in place for NetCDF serialization.

    Args:
        attrs: Attrs dict to sanitize (e.g. ``data_array.attrs``) -- mutated
            in place. ``None`` becomes ``""``; dicts and nested lists/tuples
            are JSON-serialized; anything else is coerced via ``str()`` unless
            it is already a NetCDF-safe scalar/list type.
    """
    for key in list(attrs.keys()):
        attrs[key] = _sanitize_netcdf_attr_value(attrs[key])


def _try_decode_json_attr_value(value):
    if not isinstance(value, str):
        return value

    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value

    try:
        decoded = json.loads(stripped)
    except (json.JSONDecodeError, TypeError, ValueError):
        return value
    return decoded


def decode_json_attrs_inplace(attrs: dict) -> None:
    """Decode JSON-serialized string attrs back into dict/list values, in place.

    Args:
        attrs: Attrs dict to decode (e.g. ``data_array.attrs``) -- mutated in
            place. Only string values that look like JSON (start with ``{`` or
            ``[``) and parse successfully are replaced; everything else is
            left untouched.
    """
    for key in list(attrs.keys()):
        attrs[key] = _try_decode_json_attr_value(attrs[key])
