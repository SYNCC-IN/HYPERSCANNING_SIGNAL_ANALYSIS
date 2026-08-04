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
        _decode_json_attrs_inplace(data_xr.attrs)
    return data_xr

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


def load_ncdf(path) -> xr.DataArray:
    """Jednolinijkowy loader dla skryptów."""
    return load_xarray_from_netcdf(str(path))

def task_regions(data_xr) -> list[dict]:
    """Skrót do _build_task_regions_from_xarray."""

    return _build_task_regions_from_xarray(data_xr)

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


def _sanitize_netcdf_attrs_inplace(attrs: dict) -> None:
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


def _decode_json_attrs_inplace(attrs: dict) -> None:
    for key in list(attrs.keys()):
        attrs[key] = _try_decode_json_attr_value(attrs[key])

def _build_task_regions_from_xarray(data_xr: xr.DataArray) -> list[dict]:
    """Build plotting regions from task-event metadata stored on an xarray export."""
    task_events = data_xr.attrs.get("task_events_structure", [])
    if isinstance(task_events, str):
        task_events = json.loads(task_events)

    regions = []
    for idx, event in enumerate(task_events or []):
        if not isinstance(event, dict):
            continue
        start_s = float(event.get("start_s", 0.0))
        duration_s = float(event.get("duration_s", 0.0))
        regions.append({
            "span": (start_s, start_s + duration_s),
            "name": str(event.get("name", f"event_{idx + 1}")),
        })
    return regions


