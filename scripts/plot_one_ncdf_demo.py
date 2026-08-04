import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np

try:
    from .ncdf import load_ncdf, task_regions
    from .plot_utils import plot_xarray_signals
except ImportError:  # pragma: no cover - fallback for direct script execution
    from src.ncdf import load_ncdf, task_regions
    from src.plot_utils import plot_xarray_signals

ncdf_file_path = Path("/Users/admin/Library/CloudStorage/GoogleDrive-j.zygierewicz@uw.edu.pl/.shortcut-targets-by-id/1N4ySQ5GO6UE8fY2jnRkRUjBFm4XHrBRv/SYNCC-IN/WP4          - Joint study/UniWAW Data collection/UNIWAW_EEG_exported_BY_TASKS/RMSSD/W_000/child/W_000_RMSSD_ch_passive_movies.nc")  # noqa: E501
#Path("/Users/admin/Library/CloudStorage/GoogleDrive-j.zygierewicz@uw.edu.pl/.shortcut-targets-by-id/1N4ySQ5GO6UE8fY2jnRkRUjBFm4XHrBRv/SYNCC-IN/WP4          - Joint study/UniWAW Data collection/UNIWAW_EEG_exported_BY_TASKS/ICA_output/EEG_ICA_CLEANED/W_000/W_000_EEG_ch_passive_movies_cleaned.nc")
modality = Path(ncdf_file_path).stem.split('_')[2]   # 'EEG', 'ECG', 'IBI', 'RMSSD', 'ET'
# ── load ─────────────────────────────────────────────────────────────────
data_xr = load_ncdf(ncdf_file_path)

# ── build regions from embedded event metadata ────────────────────────────
regions = task_regions(data_xr)

# ── pull event duration and margin from attrs ─────────────────────────────
event_duration = float(data_xr.attrs.get("task_duration", np.nan))
time_margin_s  = float(data_xr.attrs.get("time_margin_s", 0.0))
dyad_id        = data_xr.attrs.get("dyad_id", "")
who            = data_xr.attrs.get("who", "")
task           = data_xr.attrs.get("task_name", ncdf_file_path.stem)

title = f"{dyad_id}  {who}  —  {task} {modality}"

# ── plot ──────────────────────────────────────────────────────────────────
fig, ax = plot_xarray_signals(
    data_xr,
    regions=regions,
    event_duration=event_duration if np.isfinite(event_duration) else None,
    time_margin_s=time_margin_s,
    normalize= True,
    title=title,
)

print(f"File    : {ncdf_file_path.name}")
print(f"Shape   : {dict(data_xr.sizes)}")
print(f"Time    : {float(data_xr.coords['time'].min()):.2f}s "
        f"→ {float(data_xr.coords['time'].max()):.2f}s")
print(f"Regions : {[r['name'] for r in regions]}")

plt.show()
 