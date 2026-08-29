"""Stage 1 - assemble data + coverage table.

First real-data stage of the interbrain ffDTF + HRV pipeline
(`DTF_analysis_notes/pipeline_plan.md`). Composes existing loaders
(`src.io_utils.get_participant_files`, `src.assemble.assemble_dyad`) into a
per-dyad container per `dyad_id`, then emits one coverage row per
`(dyad_id, role, film, modality)` describing which cases exist and are
usable. No signal processing, no alignment, no envelopes, no MVAR -- see
Stage 2+ for those. Film windows are handed off as metadata (not cut here),
so Stage 2's filter -> Hilbert -> downsample -> segment order can keep edge
transients out of the retained per-film data.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.assemble import ROLE_CODE_OF, assemble_dyad
from src.io_utils import ensure_dir, get_participant_files

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DRIVE_ROOT = Path(
    "/Users/admin/Library/CloudStorage/GoogleDrive-j.zygierewicz@uw.edu.pl/"
    "Mój dysk/SYNCC-IN/WP4          - Joint study/UniWAW Data collection"
)
EEG_CLEANED_ROOT = DRIVE_ROOT / "UNIWAW_EEG_exported_BY_TASKS" / "ICA_output" / "EEG_ICA_CLEANED"
IBI_ROOT = DRIVE_ROOT / "UNIWAW_EEG_exported_BY_TASKS" / "IBI"

ANALYSIS_ROOT = PROJECT_ROOT / "Interbrain_ffDTF_analysis"
OUTPUT_DIR = ensure_dir(ANALYSIS_ROOT / "01_coverage")

# ROI as config -- P7/P8 (TPJ proxy) is the primary track; swap to
# ROI_LABEL = "frontal_midline", ROI_CHANNELS = ["Fz"] for the comparison track.
ROI_LABEL = "temporo-parietal"
ROI_CHANNELS = ["P7", "P8"]

FILMS = ["Peppa", "Incredibles", "Brave"]  # set of labels to match, not a presentation order
MODALITIES = ["EEG", "IBI"]
ROLES = ["child", "caregiver"]

EXPECTED_FILM_LEN_S = (55.0, 65.0)  # QC range around the ~60 s films

# Dyad-selection criterion feeding Stage 2+ (dyad_selection.json below). A dyad
# is INCLUDED iff its group is one of INCLUDED_GROUPS, every roi_ok value
# recorded for it in coverage_df is True (IBI rows carry roi_ok = None and are
# ignored), and all films/modalities/roles are present (no gaps like a missing
# movie). Edit INCLUDED_GROUPS or the checks near the bottom of this script to
# experiment with different criteria.
INCLUDED_GROUPS = ["TD", "ASD"]

# ---------------------------------------------------------------------------
# 1. Discover participants (EEG-anchored) and assemble each dyad
# ---------------------------------------------------------------------------
participant_files = get_participant_files(EEG_CLEANED_ROOT)
dyad_ids = sorted(participant_files["dyad_id"].unique())
print(f"Discovered {len(dyad_ids)} dyads under {EEG_CLEANED_ROOT}")

dyads = {}
for dyad_id in dyad_ids:
    eeg_files = participant_files[participant_files["dyad_id"] == dyad_id]
    dyads[dyad_id] = assemble_dyad(dyad_id, eeg_files, IBI_ROOT, ROI_CHANNELS)

# ---------------------------------------------------------------------------
# 2. Build one coverage row per (dyad_id, role, film, modality)
# ---------------------------------------------------------------------------
rows = []
for dyad_id, dyad in dyads.items():
    dyad_notes = list(dyad["notes"])

    for role in ROLES:
        eeg_entry = dyad["eeg"][role]
        ibi_entry = dyad["ibi"][role]

        for film in FILMS:
            span = dyad["film_windows"].get(film)
            film_present = span is not None
            film_start_s, film_end_s = span if film_present else (None, None)
            film_len_s = (film_end_s - film_start_s) if film_present else None
            film_notes = list(dyad_notes)
            if film_present and not (EXPECTED_FILM_LEN_S[0] <= film_len_s <= EXPECTED_FILM_LEN_S[1]):
                film_notes.append(f"{film} duration {film_len_s:.1f}s outside expected range")
            if not film_present:
                film_notes.append(f"{film} missing from film_windows")

            for modality, entry in [("EEG", eeg_entry), ("IBI", ibi_entry)]:
                present = film_present and entry is not None
                sfreq = entry["sfreq"] if entry is not None else None
                n_samples_in_window = (
                    int(round(film_len_s * sfreq)) if present and sfreq is not None else None
                )

                row = {
                    "dyad_id": dyad_id,
                    "group": dyad["group"],
                    "role": role,
                    "role_code": ROLE_CODE_OF[role],
                    "film": film,
                    "film_order_idx": (
                        list(dyad["film_windows"].keys()).index(film) if film_present else None
                    ),
                    "modality": modality,
                    "present": bool(present),
                    "film_start_s": film_start_s,
                    "film_end_s": film_end_s,
                    "film_len_s": film_len_s,
                    "sfreq": sfreq,
                    "n_samples_in_window": n_samples_in_window,
                    "roi_label": ROI_LABEL if modality == "EEG" else None,
                    "roi_channels_expected": "|".join(ROI_CHANNELS) if modality == "EEG" else None,
                    "roi_channels_found": (
                        "|".join(entry["roi_found"]) if modality == "EEG" and entry is not None else None
                    ),
                    "roi_ok": entry["roi_ok"] if modality == "EEG" and entry is not None else None,
                    "age_months": dyad["meta"]["age_months"] if role == "child" else None,
                }

                reasons = []
                if film_present and entry is None:
                    reasons.append(f"{modality} file absent for this role")
                if entry is not None and modality == "EEG" and not entry["roi_ok"]:
                    missing = sorted(set(ROI_CHANNELS) - set(entry["roi_found"]))
                    if missing:
                        reasons.append(f"ROI channels missing: {','.join(missing)}")
                    if entry["roi_interpolated"]:
                        reasons.append(f"ROI channels interpolated: {','.join(entry['roi_interpolated'])}")
                if entry is not None and modality == "IBI" and not entry["grid_matches_eeg"]:
                    reasons.append("IBI time grid does not match EEG")
                row["notes"] = "; ".join(film_notes + reasons)

                rows.append(row)

coverage_df = pd.DataFrame(rows)
coverage_df.to_csv(OUTPUT_DIR / "coverage.csv", index=False)
print(f"Wrote {len(coverage_df)} coverage rows to {OUTPUT_DIR / 'coverage.csv'}")

# ---------------------------------------------------------------------------
# 3. Summary
# ---------------------------------------------------------------------------
n_dyads = len(dyads)
expected_cells_per_dyad = len(ROLES) * len(FILMS) * len(MODALITIES)

complete_mask = coverage_df.groupby("dyad_id")["present"].transform("all") & (
    coverage_df.groupby("dyad_id")["roi_ok"].transform(lambda s: s.fillna(True).all())
)
fully_complete_dyads = sorted(coverage_df.loc[complete_mask, "dyad_id"].unique())

print(f"\n=== Stage 1 coverage summary ===")
print(f"n_dyads discovered:        {n_dyads}")
print(f"expected cells per dyad:   {expected_cells_per_dyad} (role x film x modality)")
print(f"fully complete dyads:      {len(fully_complete_dyads)} / {n_dyads}")

anomalous = coverage_df[(~coverage_df["present"]) | (coverage_df["roi_ok"] == False) | (coverage_df["notes"] != "")]
anomalous = anomalous.drop_duplicates(subset=["dyad_id", "role", "film", "modality"])
print(f"\nAnomalous/missing cells ({len(anomalous)}):")
for _, r in anomalous.iterrows():
    print(f"  {r['dyad_id']} {r['role']:9s} {r['film']:12s} {r['modality']:3s} "
          f"present={r['present']!s:5s} roi_ok={r['roi_ok']!s:5s} notes={r['notes']}")

# ---------------------------------------------------------------------------
# 4. Interactive HTML gate
# ---------------------------------------------------------------------------
timelines = {}
for dyad_id, dyad in dyads.items():
    regions = [
        {"name": name, "start_s": span[0], "end_s": span[1]}
        for name, span in dyad["film_windows"].items()
    ]
    task_end_s = max((r["end_s"] for r in regions), default=0.0)
    timelines[dyad_id] = {
        "group": dyad["group"],
        "regions": regions,
        "task_end_s": task_end_s,
        "roi_ok": {
            role: (dyad["eeg"][role]["roi_ok"] if dyad["eeg"][role] is not None else None)
            for role in ROLES
        },
    }

coverage_records = coverage_df.where(pd.notnull(coverage_df), None).to_dict(orient="records")

html_data = {
    "coverage": coverage_records,
    "timelines": timelines,
    "dyad_ids": dyad_ids,
    "roles": ROLES,
    "films": FILMS,
    "modalities": MODALITIES,
    "roi_label": ROI_LABEL,
    "roi_channels": ROI_CHANNELS,
}

HTML_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Stage 1 coverage gate</title>
<style>
  body { font-family: -apple-system, sans-serif; margin: 1.5em; color: #1a1a1a; }
  h1 { font-size: 1.3em; }
  #matrix { border-collapse: collapse; font-size: 0.75em; }
  #matrix td, #matrix th { border: 1px solid #ccc; padding: 2px 4px; text-align: center; }
  #matrix th { background: #f0f0f0; position: sticky; top: 0; }
  .dyad-id { text-align: left; cursor: pointer; color: #0645ad; font-weight: 600; }
  .dyad-id:hover { text-decoration: underline; }
  .cell-ok { background: #bfe6bf; }
  .cell-issue { background: #f7e08a; }
  .cell-absent { background: #eee; color: #999; }
  #detail { margin-top: 1.5em; border-top: 2px solid #333; padding-top: 1em; }
  svg { border: 1px solid #ccc; }
  .region-label { font-size: 11px; }
  .legend span { display: inline-block; width: 12px; height: 12px; margin-right: 4px; vertical-align: middle; }
</style>
</head>
<body>
<h1>Stage 1 coverage gate</h1>
<p>ROI: <b>__ROI_LABEL__</b> (__ROI_CHANNELS__). Click a dyad_id to see its per-film timeline.</p>
<div class="legend">
  <span class="cell-ok"></span> present + roi_ok &nbsp;
  <span class="cell-issue"></span> present, ROI issue &nbsp;
  <span class="cell-absent"></span> absent
</div>
<table id="matrix"></table>
<div id="detail"></div>
<script>
const DATA = __DATA_JSON__;

function cellClass(row) {
  if (!row.present) return 'cell-absent';
  if (row.modality === 'EEG' && row.roi_ok === false) return 'cell-issue';
  return 'cell-ok';
}

function buildMatrix() {
  const table = document.getElementById('matrix');
  const cols = [];
  for (const role of DATA.roles) {
    for (const film of DATA.films) {
      for (const modality of DATA.modalities) {
        cols.push({role, film, modality});
      }
    }
  }
  const headRow1 = document.createElement('tr');
  headRow1.innerHTML = '<th>dyad_id</th>' + cols.map(c => `<th>${c.role[0]}/${c.film.slice(0,3)}/${c.modality}</th>`).join('');
  table.appendChild(headRow1);

  const byKey = {};
  for (const row of DATA.coverage) {
    byKey[`${row.dyad_id}|${row.role}|${row.film}|${row.modality}`] = row;
  }

  for (const dyadId of DATA.dyad_ids) {
    const tr = document.createElement('tr');
    const tdId = document.createElement('td');
    tdId.className = 'dyad-id';
    tdId.textContent = dyadId;
    tdId.onclick = () => showDetail(dyadId);
    tr.appendChild(tdId);
    for (const c of cols) {
      const row = byKey[`${dyadId}|${c.role}|${c.film}|${c.modality}`];
      const td = document.createElement('td');
      td.className = row ? cellClass(row) : 'cell-absent';
      td.title = row && row.notes ? row.notes : '';
      tr.appendChild(td);
    }
    table.appendChild(tr);
  }
}

function showDetail(dyadId) {
  const tl = DATA.timelines[dyadId];
  const detail = document.getElementById('detail');
  const width = 900, height = 90, margin = 40;
  const scale = (width - 2 * margin) / tl.task_end_s;
  const colors = {Peppa: '#8ecae6', Incredibles: '#ffb703', Brave: '#adb5bd'};

  let svg = `<svg width="${width}" height="${height}">`;
  svg += `<line x1="${margin}" y1="50" x2="${margin + tl.task_end_s * scale}" y2="50" stroke="#333" stroke-width="2"/>`;
  for (const r of tl.regions) {
    const x = margin + r.start_s * scale;
    const w = (r.end_s - r.start_s) * scale;
    svg += `<rect x="${x}" y="30" width="${w}" height="40" fill="${colors[r.name] || '#ccc'}" opacity="0.8"/>`;
    svg += `<text class="region-label" x="${x + w/2}" y="25" text-anchor="middle">${r.name}</text>`;
    svg += `<text class="region-label" x="${x + w/2}" y="85" text-anchor="middle">${r.end_s - r.start_s > 0 ? (r.end_s - r.start_s).toFixed(1)+'s' : ''}</text>`;
  }
  svg += '</svg>';

  const roiInfo = DATA.roles.map(r => `${r}: roi_ok=${tl.roi_ok[r]}`).join(', ');
  detail.innerHTML = `<h2>${dyadId} (group=${tl.group})</h2><p>${roiInfo}</p>${svg}`;
}

buildMatrix();
</script>
</body>
</html>
"""

html = HTML_TEMPLATE.replace("__DATA_JSON__", json.dumps(html_data, default=str))
html = html.replace("__ROI_LABEL__", ROI_LABEL).replace("__ROI_CHANNELS__", "|".join(ROI_CHANNELS))
(OUTPUT_DIR / "coverage_gate.html").write_text(html, encoding="utf-8")
print(f"\nWrote interactive gate to {OUTPUT_DIR / 'coverage_gate.html'}")

# ---------------------------------------------------------------------------
# 5. Dyad selection for downstream stages (INCLUDED_DYADS / EXCLUDED_DYADS)
# ---------------------------------------------------------------------------
dyad_group = coverage_df.groupby("dyad_id")["group"].first()
dyad_roi_all_ok = coverage_df.groupby("dyad_id")["roi_ok"].apply(lambda s: s.dropna().eq(True).all())
dyad_all_present = coverage_df.groupby("dyad_id")["present"].all()

INCLUDED_DYADS = sorted(
    dyad_id for dyad_id in dyad_ids
    if dyad_group[dyad_id] in INCLUDED_GROUPS
    and dyad_roi_all_ok[dyad_id]
    and dyad_all_present[dyad_id]
)
EXCLUDED_DYADS = sorted(set(dyad_ids) - set(INCLUDED_DYADS))

dyad_selection_path = OUTPUT_DIR / "dyad_selection.json"
dyad_selection_path.write_text(
    json.dumps({"INCLUDED_DYADS": INCLUDED_DYADS, "EXCLUDED_DYADS": EXCLUDED_DYADS}, indent=2),
    encoding="utf-8",
)
print(f"\nDyad selection (groups={INCLUDED_GROUPS}, roi_ok + full film coverage required): "
      f"{len(INCLUDED_DYADS)} included, {len(EXCLUDED_DYADS)} excluded")
print(f"Wrote dyad selection to {dyad_selection_path}")

# ---------------------------------------------------------------------------
# 6. Basic sample statistics for the included subset
# ---------------------------------------------------------------------------
included_meta = pd.DataFrame([
    {"dyad_id": dyad_id, "group": dyads[dyad_id]["group"], **dyads[dyad_id]["meta"]}
    for dyad_id in INCLUDED_DYADS
])

print(f"\n=== Included subset ({len(INCLUDED_DYADS)} dyads) sample statistics, by group ===")
for group_label, group_meta in included_meta.groupby("group"):
    age = group_meta["age_months"].dropna()
    sex_counts = group_meta["sex"].value_counts()
    n_sexed = sex_counts.sum()
    sex_str = ", ".join(
        f"{sex_code}={count} ({100 * count / n_sexed:.1f}%)" for sex_code, count in sex_counts.items()
    )
    print(f"\n{group_label} (n={len(group_meta)}):")
    print(f"  age (months): mean={age.mean():.1f} +/- {age.std():.1f}, "
          f"range=[{age.min():.0f}, {age.max():.0f}] (n={len(age)})")
    print(f"  sex: {sex_str}")
