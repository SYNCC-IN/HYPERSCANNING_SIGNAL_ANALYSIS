# Interbrain ffDTF + HRV pipeline — staged implementation plan

Target hypotheses: **H2** (temporo-parietal social-attention coupling) and **H4**
(autonomic co-regulation), implemented as one 4-variable MVAR / ffDTF model per dyad ×
film, tested against surrogate dyads and modelled at the group level in `brms`.

This document is the **contract** for building that pipeline. It is written so that each
stage can be implemented in an **independent chat** to save context/tokens. When you open
a stage in a fresh chat, load exactly these four things and you have everything you need:

1. this file (`docs/pipeline_plan.md`),
2. `CLAUDE.md` (repo conventions — binding),
3. `docs/function_reference.md` (catalog of the existing `src/` library),
4. the project note *"DTF interbrain + HRV (H2, H4)"* (the scientific rationale).

Then read only the section for your stage plus §1–§5 below.

---

## 0. How to use this plan (independent-chat workflow)

- **One stage per chat.** Each stage in §6 lists: *Read first · Inputs · Reuse · Build ·
  Script · Output · Gate artifact · Pass criterion · Hands off*. That is the full task.
- **Every stage ends with a gate artifact** — a check that the stage's output is correct
  before anything downstream trusts it. This is deliberate: for MVAR/DTF on envelopes with
  a child/adult centre-frequency mismatch, a wrong result can look perfectly reasonable, so
  we validate against ground truth (Stage 0) and against the signal itself (per-stage QC)
  rather than assuming.
- **Stages hand off through files**, not through chat memory. The intermediate layout in
  §4.5 is the interface between stages. Stage N reads Stage N−1's files and writes its own.
- **Before trusting any signature** quoted here or in `function_reference.md`, `grep` it in
  the actual `src/*.py` — the reference can go stale (per CLAUDE.md).

---

## 1. Scientific goal (condensed — full detail in the project note)

**Model.** One MVAR per dyad × film over **4 variables**, z-scored per channel in time,
downsampled to a common rate (~2.5 Hz, Nyquist ~1.25 Hz — set by the raw IBI, see below):

| Variable     | Signal                                                                 |
|--------------|------------------------------------------------------------------------|
| `child:ROI`  | child fast-band **instantaneous-amplitude envelope**, individual band from `band_assignments.csv` |
| `cg:ROI`     | caregiver fast-band envelope, individual band                          |
| `child:HRV`  | child **raw (interpolated) IBI**, downsampled only — no HF band-pass, no Hilbert |
| `cg:HRV`     | caregiver raw (interpolated) IBI, downsampled only                     |

**HRV is the raw IBI, not an HF-band envelope (reversed from the original project note).**
On inspection of real data, the EEG rhythm envelopes fluctuate in a band that overlaps the
raw IBI (RSA, ~0.2–1 Hz), while the *envelope of HF-IBI* is a second-order, much slower
signal that no longer sits in that band. Feeding the raw IBI keeps both modalities in a
comparable band at the shared low rate. Consequence, accepted explicitly: the EEG side is a
second-order quantity (amplitude envelope of a fast rhythm) while the HRV side is a
first-order oscillation (the IBI itself) — internally consistent within each modality, but
relevant to interpreting the exploratory cross brain–heart edges below. The age-adjusted HF
band (child ~0.24–1.04 Hz, caregiver ~0.15–0.40 Hz) is kept only as recorded metadata
(`hf_reference` in the Stage 2 output attrs), describing where the raw IBI's RSA content is
expected to sit — it is never used to filter anything.

`ROI` is **P7/P8** (TPJ proxy) for the primary analysis and **Fz** (frontal-midline) for the
comparison track. ROI is a **config choice** — the pipeline must make swapping it trivial
(see §3).

**Edges & hypotheses.** 4 variables → 12 directed edges. Primary:
- `cg:ROI → child:ROI` — **H2 primary** (+ reverse edge = asymmetry test)
- `cg:HRV → child:HRV` — **H4 primary** (+ reverse edge = asymmetry test)
- `cg:HRV → child:ROI` and `child:HRV → cg:ROI` — **exploratory** cross brain–heart.

**Estimator.** ffDTF (full-frequency DTF). Start with the estimator already in the repo
(`src.mtmvar.full_freq_dtf`); windowed-ACF sDTF and a regularised Bayesian MVAR are later
swap-ins behind the same interface.

**Dependent variable.** ΔffDTF vs surrogate dyads (same film, mismatched pair) **and**
`z_vs_surrogate`, computed **per film** — this removes the common-stimulus component (both
partners watch the same movie).

**Group model (R / brms).**
```
delta_dtf ~ film * group + (1 | dyad) + (1 | child) + (1 | caregiver)
```
`family = student()`; planned contrast `Incredibles vs (Peppa + Brave)/2`; inference by
posterior probability + HDI. Films are three **qualitatively different** valence conditions
(Peppa = happy/low-arousal, Incredibles = conflict/high-arousal, Brave = mixed), **not** an
ordered scale — never average across films.

---

## 2. Repo conventions (from CLAUDE.md — binding on every stage)

- **`src/` = library, `scripts/` = pipelines.** `src/` holds simple, reusable, well-named
  functions; `scripts/` composes them into pipelines. New reusable logic goes in `src/`.
- **Simple, no defensive code.** Prefer a runtime error over silently swallowing a problem.
  No catch-all `try/except`, no silent fallbacks, no "just in case" validation. An error is a
  signal to rethink, not to mask.
- **Every function has a docstring.**
- **No hard-coded constants or paths inside functions.** All paths, bands, frequencies,
  thresholds, ROI definitions live as **config at the top of the script** and are passed in
  as arguments. This keeps `src/` free of environment-specific literals.
- **Documentation in English.**
- Functions stay small and composable so pipelines read as a sequence of calls.

---

## 3. Cross-cutting decisions (locked in for all stages)

- **Estimator: windowed-ACF-averaged sDTF (Kamiński core) is the default, not a deferred
  swap-in.** Each per-film design matrix is cut into short windows (`src.design.window_stack`),
  detrended per window (`detrend_windows`), and stacked on a trials axis; `src.mtmvar.ar_coeff`'s
  existing `count_corr`-based averaging over that axis produces **one** MVAR estimate per film
  from the averaged autocovariance. `full_freq_dtf`/`multivariate_spectra` then run on that
  identical 3-D stack with an **explicit** model order (`optimal_model_order=p_used`; **never
  `None`** on 3-D input — `mvar_criterion` crashes on it, see Stage 3). Rationale: the HRV
  variables are non-stationary within a 60 s film (drifting RSA centre frequency and variance),
  biasing a single global 2-D fit; short windows are closer to locally stationary, and the
  averaged fit empirically pulls AR roots inside the unit circle and fixes residual whiteness
  for the HRV variables specifically (confirmed on real data at Stage 3, see below) — a
  Bayesian-regularised MVAR remains a future swap-in behind the same 3-D-stack interface if
  needed on top of this.
- **Z-scoring:** the design matrix entering MVAR is **z-scored per channel in time** — the
  same convention `src.mne_bridge.load_eeg_signals` already applies to raw EEG. Apply it once,
  **globally per film** at design-matrix assembly (`src.design.assemble_design_matrix`,
  Stage 3), before windowing — never per-window (that would remove the cross-window variance
  drift the averaged-ACF fit is meant to absorb). The persisted Stage 2 envelope stays in
  physical units for QC.
- **Envelope-then-segment ordering (critical):** on the **continuous** `passive_movies`
  chunk, run filter + Hilbert + downsample for the EEG-band envelope, and downsample-only for
  the raw IBI, *then* cut per-film fragments from the event structure. Each signal file has a
  **20 s margin** before the first / after the last event and **~10 s gaps** between films, so
  filter/Hilbert/anti-alias transients fall in the margins/gaps and are discarded by
  segmentation. Never filter (or downsample) a pre-cut film.
- **ROI as config:** the per-person "brain" variable is an ROI-reduced envelope. The ROI
  (channel set) is a script-level config resolved via `src.roi.define_rois_theory()`, passed
  as an argument. Swapping P7/P8 → Fz must be a one-line config change, no code edit.
- **Gate-artifact principle:** when validating the estimator or the signal itself, generate
  the gate with the **repo's own functions** (matplotlib: `mvar_plot`, envelope/QC plots) so
  the real code path is under test. When the value is in browsing many cases (coverage,
  per-subject QC, null distributions), an interactive HTML artifact is fine.
- **Known repo issue:** `src.mtmvar.compute_and_plot_mvar` is **broken** (bad internal
  import). Do **not** use it. Call `full_freq_dtf`, `multivariate_spectra`, `mvar_plot`
  directly on a signal array.
- **Prior art, not to extend:** `src/eeg_alpha_ibi_ffdtf.py` (`EEG_IBI_FFDTF_Pipeline`) already
  does EEG-alpha-envelope + IBI multivariate ffDTF (global + windowed). Use it as
  **inspiration only** — extract simpler, reusable functions into `src/` per §2; do not build
  on the monolith.

---

## 4. Data contracts

### 4.1 Cleaned EEG (input to the whole analysis)
- Location: `UNIWAW_EEG_exported_BY_TASKS/ICA_output/EEG_ICA_CLEANED/<dyad_id>/…_passive_movies_cleaned.nc`
  (absolute root is a script-level config value).
- Loader: `src.io_utils.load_eeg_nc(filepath)` → dict with `data` (chan × time, µV),
  `channel_names`, `sfreq`, `time`, `dyad_id`, `role_code`, `role`, `movies` (per-movie
  boundaries within the `passive_movies` chunk), `age_months`, `group`, `sex`.
- File discovery: `src.io_utils.get_participant_files(data_dir)` → one row per participant
  (`filepath`, `dyad_id`, `role_code`, `role`).

### 4.2 IBI (confirmed contract)
- IBI is exported **per-task as NCDF on the EEG time grid, with the same event structure as
  EEG** (already interpolated to the EEG sampling rate and synchronised). Expected layout,
  mirroring the by-task EEG export:
  `UNIWAW_EEG_exported_BY_TASKS/IBI/<dyad_id>/<child|caregiver>/<dyad_id>_IBI_<ch|cg>_passive_movies.nc`
  (grep/confirm the exact root once, then put it in script config).
- Loader: `src.io_utils.load_ibi_nc(path)` (thin wrapper over `src.netcdf_io`'s core; was
  `src.ncdf.load_ncdf(path)`). Because the grid matches EEG, reuse the **movie boundaries from
  the EEG load** (§4.1 `movies`) — no separate alignment.
- **Out of scope:** the SECORE / H10 IBI branch (`src.secore_loader`, `src.secore_utils`) is a
  different part of the experiment. Do **not** use it here.

### 4.3 Individual bands
- `04_band_assignment/band_assignments.csv` — per participant × ROI: `slow_cf`, `fast_cf`,
  `bw` (produced by `src.bands.assign_bands_all_rois`). The EEG variable uses `fast_cf ± bw/2`
  for the chosen ROI.
- `04_band_assignment/iaf_metrics.csv` — IAF metrics + dyadic child–caregiver distance
  (covariate for the group model), from `src.bands.compute_iaf_metrics`.

### 4.4 Movie boundaries & event structure
- Within the `passive_movies` chunk, per-film boundaries come from `movies` (§4.1) or from the
  `task_events_structure` attr via `src.netcdf_io.task_regions(data_xr, reference='relative')`
  (also drives plotting `regions=`; `reference='relative'` is the default and matches the
  chunk-relative `time` coordinate -- see the netcdf_io core refactor). Films: `Peppa`,
  `Incredibles`, `Brave`.

### 4.5 Intermediate layout (the hand-off interface between stages)
Root `<ANALYSIS_ROOT>` is a script-level config value.
```
<ANALYSIS_ROOT>/
  01_coverage/coverage.csv                    # dyad × film × modality × role presence + film lengths
  02_envelopes/<dyad>_<film>.nc               # xarray [variable, time]; 4 vars, common downsampled rate; NOT z-scored
                                              #   coords: variable=[child:ROI, cg:ROI, child:HRV, cg:HRV]
                                              #   *:ROI = EEG-band envelope; *:HRV = raw (interpolated) IBI, downsampled only
                                              #   attrs: fs, roi, band params, hrv_signal="raw_ibi", hf_reference (metadata only),
                                              #          film, dyad, group, age_months
  03_mvar/<dyad>_<film>_order.json            # p_eeg, p_hrv, p_used; residual-whiteness + AR-stability diagnostics
  04_ffdtf/<dyad>_<film>.npz                  # ffdtf (k,k,nf), spectra (k,k,nf), freqs, var_order, film
  05_surrogate/<film>_null.npz                # null distributions per edge (surrogate dyads, same film)
                 <dyad>_<film>_delta.npz      # delta_ffdtf, z_vs_surrogate per edge
  06_group/                                   # brms inputs/outputs (R)
```
Variable order is fixed as `[child:ROI, cg:ROI, child:HRV, cg:HRV]` everywhere so edge
indexing is stable across stages.

---

## 5. Existing functions map (the subset each stage builds on)

Full catalog: `docs/function_reference.md`. The functions this pipeline actually uses:

- **Load / segment:** `io_utils.get_participant_files`, `io_utils.load_eeg_nc`,
  `io_utils.load_ibi_nc`, `io_utils.trim_to_event_window`, `netcdf_io.load_ncdf`,
  `netcdf_io.task_regions` (renamed from `ncdf.*`).
- **ROI:** `roi.define_rois_theory` (P7/P8, Fz, …), `roi.average_psd_within_roi` (PSD-space;
  for envelopes use `envelopes.average_channels`).
- **Bands:** `bands.assign_bands_all_rois`, `bands.compute_iaf_metrics` (already produce the
  CSVs in §4.3 — normally you just read the CSVs).
- **Envelopes:** `src.design.roi_band_envelope` (EEG-band envelope on a continuous ROI signal,
  honouring the ROI-reduction switch, §6 Stage 2), built on `filter_individual_band`,
  `hilbert_envelope`, `downsample`, `average_channels`. The HRV variable is `downsample`
  applied directly to the raw IBI — no band-pass, no Hilbert (`envelopes.hrv_hf_envelope`
  still exists and is used only if an HF-envelope comparison variant is wanted, not the
  default pipeline). QC plots `plot_signal_filtered_envelope`, `plot_dyad_envelopes`,
  `plot_eeg_hrv_envelopes` (the latter fed z-scored inputs for QC, §Stage 2).
- **MVAR / DTF:** `mtmvar.mvar_criterion(data, max_model_order, crit_type)`,
  `mtmvar.ar_coeff`, `mtmvar.mvar_transfer_function`,
  `mtmvar.full_freq_dtf(signals, freqs, fs, …)`,
  `mtmvar.multivariate_spectra(signals, freqs, fs, …)`,
  `mtmvar.mvar_plot(on_diag, off_diag, freqs, x_label, y_label, chan_names, top_title)` —
  the grid with **diagonal = spectra, off-diagonal = ffDTF, columns = source, rows = target**,
  `mtmvar.graph_plot` (directed-graph view). **Avoid** `compute_and_plot_mvar` (broken).
- **Signal convention reference:** `mne_bridge.load_eeg_signals` (shows the z-score-per-channel
  convention; note its filter-before-trim behaviour, the analogue of our envelope-then-segment
  rule).

---

## 6. Stages

Legend for each stage: **Read first · Inputs · Reuse · Build (new in `src/`) · Script ·
Output · Gate · Pass · Hands off.**

### Stage 0 — Synthetic MVAR harness (validation infrastructure)
The only fully new-code stage; unblocks trusting the estimator before it touches real data.

- **Read first:** §1, §3, §5. No real data needed.
- **Inputs:** none (generates its own signals).
- **Reuse:** `full_freq_dtf`, `multivariate_spectra`, `mvar_plot`, `graph_plot`,
  `mvar_criterion`; for the envelope-vs-phase demo `filter_individual_band`,
  `hilbert_envelope`.
- **Build (`src/synthetic_mvar.py`):** a generator producing a 4-variable coupled AR process
  with a **specified coupling/directionality matrix**, **specified per-node centre
  frequencies** (to mimic child ~8–10 Hz vs adult ~10–12 Hz), and **specified SNR** — all as
  arguments, no literals inside.
- **Script (`scripts/stage00_synthetic_validation.py`):** inject a known matrix → run the real
  `full_freq_dtf` + `multivariate_spectra` → render with `mvar_plot`; separately compare
  envelope-based vs phase-based ffDTF as the child/adult CF gap grows (6–14 Hz).
- **Gate (repo-fn, matplotlib):** (a) injected vs recovered ffDTF matrix side by side;
  (b) `mvar_plot` grid on synthetic data; (c) envelope vs phase ffDTF across the CF gap.
- **Pass:** envelope ffDTF recovers the injected directionality regardless of CF gap; the
  **source/target orientation of `full_freq_dtf` output is confirmed** (matches
  columns=source / rows=target); the phase-based version shows the expected CF-gap bias.
- **Hands off:** a trusted estimator call + a verified orientation convention reused by
  Stage 4, plus the empirical figure justifying the envelope decision (methods-ready).

### Stage 1 — Assemble data (composition of existing loaders)
Synchronisation already exists in the NCDF files (§4.2) — this stage does **not** build
alignment.

- **Read first:** §4.1, §4.2, §4.4, §5.
- **Inputs:** cleaned EEG (§4.1); IBI per-task NCDF (§4.2).
- **Reuse:** `get_participant_files`, `load_eeg_nc` (gives `movies`, `group`, `age_months`),
  `io_utils.load_ibi_nc`, `netcdf_io.read_core_attrs`, `netcdf_io.task_regions`
  (renamed from `ncdf.*`), `roi.define_rois_theory`.
- **Build (`src/assemble.py`):** a thin loader returning a per-dyad×film container
  `{EEG-ROI channels, IBI, movie window, meta}` on the shared grid — pure composition of the
  above, ROI passed as an argument. No new alignment logic.
- **Script (`scripts/stage01_coverage.py`):** iterate dyads, build containers, emit the
  coverage table.
- **Output:** `01_coverage/coverage.csv`.
- **Gate (interactive):** coverage matrix dyad × film × modality × role; a clickable timeline
  of one dyad showing film boundaries + gaps (via `task_regions`); a check that the ROI
  channels survived ICA.
- **Pass:** coverage matches expectation (~46 dyads × 3 films × 2 modalities × 2 people); film
  boundaries sensible; no unintended drops.
- **Hands off:** the confirmed set of usable dyad×film cases and the loader other stages call.

### Stage 2 — Individual-band envelopes + raw IBI (functions exist; the ordering is the new part)
- **Read first:** §3 (envelope-then-segment, ROI-as-config), §4.3, §4.4, §5.
- **Inputs:** Stage 1 loader + `coverage.csv`; `band_assignments.csv`.
- **Reuse:** `src.design.roi_band_envelope`, `filter_individual_band`, `hilbert_envelope`,
  `downsample`, `average_channels`; QC plots `plot_signal_filtered_envelope`,
  `plot_eeg_hrv_envelopes` (z-scored inputs for QC). **Inspiration:** `eeg_alpha_ibi_ffdtf.py`
  (extract simpler functions, don't extend it).
- **Build (`src/design.py`):** `roi_band_envelope` (EEG-band envelope honouring the
  ROI-reduction switch), `segment_signal` (segments any continuous downsampled signal — used
  for both the EEG envelope and the raw IBI), `stack_design_variables`. All
  bands/orders/target_sfreq/ROI are arguments; no `src/` literals.
- **Script (`scripts/stage02_envelopes.py`):** config block (paths, ROI, `target_sfreq`≈2.5 Hz
  — set by the raw IBI's RSA content, not the EEG envelope, see below — filter order,
  age-adjusted HF-reference bands kept as metadata only) → write one file per dyad×film.
- **HRV operationalisation (reversed from the original project note):** the HRV variable is
  the **raw (interpolated) IBI, downsampled only** — no band-pass, no Hilbert — not an HF-band
  envelope. See §1 for the rationale (comparable band to the EEG envelope) and the accepted
  order-asymmetry consequence. `TARGET_SFREQ` is therefore set by the raw IBI's RSA content
  (up to ~1.04 Hz for children), not by the EEG envelope, pushing the shared rate from the
  original ~2 Hz proposal to ~2.5 Hz (Nyquist ~1.25 Hz).
- **Output:** `02_envelopes/<dyad>_<film>.nc` (4 variables × time, physical units, **not**
  z-scored — z-scoring is a Stage 3 concern; QC plots z-score for display only, never
  persisted).
- **Gate (repo-fn):** ROI PSD with the individual band overlaid; raw vs filtered vs envelope on
  the retained window; a **z-scored PSD comparison of all four variables on one axis** — checks
  both no aliasing above the new Nyquist *and* that the EEG envelope and raw IBI actually
  occupy a comparable band (the point of the HRV reversal above); EEG envelope + raw IBI trace,
  z-scored; visual confirmation that filter/anti-alias edges landed in the discarded margin/gap
  for all four continuous signals. Subject/dyad selector.
- **Pass:** bands sit on real peaks; no filter/Hilbert edge artefacts in the retained window; no
  aliasing after downsampling; EEG envelope and raw IBI share a comparable band in the z-scored
  PSDs.
- **Hands off:** the per-dyad×film design-variable files — the substrate for MVAR.
- **Open decision (confirm at start of this stage):** ROI reduction order — average the
  band-filtered channels *then* Hilbert, vs Hilbert per channel *then* average envelopes.
  Default proposal: **average instantaneous envelopes across ROI channels** (robust to
  within-ROI phase differences, consistent with the envelope rationale). Flag, don't silently
  pick.
- **Known caveat (flag in QC, don't silently fix):** the raw IBI is not detrended — it can carry
  a large LF/VLF trend and DC offset. If the per-film PSD shows LF/VLF dominating the 60 s
  window, note it; a high-pass/detrend would be a Stage 3 design-matrix decision, not a silent
  Stage 2 step.

### Stage 3 — Design matrix, model order, and windowed-ACF-averaged MVAR fit
- **Read first:** §3 (estimator + z-scoring), §4.5, §5.
- **Inputs:** `02_envelopes/<dyad>_<film>.nc`.
- **Reuse:** `mvar_criterion` (AIC/HQ/SC, 2-D only), `ar_coeff` (averages over a trials axis for
  free via `count_corr` when given 3-D input), `mvar_transfer_function`.
- **Build:**
  - `src/design.py`: `assemble_design_matrix(envelopes, zscore=True)` → `(4, n_samp)` in the
    fixed variable order, z-scored per channel in time (single source of truth reused by
    Stage 4); `window_stack(design, win_len, step)` → `(4, win_len, n_windows)`, discarding a
    trailing partial window; `detrend_windows(stack, dtype='linear')` → per-window mean+slope
    removal (not variance), applied after windowing and after the global z-score, never to the
    whole film.
  - `src/mvar_diag.py`: `fit_mvar_avg_acf(design_3d, p)` (thin wrapper over `ar_coeff` on the
    detrended 3-D stack — the averaging is the library's, this just documents the Kamiński
    averaged-ACF semantics); `residual_whiteness(design_3d, ar_coeffs, max_lag)` (per-window
    residuals, no cross-window prediction, ACF averaged across windows, Bartlett-band
    whiteness summary); `ar_root_stability(ar_coeffs)` (companion-matrix eigenvalues, max
    modulus, stability flag). Both diagnostics work identically on the windowed 3-D stack or a
    global design reshaped to one window, so the same functions serve the global-vs-windowed
    gate comparison — genuinely new code, not in the library before.
- **Script (`scripts/stage03_mvar_order.py`):** select `p_used` on the **global 2-D** design
  (`mvar_criterion` cannot take 3-D — documented caveat, not patched); report `p_eeg`/`p_hrv`
  sub-block orders diagnostically only (`p_used` is always the shared joint-system order, since
  the exploratory cross brain-heart edges only exist in the joint model); window/detrend/fit;
  run both diagnostics on the windowed fit *and* a global single-window comparison fit; write
  the order JSON + manifest; run a small window-choice sensitivity check (1-2 dyads, `10 s/50%`
  vs `15 s/50%` vs `10 s/0%`, `p_used` held fixed) comparing the primary ffDTF edges.
- **Locked window geometry:** `WIN_LEN_S = 10`, `OVERLAP_FRAC = 0.5` → 25 samples/window, 11
  windows per 60 s film at 2.5 Hz. `1/WIN_LEN_S = 0.1 Hz` stays below the ~0.2-1 Hz coupling
  band (so linear detrend removes only sub-band drift); confirmed, not just asserted, by the
  sensitivity check.
- **Output:** `03_mvar/<dyad>_<film>_order.json`, `03_mvar/stage03_manifest.csv`,
  `03_mvar/sensitivity_band_averages.csv`.
- **Gate:** AIC/HQ/SC curves (full system + EEG/HRV sub-blocks, global 2-D signal); AR roots on
  the unit circle, **global vs windowed fit overlaid**; residual ACF, **global vs windowed**,
  per variable; example windows pre-/post-detrend; window-choice sensitivity spectra + a
  band-averaged table for the primary edges.
- **Pass:** windowed residuals white at `p_used`, AR roots inside the unit circle, a deliberate
  `p` decision recorded, the HRV-variable improvement over the global fit visible, and the
  primary edges stable across the window-sensitivity configs.
- **Empirical result (2026, real data, all 126 dyad×film cases):** the windowed-ACF-averaged fit
  reached `quality_ok` (stable + white) on **126/126** cases, vs the global single-window fit's
  **0/126** (residual whiteness failed for every case globally — confirmed not a bug via an
  independent `statsmodels.VAR` cross-check and by refitting near the order cap). This is the
  empirical case for making the windowed method the default rather than a later swap-in.
- **Hands off:** the order + the detrended, windowed 3-D stack + a clean list of dyad×film
  cases fit to estimate.

### Stage 4 — ffDTF estimation (estimator already in repo; input is now the windowed 3-D stack)
- **Read first:** §3 (estimator + windowed-ACF-averaged default), Stage 0 result, Stage 3, §5.
- **Inputs:** `02_envelopes` (via `assemble_design_matrix` → `window_stack` → `detrend_windows`,
  identical to Stage 3) + `03_mvar` order (`p_used`).
- **Reuse:** `full_freq_dtf`, `multivariate_spectra`, `mvar_plot`, `graph_plot`.
- **Build (`src/connectivity.py`):** a thin wrapper `estimate_ffdtf(design_3d, freqs, fs, p_used)`
  calling `full_freq_dtf`/`multivariate_spectra` with **`optimal_model_order=p_used` always
  explicit — never `None`** (`mvar_criterion`, which the `None` path calls, crashes on 3-D
  input). No new DTF maths; the only change from a v1 2-D design is the 3-D windowed stack and
  the always-explicit order. A **Bayesian MVAR** can still replace the internals later behind
  this same signature.
- **Script (`scripts/stage04_ffdtf.py`):** per dyad×film → ffDTF cube + spectra, on the
  detrended windowed stack.
- **Output:** `04_ffdtf/<dyad>_<film>.npz`.
- **Gate (repo-fn):** `mvar_plot` grid for the 12 edges per dyad×film, shown **next to** the
  Stage 0 synthetic result so the numbers are anchored to a known-truth case.
- **Pass:** ffDTF on synthetic reproduces truth; real values in a sensible range; no
  normalisation artefacts.
- **Hands off:** per-dyad×film ffDTF cubes for the surrogate/Δ stage.

### Stage 5 — Surrogates and ΔffDTF / z_vs_surrogate (new code)
- **Read first:** §1 (DV definition), §4.4, §4.5.
- **Inputs:** `02_envelopes` (to re-pair) or `04_ffdtf` cubes; film labels.
- **Reuse:** `full_freq_dtf` / the Stage 4 wrapper on mismatched pairs.
- **Build (`src/surrogate.py`):** surrogate-dyad generation **matched by film** (foreign
  child/caregiver watching the same movie), null distribution per edge, ΔffDTF, and
  `z_vs_surrogate` normalisation.
- **Script (`scripts/stage05_surrogate.py`):** build nulls per film, compute Δ and z per
  dyad×film×edge.
- **Output:** `05_surrogate/<film>_null.npz`, `05_surrogate/<dyad>_<film>_delta.npz`.
- **Gate (interactive):** null distribution per edge with the real value overlaid; a pairing
  sanity check (never a real dyad; film always matches).
- **Pass:** pairing correct; common-stimulus component demonstrably removed; Δ departs (or
  not) from null sensibly.
- **Hands off:** the tidy `delta_dtf` / `z_vs_surrogate` table for the group model.

### Stage 6 — Group model (R / brms)
- **Read first:** §1 (model formula, planned contrast).
- **Inputs:** Stage 5 Δ / z table + `iaf_metrics.csv` covariate.
- **Build (`scripts/stage06_group_model.R`):** `delta_dtf ~ film * group + (1|dyad) +
  (1|child) + (1|caregiver)`, `family = student()`, weakly-informative priors; planned
  contrast `Incredibles vs (Peppa + Brave)/2`; posterior probability + HDI; mild FDR on the
  primary H2/H4 contrasts.
- **Output:** `06_group/` posterior summaries + figures.
- **Gate:** forest plots + HDI of the primary contrasts (H4 with the sign flagged as
  informative, not assumed TD>ASD); `pp_check`, LOO, Rhat/ESS table — the gate for k=4
  vs the 2-variable fallback.
- **Pass:** MCMC converges (Rhat < 1.01, ESS > 400), `pp_check` sensible; otherwise a
  deliberate drop to the 2-variable model.

---

## 7. Open decisions still to confirm (not blocking Stage 0)

1. **ROI reduction order** (Stage 2): average band-filtered channels then Hilbert, vs Hilbert
   per channel then average envelopes. Default: average envelopes.
2. **Downsample target**: ~2.5 Hz (Nyquist ~1.25 Hz) — **resolved**, set by the raw IBI's RSA
   content (child HF-reference up to ~1.04 Hz) now that HRV = raw IBI rather than the original
   ~2 Hz envelope-only proposal.
3. **HRV operationalisation** (Stage 2): raw (interpolated) IBI, downsampled only — **resolved,
   reversing the original project note's HF-band-envelope choice** (see §1 and Stage 2). An
   HF-envelope comparison variant remains available via `envelopes.hrv_hf_envelope` if wanted
   later, but is not the default pipeline.
4. **Model order `p`**: shared across EEG/HRV vs separate — **resolved: shared, fit on the joint
   4-variable system** (required for the exploratory cross brain-heart edges). `p_eeg`/`p_hrv`
   sub-block orders are still reported per case, diagnostically only, to expose a mismatch.
5. **sDTF swap-in window scheme** — **resolved, and no longer a "later" swap-in: windowed-ACF-
   averaged sDTF (Kamiński core) is now the Stage 3/4 default estimation path**, not deferred.
   Window geometry locked at `WIN_LEN_S = 10`, `OVERLAP_FRAC = 0.5` (25 samples/window, ~11
   windows per 60 s film at 2.5 Hz), confirmed (not just asserted) by a window-choice
   sensitivity check comparing `10 s/50%` against `15 s/50%` and `10 s/0%` on the primary ffDTF
   edges. Order selection (`p_used`) still runs on the global 2-D signal, since `mvar_criterion`
   does not support 3-D input and is not being patched (see Stage 3). A Bayesian-regularised
   MVAR remains a possible future addition on top of this, behind the same 3-D-stack interface.

---

## 8. Build order

Stage 0 first (unblocks trust in the estimator), then 1 → 2 → 3 → 4 → 5 → 6 in sequence.
Stage 0 can proceed fully in parallel with Stage 1, since it needs no real data.
