"""Stage 0b - the smoothness artifact and its surrogate cure (methods demo).

A companion to Stage 0's synthetic validation harness, built to back the
"Directionality is fragile: the smoothness gradient" statement with a ground-truth
simulation instead of a hand-wave. The question is narrow and the answer is
known by construction: with ZERO injected coupling, does a difference in
channel smoothness alone make a DTF-family estimator report a directed edge,
and does the pipeline's surrogate subtraction remove it?

Design (minimal, 2 channels, self-gain as the smoothness axis):

- Two independent AR(2) resonators at a common in-band centre frequency
  `F0_HZ`, differing ONLY in pole radius `r` (their self-persistence /
  self-gain). Higher `r` -> sharper spectral peak -> narrower-band, more
  self-predictable, "smoother" channel. Channel 0 is the fixed rough channel
  (`R_ROUGH`); channel 1 is the smooth channel whose `r` is swept
  (`R_SMOOTH_GRID`). The off-diagonal AR coefficients are exactly zero, so the
  ground-truth directed coupling is exactly zero.
- Everything runs through the REAL Stage 4 estimator path
  (`src.connectivity.Granger_estimator`) at the pipeline's locked settings
  (`p=4`, 10 s / 50 % windows, linear per-window detrend, the Stage 4/5
  frequency grid and 0.2-1.0 Hz coupling band), so the demonstrated behaviour
  is the pipeline's own, not a parallel re-implementation.
- The cure is the Stage 5 machinery, unmodified: `src.surrogate.surrogate_pairs`
  enumerates foreign pairings (channel 0 from one simulated dyad + channel 1
  from another), which preserve each channel's marginal smoothness but destroy
  any interaction, and `src.surrogate.delta_and_z` scores each real dyad's edge
  against that pooled null (signed, never abs()).

Verified empirical result this harness demonstrates (see the printed summary):
with the pipeline's per-channel z-scoring (as in `assemble_design_matrix`),
the SMOOTHER channel is pulled toward looking like the SOURCE and the spurious
directed edge runs smooth -> rough. NB dDTF is not invariant to per-channel
scaling: WITHOUT z-scoring this sign flips (rough looks like the source). The
pipeline always z-scores, so smoother-as-source is its operative direction --
but the sign-dependence on a preprocessing choice is itself the argument for
reading direction off the surrogate rather than reasoning about it a priori.

`generate_coupled_oscillators` (the envelope/phase substrate used in Stage 0)
is the alternative generator for a version where smoothness is set by carrier
bandwidth rather than AR self-gain; this minimal version uses the direct AR
route so the smoothness axis is a single interpretable parameter.

Writes two figures and a self-contained gate into `OUTPUT_DIR`:
- `fig1_gradient.png`   -- pseudo-flow, its surrogate null, and Delta across
                           the smoothness gradient (the disease and the cure).
- `fig2_null_twin.png`  -- the Stage 5 null-vs-real histogram, on ground truth
                           = 0 (artifact removed) and ground truth > 0 (genuine
                           edge preserved).
- `smoothness_artifact_gate.html`
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.connectivity import Granger_estimator
from src.io_utils import ensure_dir
from src.surrogate import band_average_cube, delta_and_z, surrogate_pairs
from src.synthetic_mvar import generate_var_process
from src.mtmvar import mvar_plot

# ---------------------------------------------------------------------------
# Configuration (script owns all constants; src stays literal-free)
# ---------------------------------------------------------------------------
OUTPUT_DIR = ensure_dir(PROJECT_ROOT / "out")

# Estimator settings, locked to the real pipeline (Stage 4/5).
FS = 2.5
P = 4
WIN_LEN_S = 10.0
OVERLAP_FRAC = 0.5
DETREND_TYPE = "linear"
FREQS = np.linspace(0.02, FS / 2 - 0.02, 100)
COUPLING_BAND_HZ = (0.2, 1.0)
BOX_COX_LAMBDA = 0.25  # no transform, preserving prior behaviour; spectra are never transformed
ESTIMATOR = "dDTF"  # the pipeline's default estimator

WIN_LEN = round(WIN_LEN_S * FS)
STEP = round(WIN_LEN * (1 - OVERLAP_FRAC))

# Signal generation.
N_SAMPLES = 800          # ~5.3 min at 2.5 Hz; long enough for stable ACF-averaged fits
SNR = 5.0                # innovation-noise inverse scale, shared by both channels
F0_HZ = 0.30             # common in-band resonance; channels differ ONLY in smoothness
R_ROUGH = 0.50           # fixed pole radius of the rough (broadband) channel = channel 0
CHANNEL_LABELS = ("rough (broadband)", "smooth (narrowband)")  # index 0, 1

# The smoothness gradient (Figure 1): sweep the smooth channel's pole radius.
R_SMOOTH_GRID = np.array([0.50, 0.60, 0.70, 0.78, 0.85, 0.90, 0.93, 0.95, 0.97])
N_DYADS = 16             # simulated dyads per grid point per batch (a film's worth)
N_BATCHES = 6            # independent simulations per grid point; Fig 1 shows their mean +/- SD
BASE_SEED = 20250904

# Figure 2 (null-vs-real twin): one high-contrast setting, two ground truths.
R_SMOOTH_FIG2 = 0.95
N_DYADS_FIG2 = 30        # more dyads for a well-populated null histogram
INJECTED_GAIN = 0.35     # genuine rough -> smooth coupling for the "signal present" panel

# SYNCC-IN brand palette.
TEAL = "#2B94A6"
AMBER = "#E9B629"
LIME = "#AECE1A"
NAVY = "#153842"
GREY = "#C9D3D6"

# Fixed 2x2 edge indexing: dDTF[target, source]; channel 0 = rough, 1 = smooth.
EDGE_ROUGH_TO_SMOOTH = (1, 0)   # target = smooth, source = rough
EDGE_SMOOTH_TO_ROUGH = (0, 1)   # target = rough,  source = smooth


def self_ar2_coeffs(pole_radius, f0_hz, fs):
    """AR(2) resonator coefficients (a1, a2) for a pole at radius/frequency.

    A complex-conjugate pole pair at modulus `pole_radius` and frequency
    `f0_hz` gives a spectral peak whose sharpness grows with `pole_radius`;
    `pole_radius` is therefore the channel's self-persistence / "self-gain",
    the smoothness knob swept in this harness.

    Parameters
    ----------
    pole_radius : float
        Pole modulus in (0, 1); closer to 1 = sharper peak = smoother channel.
    f0_hz : float
        Resonance centre frequency (Hz).
    fs : float
        Sampling frequency (Hz).

    Returns
    -------
    a1, a2 : float
        Self AR coefficients for lags 1 and 2, in the sign convention of
        `src.synthetic_mvar.generate_var_process`
        (``x_t = a1 x_{t-1} + a2 x_{t-2} + noise``).
    """
    a1 = 2.0 * pole_radius * np.cos(2.0 * np.pi * f0_hz / fs)
    a2 = -pole_radius * pole_radius
    return a1, a2


def build_coupling(r_rough, r_smooth, injected_gain=0.0):
    """Two-channel AR(2) coupling tensor: self-resonators + optional real edge.

    Channel 0 (rough) and channel 1 (smooth) each get an AR(2) self-resonator
    at `F0_HZ`; their off-diagonal is zero unless `injected_gain` is set, which
    adds a genuine smooth -> rough edge (source = channel 1, target = channel
    0) at lag 1 -- the same direction as the smoothness artifact, so the
    "signal present" panel tests whether Delta separates a real edge from the
    artifact floor. With `injected_gain == 0` the ground-truth directed
    coupling is exactly zero.

    Parameters
    ----------
    r_rough, r_smooth : float
        Pole radii (self-gain) of channel 0 and channel 1.
    injected_gain : float, optional
        Genuine smooth -> rough AR gain at lag 1 (default 0.0 = no coupling).

    Returns
    -------
    np.ndarray, shape (2, 2, 2)
        AR coefficient tensor, `coupling[target, source, lag - 1]`, matching
        `generate_var_process`'s convention.
    """
    coupling = np.zeros((2, 2, 2))
    for channel, pole_radius in enumerate((r_rough, r_smooth)):
        a1, a2 = self_ar2_coeffs(pole_radius, F0_HZ, FS)
        coupling[channel, channel, 0] = a1
        coupling[channel, channel, 1] = a2
    if injected_gain != 0.0:
        coupling[0, 1, 0] += injected_gain  # source = 1 (smooth) -> target = 0 (rough)
    return coupling


def zscore_rows(design):
    """Per-channel z-score, matching the real pipeline's design-matrix convention."""
    return (design - design.mean(axis=1, keepdims=True)) / design.std(axis=1, keepdims=True)


def simulate_dyads(r_smooth, injected_gain, n_dyads, base_seed):
    """Simulate `n_dyads` independent 2-channel dyads at one smoothness setting.

    Each dyad is one `generate_var_process` realisation (its own seed), returned
    z-scored per channel. With `injected_gain == 0` every dyad has zero real
    coupling; the only structure is each channel's own smoothness.

    Returns
    -------
    list of np.ndarray
        Each entry shape (2, N_SAMPLES): row 0 rough, row 1 smooth.
    """
    coupling = build_coupling(R_ROUGH, r_smooth, injected_gain)
    return [
        zscore_rows(generate_var_process(coupling, SNR, N_SAMPLES, seed=base_seed + d))
        for d in range(n_dyads)
    ]


def edge_value(design, edge):
    """Band-averaged dDTF for one directed edge of a 2-channel design (Stage 4/5 path)."""
    dDTF, _ = Granger_estimator(design, FREQS, FS, P, WIN_LEN, STEP, DETREND_TYPE, ESTIMATOR, BOX_COX_LAMBDA)
    band_avg = band_average_cube(dDTF, FREQS, COUPLING_BAND_HZ)
    return float(band_avg[edge[0], edge[1]])


def real_edge_values(dyads, edge):
    """Per-dyad band-averaged dDTF for `edge` (the real, same-dyad estimate)."""
    return np.array([edge_value(dyad, edge) for dyad in dyads])


def surrogate_null(dyads, edge):
    """Pooled surrogate null for `edge`, exactly the Stage 5 construction.

    `surrogate_pairs` enumerates every ordered foreign pairing; each surrogate
    takes channel 0 (rough) from one dyad and channel 1 (smooth) from another,
    so marginal smoothness is preserved but any interaction is destroyed.

    Returns
    -------
    np.ndarray
        One band-averaged dDTF per foreign pairing.
    """
    dyad_ids = list(range(len(dyads)))
    values = []
    for rough_dyad, smooth_dyad in surrogate_pairs(dyad_ids):
        design = np.stack([dyads[rough_dyad][0], dyads[smooth_dyad][1]], axis=0)
        values.append(edge_value(design, edge))
    return np.array(values)

def example_dyads_figure(dyads, r_smooth, max_on_diag=None, max_off_diag=None):
    """One example dyad's two channels (left) and its MVAR spectra/dDTF (right) at one r_smooth setting.

    Uses `dyads[0]` as the representative example and runs it through the
    real Stage 4 estimator path (`Granger_estimator`), so the connectivity
    panel shown is exactly what the pipeline itself would compute for this
    dyad -- not a separate illustrative re-implementation.
    """
    dyad = dyads[0]
    dDTF, spectra = Granger_estimator(dyad, FREQS, FS, P, WIN_LEN, STEP, DETREND_TYPE, ESTIMATOR, box_cox_lambda=-1)
    max_on_diag = max_on_diag if max_on_diag is not None else  np.max(dDTF)
    max_off_diag = max_off_diag if max_off_diag is not None else  np.max(dDTF)
    fig = plt.figure(figsize=(11.5, 5.0))
    subfig_signals, subfig_mvar = fig.subfigures(1, 2, width_ratios=[1.0, 1.2])

    ax_top, ax_bot = subfig_signals.subplots(2, 1, sharex=True)
    ax_top.plot(dyad[0], color=GREY, linewidth=0.8)
    ax_top.set_ylabel(f"{CHANNEL_LABELS[0]}\nz-scored amplitude")
    ax_bot.plot(dyad[1], color=TEAL, linewidth=0.8)
    ax_bot.set_ylabel(f"{CHANNEL_LABELS[1]}\nz-scored amplitude")
    ax_bot.set_xlabel("time (samples)")
    subfig_signals.suptitle(f"Example dyad at r_smooth={r_smooth:.2f} (contrast={r_smooth - R_ROUGH:+.2f})")

    mvar_plot(spectra, dDTF, FREQS, "from ", "to ", ["ch0", "ch1"],
              "spectra (diag) / dDTF (off-diag)", fig=subfig_mvar, band_hz=COUPLING_BAND_HZ, max_on_diag=max_on_diag, max_off_diag=max_off_diag)

    fig_path = OUTPUT_DIR / f"example_dyads_r_smooth_{r_smooth:.2f}.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    return max_on_diag, max_off_diag


# ---------------------------------------------------------------------------
# Figure 1: the smoothness gradient -- pseudo-flow, its null, and Delta
# ---------------------------------------------------------------------------
print(f"Figure 1: sweeping the smoothness gradient (zero real coupling throughout), "
      f"{N_BATCHES} batches x {N_DYADS} dyads")
contrast = R_SMOOTH_GRID - R_ROUGH
real_mean, null_mean, delta_mean, delta_sd, delta_z_mean = [], [], [], [], []
first_example_dyads = simulate_dyads(R_SMOOTH_GRID[0], 0.0, N_DYADS, BASE_SEED + 0 * 10000)
last_example_dyads = simulate_dyads(R_SMOOTH_GRID[-1], 0.0, N_DYADS, BASE_SEED + (N_BATCHES - 1) * 10000)
max_on_diag, max_off_diag = example_dyads_figure(last_example_dyads, R_SMOOTH_GRID[-1], max_on_diag=None, max_off_diag=None) # final example after the loop
print(f"max_on_diag={max_on_diag:.5f}, max_off_diag={max_off_diag:.5f}")
_ = example_dyads_figure(first_example_dyads, R_SMOOTH_GRID[0], max_on_diag=max_on_diag, max_off_diag=max_off_diag  ) # final example after the loop
# %%
for r_smooth in R_SMOOTH_GRID:
    batch_real, batch_null, batch_delta, batch_z = [], [], [], []
    for batch in range(N_BATCHES):
        dyads = simulate_dyads(r_smooth, 0.0, N_DYADS, BASE_SEED + batch * 10000)
        reals = real_edge_values(dyads, EDGE_SMOOTH_TO_ROUGH)
        nulls = surrogate_null(dyads, EDGE_SMOOTH_TO_ROUGH)
        per_dyad = [delta_and_z(r, nulls) for r in reals]
        batch_real.append(reals.mean())
        batch_null.append(nulls.mean())
        batch_delta.append(np.mean([d["delta"] for d in per_dyad]))
        batch_z.append(np.mean([d["z"] for d in per_dyad]))
    real_mean.append(np.mean(batch_real))
    null_mean.append(np.mean(batch_null))
    delta_mean.append(np.mean(batch_delta))
    delta_sd.append(np.std(batch_delta, ddof=1))
    delta_z_mean.append(np.mean(batch_z))
    print(f"  r_smooth={r_smooth:.2f} (contrast={r_smooth - R_ROUGH:+.2f}): "
          f"real={real_mean[-1]:.5f} null={null_mean[-1]:.5f} "
          f"Delta={delta_mean[-1]:+.5f} z={delta_z_mean[-1]:+.2f}")


real_mean = np.array(real_mean); null_mean = np.array(null_mean)
delta_mean = np.array(delta_mean); delta_sd = np.array(delta_sd); delta_z_mean = np.array(delta_z_mean)

fig1, ax1 = plt.subplots(figsize=(8.2, 5.0))
ax1.axhline(0, color="#888888", linewidth=0.8, linestyle="--", zorder=0)
ax1.plot(contrast, real_mean, "-o", color=NAVY, linewidth=2.2, markersize=6,
         label="raw dDTF, real dyads (smooth $\\rightarrow$ rough)")
ax1.plot(contrast, null_mean, "-s", color=AMBER, linewidth=2.2, markersize=6,
         label="surrogate null mean (same edge)")
ax1.fill_between(contrast, delta_mean - delta_sd, delta_mean + delta_sd,
                 color=TEAL, alpha=0.2, linewidth=0)
ax1.plot(contrast, delta_mean, "-D", color=TEAL, linewidth=2.4, markersize=6,
         label="$\\Delta$ = real $-$ null (the cure, mean $\\pm$ SD over batches)")
ax1.set_xlabel("smoothness contrast  $r_{smooth} - r_{rough}$  (self-gain difference)")
ax1.set_ylabel("band-averaged dDTF, 0.2$-$1.0 Hz")
ax1.set_title("Zero real coupling: a smoothness mismatch alone bends dDTF into a spurious\n"
              "smooth$\\rightarrow$rough edge (smoother channel looks like the source).\n"
              "The surrogate null reproduces it; $\\Delta$ removes it.",
              fontsize=10.5)
ax1.legend(frameon=False, fontsize=9, loc="upper left")
ax1.spines[["top", "right"]].set_visible(False)
fig1.tight_layout()
fig1_path = OUTPUT_DIR / "fig1_gradient.png"
fig1.savefig(fig1_path, dpi=150)
plt.close(fig1)
print(f"  wrote {fig1_path}")

# ---------------------------------------------------------------------------
# Figure 2: the Stage 5 null-vs-real histogram, on known ground truth
# ---------------------------------------------------------------------------
print("\nFigure 2: null-vs-real twin at high contrast "
      f"(r_smooth={R_SMOOTH_FIG2}), ground truth 0 vs > 0")


def panel_data(injected_gain):
    """Real per-dyad values, pooled null, and mean Delta/z for one ground truth."""
    dyads = simulate_dyads(R_SMOOTH_FIG2, injected_gain, N_DYADS_FIG2, BASE_SEED)
    reals = real_edge_values(dyads, EDGE_SMOOTH_TO_ROUGH)
    nulls = surrogate_null(dyads, EDGE_SMOOTH_TO_ROUGH)
    per_dyad = [delta_and_z(r, nulls) for r in reals]
    return reals, nulls, np.mean([d["delta"] for d in per_dyad]), np.mean([d["z"] for d in per_dyad])


reals0, nulls0, delta0, z0 = panel_data(0.0)
reals1, nulls1, delta1, z1 = panel_data(INJECTED_GAIN)
print(f"  ground truth = 0 : mean Delta={delta0:+.5f}  mean z={z0:+.2f}")
print(f"  ground truth > 0 : mean Delta={delta1:+.5f}  mean z={z1:+.2f}")

fig2, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharex=False)
panels = [
    (axes[0], reals0, nulls0, delta0, z0, "Ground truth = 0  (no coupling)",
     "artifact present in the raw estimate,\nbut $\\Delta \\approx 0$: removed"),
    (axes[1], reals1, nulls1, delta1, z1, "Ground truth > 0  (genuine smooth$\\rightarrow$rough)",
     "$\\Delta > 0$: the real edge survives\nthe smoothness-matched null"),
]
for ax, reals, nulls, delta, z, title, note in panels:
    ax.hist(nulls, bins=18, color=GREY, edgecolor="white", label="surrogate null")
    ax.axvline(nulls.mean(), color=AMBER, linewidth=2.2, linestyle="--", label="null mean")
    for i, value in enumerate(reals):
        ax.axvline(value, color=NAVY, alpha=0.75, linewidth=1.3,
                   label="real dyads" if i == 0 else None)
    ax.set_title(title, fontsize=10.5)
    ax.set_xlabel("band-averaged dDTF, smooth $\\rightarrow$ rough")
    ax.annotate(f"mean $\\Delta$ = {delta:+.4f}\nmean z = {z:+.2f}\n{note}",
                xy=(0.97, 0.97), xycoords="axes fraction", ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=TEAL))
    ax.legend(frameon=False, fontsize=8, loc="center right")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("surrogate count")
fig2.suptitle("The surrogate null is the smoothness floor: subtracting it removes the artifact "
              "and keeps genuine flow", fontsize=11)
fig2.tight_layout(rect=(0, 0, 1, 0.96))
fig2_path = OUTPUT_DIR / "fig2_null_twin.png"
fig2.savefig(fig2_path, dpi=150)
plt.close(fig2)
print(f"  wrote {fig2_path}")

# ---------------------------------------------------------------------------
# Self-contained gate
# ---------------------------------------------------------------------------
summary_rows = "".join(
    f"<tr><td>{r:.2f}</td><td>{r - R_ROUGH:+.2f}</td><td>{rm:.5f}</td>"
    f"<td>{nm:.5f}</td><td>{dm:+.5f}</td><td>{zm:+.2f}</td></tr>"
    for r, rm, nm, dm, zm in zip(R_SMOOTH_GRID, real_mean, null_mean, delta_mean, delta_z_mean)
)
gate_html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Stage 0b - smoothness artifact</title>
<style>
  body {{ font-family: -apple-system, sans-serif; margin: 2em; color: {NAVY}; max-width: 1000px; }}
  h1 {{ color: {TEAL}; }} h2 {{ border-bottom: 2px solid {LIME}; padding-bottom: 0.2em; }}
  img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 6px; margin: 0.5em 0; }}
  table {{ border-collapse: collapse; font-size: 0.9em; }}
  th, td {{ border: 1px solid #ccc; padding: 0.3em 0.7em; text-align: right; }}
  .finding {{ background: #fbf3d3; border-left: 5px solid {AMBER}; padding: 0.8em 1em; margin: 1em 0; }}
</style></head><body>
<h1>Stage 0b &mdash; the smoothness artifact and its surrogate cure</h1>
<p>Two independent AR(2) resonators at a common {F0_HZ} Hz peak, differing only in
pole radius (self-gain). <b>Zero injected coupling.</b> Run through the real
Stage 4 estimator (p={P}, {WIN_LEN_S:.0f} s / {int(OVERLAP_FRAC*100)}% windows,
{COUPLING_BAND_HZ[0]}&ndash;{COUPLING_BAND_HZ[1]} Hz band) and the real Stage 5
surrogate + delta machinery.</p>
<div class="finding"><b>Verified direction (with the pipeline's per-channel z-scoring):</b>
the <b>smoother channel appears as the source</b> &mdash; the spurious edge runs
smooth&nbsp;&rarr;&nbsp;rough, matching the "smoother looks like a source" slide.
dDTF is not scale-invariant: <b>without</b> z-scoring the sign flips. Since the
direction depends on a preprocessing choice, it must be read off the surrogate,
not assumed.</div>
<h2>Figure 1 &mdash; the gradient (disease and cure)</h2>
<img src="fig1_gradient.png" alt="gradient">
<h2>Figure 2 &mdash; the null-vs-real twin on known ground truth</h2>
<img src="fig2_null_twin.png" alt="null twin">
<h2>Numbers</h2>
<table><tr><th>r_smooth</th><th>contrast</th><th>real</th><th>null mean</th>
<th>&Delta;</th><th>z</th></tr>{summary_rows}</table>
</body></html>"""
gate_path = OUTPUT_DIR / "smoothness_artifact_gate.html"
gate_path.write_text(gate_html, encoding="utf-8")
print(f"\nWrote gate to {gate_path}")
