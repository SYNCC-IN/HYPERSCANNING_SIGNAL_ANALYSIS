"""Stage 0 - Synthetic ground-truth validation of the dDTF connectivity estimator.

Interbrain ffDTF + HRV pipeline, Stage 0 (see DTF_analysis_notes/pipeline_plan.md).
Uses no real data: everything is generated with a known coupling structure, so
`src.mtmvar.direct_dtf` can be trusted before it ever touches real EEG/HRV
envelopes.

Part A: inject a known, asymmetric 4-variable coupling (fixed variable order
[child:ROI, cg:ROI, child:HRV, cg:HRV] = [0, 1, 2, 3]) and confirm both that
`direct_dtf` recovers it and which array index carries source vs. target.

Part B: sweep the child/caregiver rhythm centre-frequency (CF) gap across
6-14 Hz and compare envelope-based vs. phase-based (raw narrow-band) dDTF,
to reproduce (on synthetic data) the claim that envelope-based connectivity is
robust to a CF mismatch between partners while phase-based connectivity is not.

"""

import contextlib
import io
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.envelopes import (
    downsample,
    filter_individual_band,
    hilbert_envelope,
    plot_signal_filtered_envelope,
)
from src.io_utils import ensure_dir
from src.mtmvar import direct_dtf, graph_plot, multivariate_spectra, mvar_plot
from src.synthetic_mvar import (
    edges_to_coupling,
    generate_coupled_oscillators,
    generate_var_process,
    summarize_coupling_strength,
)

plt.style.use('seaborn-v0_8-whitegrid')

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 44

# Later pipeline stages (1-6) will write their own numbered subfolders under
# this same root - see DTF_analysis_notes/pipeline_plan.md section 4.5.
ANALYSIS_ROOT = PROJECT_ROOT / 'Interbrain_ffDTF_analysis'
OUTPUT_DIR = ensure_dir(ANALYSIS_ROOT / '00_synthetic_validation')

# Fixed variable order, used everywhere downstream in this pipeline.
CHAN_NAMES = ['child:ROI', 'cg:ROI', 'child:HRV', 'cg:HRV']

# --- Part A: 4-node injected coupling (clean VAR at the analysis/envelope
# sampling rate - slow, no carrier) -----------------------------------------
PART_A_FS = 2.0             # Hz, matches the real pipeline's ~2 Hz envelope rate
PART_A_N_SAMPLES = 4000
PART_A_SNR = 5.0
PART_A_MAX_MODEL_ORDER = 10
CRIT_TYPE = 'AIC'
PART_A_FREQ_STEP = 0.01     # Hz

SELF_AR_GAIN = 0.3          # mild self-persistence on every node, for spectral shape
EDGE_GAIN_H2 = 0.5          # cg:ROI(1) -> child:ROI(0)
EDGE_GAIN_H4 = 0.5          # cg:HRV(3) -> child:HRV(2)
EDGE_GAIN_NOVEL = 0.35      # cg:HRV(3) -> child:ROI(0), novel exploratory edge, lag 2
PART_A_EDGES = [
    (0, 0, 1, SELF_AR_GAIN), # (source, target, lag, gain)
    (1, 1, 1, SELF_AR_GAIN),
    (2, 2, 1, SELF_AR_GAIN),
    (3, 3, 1, SELF_AR_GAIN),
    (1, 0, 1, EDGE_GAIN_H2),
    (3, 2, 2, EDGE_GAIN_H4),
    (3, 0, 3, EDGE_GAIN_NOVEL),
]

# --- Orientation check: single unidirectional edge 0 -> 1, everything else zero ---
ORIENTATION_GAIN = 0.6
ORIENTATION_SNR = 10.0
ORIENTATION_N_SAMPLES = 3000
ORIENTATION_FS = 2.0
ORIENTATION_MAX_MODEL_ORDER = 5
ORIENTATION_EDGES = [(0, 1, 1, ORIENTATION_GAIN)]
ORIENTATION_DOMINANCE_RATIO = 3.0  # dominant direction must exceed the reverse by this factor

# --- Part B: envelope vs. phase across the child/caregiver CF gap ---------
N_REALIZATIONS = 100          # independent draws per CF gap, for mean +/- 95% CI
CHILD_CF_HZ = 8.0             # fixed child fast-rhythm centre frequency
CF_GAPS_HZ = np.arange(0.0, 10.0, 0.5)  # Hz, caregiver CF = CHILD_CF_HZ + gap
BANDWIDTH_HZ = 1.5            # half-width, see src.envelopes.filter_individual_band
CARRIER_ORDER = 4             # Butterworth filter order used for band-limiting signals to a narrow rhythm band in Part B 
FS_CARRIER = 128.0            # Hz, matches typical decimated real EEG sampling rate
DURATION_S = 180.0            # 3 min, mimics a concatenated-movies-scale recording
PART_B_N_SAMPLES = int(DURATION_S * FS_CARRIER)
ENVELOPE_TARGET_SFREQ = 2.0    # Hz
PART_B_SNR = 4.0
PART_B_EDGE_GAIN = 0.5        # cg-proxy(1) -> child-proxy(0), analogous to H2
PART_B_SELF_AR_GAIN = 0.3
PART_B_MAX_MODEL_ORDER_ENV = 10
PART_B_MAX_MODEL_ORDER_RAW = 15
FREQ_STEP_ENV = 0.01           # Hz
FREQ_STEP_RAW = 0.1            # Hz
PART_B_EDGES = [
    (0, 0, 1, PART_B_SELF_AR_GAIN),
    (1, 1, 1, PART_B_SELF_AR_GAIN),
    (1, 0, 1, PART_B_EDGE_GAIN),
]

# Pass-criteria thresholds
ENVELOPE_PASS_THRESHOLD_FRACTION = 0.5   # must retain >= 50% of the no-CF-gap-confound reference
PHASE_DEGRADATION_THRESHOLD_FRACTION = 0.5  # must drop by >= 50% from smallest to largest gap

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def get_edge_strength(d_dtf, source, target):
    """Read the frequency-averaged dDTF value for one source->target edge.

    Uses the orientation confirmed above: if `orientation_is_target_source`,
    `d_dtf` is indexed ``[target, source, freq]``; otherwise
    ``[source, target, freq]``.
    """
    if orientation_is_target_source:
        return d_dtf[target, source, :].mean()
    return d_dtf[source, target, :].mean()


def mean_and_sem(values, axis=0):
    """Mean and standard error of the mean (SEM) across realizations.

    Uses the standard error of the mean (SEM), 
    which is adequate for `N_REALIZATIONS` in the tens-to-hundreds range used here.

    Parameters
    ----------
    values : np.ndarray
        Array of realizations to summarize.
    axis : int, optional
        Axis indexing independent realizations (default 0).

    Returns
    -------
    mean : np.ndarray
        Mean across `axis`.
    sem : np.ndarray
        Standard error of the mean across `axis`; the mean +/- sem gives an approximate 68% CI.
    """
    mean = values.mean(axis=axis)
    sem = values.std(axis=axis, ddof=1) / np.sqrt(values.shape[axis])
    return mean, sem


# ---------------------------------------------------------------------------
# Sanity checks on the generators: fixed seed -> fixed output, correct shape
# ---------------------------------------------------------------------------
_sanity_coupling = edges_to_coupling(PART_A_EDGES, n_nodes=4)
_sanity_a1 = generate_var_process(_sanity_coupling, PART_A_SNR, 50, seed=SEED)
_sanity_a2 = generate_var_process(_sanity_coupling, PART_A_SNR, 50, seed=SEED)
assert _sanity_a1.shape == (4, 50)
assert np.array_equal(_sanity_a1, _sanity_a2)

_sanity_osc_coupling = edges_to_coupling(PART_B_EDGES, n_nodes=2)
_sanity_b1 = generate_coupled_oscillators(
    [CHILD_CF_HZ, CHILD_CF_HZ + 8.0], _sanity_osc_coupling, PART_B_SNR,
    FS_CARRIER, 512, BANDWIDTH_HZ, envelope_fs=ENVELOPE_TARGET_SFREQ, seed=SEED,
)
_sanity_b2 = generate_coupled_oscillators(
    [CHILD_CF_HZ, CHILD_CF_HZ + 8.0], _sanity_osc_coupling, PART_B_SNR,
    FS_CARRIER, 512, BANDWIDTH_HZ, envelope_fs=ENVELOPE_TARGET_SFREQ, seed=SEED,
)
assert _sanity_b1.shape == (2, 512)
assert np.array_equal(_sanity_b1, _sanity_b2)
print("Sanity check passed: both generators are deterministic given a fixed seed.")

# Artifact: generated oscillators with their Hilbert envelopes overlaid, one
# figure per node, via the same filter_individual_band -> hilbert_envelope
# path used for real analysis (reusing src.envelopes.plot_signal_filtered_envelope).
_sanity_center_freqs = [CHILD_CF_HZ, CHILD_CF_HZ + 8.0]
_sanity_labels = ['node 0 (child-proxy)', 'node 1 (cg-proxy)']
for node, label in enumerate(_sanity_labels):
    filtered = filter_individual_band(
        _sanity_b2[node], FS_CARRIER, _sanity_center_freqs[node], BANDWIDTH_HZ, CARRIER_ORDER,
    )
    envelope = hilbert_envelope(filtered)
    sanity_fig = plot_signal_filtered_envelope(
        _sanity_b2[node], filtered, envelope, FS_CARRIER,
        f"Sanity check: generate_coupled_oscillators() output - {label}",
    )
    sanity_fig.savefig(OUTPUT_DIR / f'sanity_generated_oscillator_node{node}.png', dpi=150)
    plt.close(sanity_fig)
print(f"Saved sanity-check envelope plots to {OUTPUT_DIR}")

# ---------------------------------------------------------------------------
# Part A, step 3 - empirically confirm direct_dtf's source/target orientation
# ---------------------------------------------------------------------------
orientation_coupling = edges_to_coupling(ORIENTATION_EDGES, n_nodes=2)
orientation_signals = generate_var_process(
    orientation_coupling, ORIENTATION_SNR, ORIENTATION_N_SAMPLES, seed=SEED,
)
orientation_freqs = np.arange(PART_A_FREQ_STEP, ORIENTATION_FS / 2, PART_A_FREQ_STEP)
ddtf_orientation = direct_dtf(
    orientation_signals, orientation_freqs, ORIENTATION_FS,
    max_model_order=ORIENTATION_MAX_MODEL_ORDER, crit_type=CRIT_TYPE,
)

mean_01 = ddtf_orientation[0, 1, :].mean()  # d_dtf[row=0, col=1]
mean_10 = ddtf_orientation[1, 0, :].mean()  # d_dtf[row=1, col=0]

orientation_is_target_source = mean_10 > mean_01
if orientation_is_target_source:
    dominant_value, other_value = mean_10, mean_01
    orientation_label = "direct_dtf output is indexed [target, source, freq] (row = target/driven, column = source/driving)"
else:
    dominant_value, other_value = mean_01, mean_10
    orientation_label = "direct_dtf output is indexed [source, target, freq] (row = source/driving, column = target/driven)"

print("\n=== Orientation check (injected edge: node 0 -> node 1 only) ===")
print(f"  mean dDTF[0, 1, :] = {mean_01:.4f}")
print(f"  mean dDTF[1, 0, :] = {mean_10:.4f}")
print(f"  CONFIRMED: {orientation_label}")

assert dominant_value > ORIENTATION_DOMINANCE_RATIO * other_value, (
    "direct_dtf did not clearly recover the injected 0 -> 1 edge "
    "(orientation check failed, or SNR/model order needs adjustment)."
)



# ---------------------------------------------------------------------------
# Part A, steps 1-2 - inject the full 4-node coupling and run the real estimator
# ---------------------------------------------------------------------------
part_a_coupling = edges_to_coupling(PART_A_EDGES, n_nodes=4)
part_a_signals = generate_var_process(part_a_coupling, PART_A_SNR, PART_A_N_SAMPLES, seed=SEED)
part_a_freqs = np.arange(0, PART_A_FS / 2, PART_A_FREQ_STEP)

ddtf_part_a = direct_dtf(
    part_a_signals, part_a_freqs, PART_A_FS,
    max_model_order=PART_A_MAX_MODEL_ORDER, crit_type=CRIT_TYPE,
)
spectra_part_a = multivariate_spectra(
    part_a_signals, part_a_freqs, PART_A_FS,
    max_model_order=PART_A_MAX_MODEL_ORDER, crit_type=CRIT_TYPE,
)

part_a_h2_pass = get_edge_strength(ddtf_part_a, 1, 0) > get_edge_strength(ddtf_part_a, 0, 1)
part_a_h4_pass = get_edge_strength(ddtf_part_a, 3, 2) > get_edge_strength(ddtf_part_a, 2, 3)
part_a_novel_pass = get_edge_strength(ddtf_part_a, 3, 0) > get_edge_strength(ddtf_part_a, 0, 3)

print("\n=== Part A: 4-node recovery ===")
print(f"  cg:ROI -> child:ROI  (1->0, H2)    : {get_edge_strength(ddtf_part_a, 1, 0):.4f}  vs reverse {get_edge_strength(ddtf_part_a, 0, 1):.4f}  [{'PASS' if part_a_h2_pass else 'FAIL'}]")
print(f"  cg:HRV -> child:HRV  (3->2, H4)    : {get_edge_strength(ddtf_part_a, 3, 2):.4f}  vs reverse {get_edge_strength(ddtf_part_a, 2, 3):.4f}  [{'PASS' if part_a_h4_pass else 'FAIL'}]")
print(f"  cg:HRV -> child:ROI  (3->0, novel) : {get_edge_strength(ddtf_part_a, 3, 0):.4f}  vs reverse {get_edge_strength(ddtf_part_a, 0, 3):.4f}  [{'PASS' if part_a_novel_pass else 'FAIL'}]")

# Gate figure (a): injected directionality matrix vs recovered dDTF matrix
injected_matrix = summarize_coupling_strength(part_a_coupling)  # (target, source), by construction
recovered_matrix = ddtf_part_a.mean(axis=2)
# We are interested only in the off-diagonal edges, so zero out the diagonals for a fair visual comparison.
np.fill_diagonal(injected_matrix, 0.0)
np.fill_diagonal(recovered_matrix, 0.0)
if not orientation_is_target_source:
    recovered_matrix = recovered_matrix.T  # reorient to (target, source) to match injected_matrix

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
for ax, matrix, title in (
    (axes[0], injected_matrix, "Injected coupling strength\n(ground truth, |gain| summed over lags)"),
    (axes[1], recovered_matrix, "Recovered mean dDTF\n(reoriented to target, source)"),
):
    im = ax.imshow(matrix, cmap='viridis')
    ax.grid(False)
    ax.set_xticks(range(4)); ax.set_xticklabels(CHAN_NAMES, rotation=45, ha='right')
    ax.set_yticks(range(4)); ax.set_yticklabels(CHAN_NAMES)
    ax.set_xlabel('source'); ax.set_ylabel('target')
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
# fig.suptitle('Stage 0, Part A - gate figure (a): injected vs. recovered directionality')
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'gate_a_injected_vs_recovered.png', dpi=150)
plt.close(fig)
print(f"Saved gate figure (a) to {OUTPUT_DIR / 'gate_a_injected_vs_recovered.png'}")

# Gate figure (b): mvar_plot grid on the synthetic 4-node data.
# mvar_plot() creates its own figure internally (default 8x8 figsize) - grab it
# via plt.gcf() afterward rather than pre-creating one.
x_label, y_label = ('source: ', 'target: ') if orientation_is_target_source else ('target: ', 'source: ')
mvar_plot(
    spectra_part_a, ddtf_part_a, part_a_freqs,
    x_label=x_label, y_label=y_label, chan_names=CHAN_NAMES,
    top_title='Stage 0 Part A: synthetic 4-node MVAR grid\n(diagonal = spectra, off-diagonal = dDTF)',
    fig_size=(8, 8),
)
fig_b = plt.gcf()
fig_b.savefig(OUTPUT_DIR / 'gate_b_mvar_plot_grid.png', dpi=150, bbox_inches='tight')
plt.close(fig_b)
print(f"Saved gate figure (b) to {OUTPUT_DIR / 'gate_b_mvar_plot_grid.png'}")

# Bonus (optional, not part of the pass criteria): directed-graph view
fig, ax = plt.subplots(figsize=(6, 6))
graph_plot(
    ddtf_part_a, ax, part_a_freqs, (part_a_freqs[0], part_a_freqs[-1]), CHAN_NAMES,
    'Stage 0, Part A - directed connectivity graph (full band)',
)
fig.savefig(OUTPUT_DIR / 'gate_bonus_graph_view.png', dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Part B - envelope vs. phase dDTF across the child/caregiver CF gap
# ---------------------------------------------------------------------------
part_b_coupling = edges_to_coupling(PART_B_EDGES, n_nodes=2)
freqs_env = np.arange(FREQ_STEP_ENV, ENVELOPE_TARGET_SFREQ / 2, FREQ_STEP_ENV)

print(f"\n=== Part B: {N_REALIZATIONS} realizations x {len(CF_GAPS_HZ)} CF gaps ===")

envelope_strengths = np.zeros((N_REALIZATIONS, len(CF_GAPS_HZ)))
phase_strengths = np.zeros((N_REALIZATIONS, len(CF_GAPS_HZ)))

for gap_idx, gap in enumerate(CF_GAPS_HZ):
    cg_cf = CHILD_CF_HZ + gap
    for realization in range(N_REALIZATIONS):
        signals = generate_coupled_oscillators(
            [CHILD_CF_HZ, cg_cf], part_b_coupling, PART_B_SNR, FS_CARRIER, PART_B_N_SAMPLES,
            BANDWIDTH_HZ, envelope_fs=ENVELOPE_TARGET_SFREQ, carrier_order=CARRIER_ORDER,
            seed=SEED + realization,
        )

        # Suppress direct_dtf's per-call "optimal model order" prints here -
        # with N_REALIZATIONS x len(CF_GAPS_HZ) calls they would otherwise
        # flood stdout; the underlying computation is unaffected.
        with contextlib.redirect_stdout(io.StringIO()):
            # Envelope path: filter -> Hilbert envelope -> downsample -> dDTF
            env0 = hilbert_envelope(filter_individual_band(signals[0], FS_CARRIER, CHILD_CF_HZ, BANDWIDTH_HZ, CARRIER_ORDER))
            env1 = hilbert_envelope(filter_individual_band(signals[1], FS_CARRIER, cg_cf, BANDWIDTH_HZ, CARRIER_ORDER))
            env0_ds, env_fs = downsample(env0, FS_CARRIER, ENVELOPE_TARGET_SFREQ)
            env1_ds, _ = downsample(env1, FS_CARRIER, ENVELOPE_TARGET_SFREQ)
            # envelopes shold be zero-mean and unit-variance, but just in case, normalise them here to avoid any scale issues with dDTF.
            env0_ds = (env0_ds - env0_ds.mean()) / env0_ds.std()
            env1_ds = (env1_ds - env1_ds.mean()) / env1_ds.std()
            n_env = min(len(env0_ds), len(env1_ds))
            env_signals = np.stack([env0_ds[:n_env], env1_ds[:n_env]])
            ddtf_env = direct_dtf(
                env_signals, freqs_env, env_fs,
                max_model_order=PART_B_MAX_MODEL_ORDER_ENV, crit_type=CRIT_TYPE,
            )

            # Phase/raw path: dDTF directly on the narrow-band filtered signals
            raw0 = filter_individual_band(signals[0], FS_CARRIER, CHILD_CF_HZ, BANDWIDTH_HZ, CARRIER_ORDER)
            raw1 = filter_individual_band(signals[1], FS_CARRIER, cg_cf, BANDWIDTH_HZ, CARRIER_ORDER)
            raw0 = (raw0 - raw0.mean()) / raw0.std() # normalise to avoid scale issues with dDTF
            raw1 = (raw1 - raw1.mean()) / raw1.std() #
            raw_signals = np.stack([raw0, raw1])
            freq_lo = min(CHILD_CF_HZ, cg_cf) - BANDWIDTH_HZ - 1.0
            freq_hi = max(CHILD_CF_HZ, cg_cf) + BANDWIDTH_HZ + 1.0
            freqs_raw = np.arange(freq_lo, freq_hi, FREQ_STEP_RAW)
            
            ddtf_raw = direct_dtf(
                raw_signals, freqs_raw, FS_CARRIER,
                max_model_order=PART_B_MAX_MODEL_ORDER_RAW, crit_type=CRIT_TYPE,
            )

        envelope_strengths[realization, gap_idx] = get_edge_strength(ddtf_env, 1, 0)
        phase_strengths[realization, gap_idx] = get_edge_strength(ddtf_raw, 1, 0)

    print(f"  gap={gap:.1f} Hz (cg CF={cg_cf:.1f} Hz) done: "
          f"envelope mean={envelope_strengths[:, gap_idx].mean():.4f}  "
          f"phase mean={phase_strengths[:, gap_idx].mean():.4f}")

# Normalise each realization to its own gap=0 value (CF_GAPS_HZ[0] == 0.0), so
# curves are comparable on the same relative scale and every realization
# starts at exactly 1.0 - only the CF-gap-induced change is left.
relative_envelope_strengths = envelope_strengths / envelope_strengths[:, [0]]
relative_phase_strengths = phase_strengths / phase_strengths[:, [0]]

env_mean, env_sem = mean_and_sem(relative_envelope_strengths)
phase_mean, phase_sem = mean_and_sem(relative_phase_strengths)

# Envelope: mean stays close to 1.0 (its own gap=0 baseline) across the sweep,
# i.e. robust to CF gap.
part_b_envelope_pass = bool(env_mean.min() >= ENVELOPE_PASS_THRESHOLD_FRACTION)

# Phase: mean drops substantially below 1.0 at some point in the sweep.
# Compared against the sweep minimum rather than the last point, since the
# bias is expected to plateau at a noise floor once the two narrow bands stop
# overlapping at all, not necessarily keep decreasing to the largest gap.
part_b_phase_pass = bool((1.0 - phase_mean.min()) >= PHASE_DEGRADATION_THRESHOLD_FRACTION)

# Gate figure (c): mean +/- SEM (across realizations) of the recovered
# directed coupling vs. CF gap, envelope vs. phase, both relative to gap=0.
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(CF_GAPS_HZ, env_mean, marker='o', color='C0', label='Envelope-based dDTF (1 -> 0)')
ax.fill_between(CF_GAPS_HZ, env_mean - env_sem, env_mean + env_sem, color='C0', alpha=0.25)
ax.plot(CF_GAPS_HZ, phase_mean, marker='s', color='C1', label='Phase/raw-based dDTF (1 -> 0)')
ax.fill_between(CF_GAPS_HZ, phase_mean - phase_sem, phase_mean + phase_sem, color='C1', alpha=0.25)
ax.axhline(1.0, color='gray', linestyle='--', lw=0.8, label='Gap = 0 baseline')
ax.axvline(BANDWIDTH_HZ, color='gray', linestyle=':', lw=1.5, label=f'Bandwidth = {BANDWIDTH_HZ} Hz')
ax.set_xlabel('Child/caregiver centre-frequency gap (Hz)')
ax.set_ylabel('Relative dDTF, injected edge (1 -> 0)\n(normalised to each realization\'s gap=0 value)')
ax.set_title(f'Envelope vs. phase coupling estimation across the CF gap\n(mean +/- SEM, N={N_REALIZATIONS} realizations)')
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'gate_c_envelope_vs_phase.png', dpi=150)
plt.close(fig)
print(f"Saved gate figure (c) to {OUTPUT_DIR / 'gate_c_envelope_vs_phase.png'}")

# ---------------------------------------------------------------------------
# PASS / FAIL summary
# ---------------------------------------------------------------------------
checks = {
    "Orientation of direct_dtf's output confirmed empirically (assert above)": True,
    "Part A: H2 edge cg:ROI->child:ROI (1->0) recovered over reverse": part_a_h2_pass,
    "Part A: H4 edge cg:HRV->child:HRV (3->2) recovered over reverse": part_a_h4_pass,
    "Part A: novel edge cg:HRV->child:ROI (3->0) recovered over reverse": part_a_novel_pass,
    f"Part B: mean envelope path stays within {ENVELOPE_PASS_THRESHOLD_FRACTION:.0%} of its gap=0 baseline across the {CF_GAPS_HZ.min():.0f}-{CF_GAPS_HZ.max():.0f} Hz CF-gap sweep (N={N_REALIZATIONS})": part_b_envelope_pass,
    f"Part B: mean phase path drops by >= {PHASE_DEGRADATION_THRESHOLD_FRACTION:.0%} from its gap=0 baseline somewhere in the sweep (N={N_REALIZATIONS})": part_b_phase_pass,
}

print("\n=== Stage 0 gate summary ===")
for name, ok in checks.items():
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

overall_pass = all(checks.values())
print(f"\nOVERALL: {'PASS' if overall_pass else 'FAIL'}")
print(f"Confirmed orientation for Stage 4 to reuse: {orientation_label}")
