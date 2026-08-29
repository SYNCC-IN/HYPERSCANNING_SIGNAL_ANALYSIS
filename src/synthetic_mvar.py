"""Synthetic ground-truth generators for validating MVAR / ffDTF connectivity.

These generators produce multivariate signals with a known, directed, lagged
coupling structure, so `src.mtmvar.full_freq_dtf` (and friends) can be checked
against a known answer before being trusted on real data. See
`scripts/stage00_synthetic_validation.py` for the validation harness built on
top of these functions.
"""

from math import gcd

import numpy as np
from scipy.signal import resample_poly

try:
    from .envelopes import filter_individual_band, hilbert_envelope
except ImportError:  # pragma: no cover - fallback for direct script execution
    from src.envelopes import filter_individual_band, hilbert_envelope


def edges_to_coupling(edges, n_nodes, max_lag=None):
    """Build a directed AR coefficient tensor from a readable edge list.

    Parameters
    ----------
    edges : list of tuple
        Each tuple is ``(source, target, lag, gain)``: node `source` drives
        node `target` at a delay of `lag` samples with linear coefficient
        `gain`. Use ``source == target`` to add self-persistence (a node's
        own past driving its own present).
    n_nodes : int
        Number of variables `k`.
    max_lag : int or None, optional
        Number of lags in the returned tensor. If None, inferred as the
        largest `lag` present in `edges`.

    Returns
    -------
    np.ndarray, shape (n_nodes, n_nodes, max_lag)
        AR coefficient tensor with the convention
        ``coupling[target, source, lag - 1] = gain`` (row = target/driven,
        column = source/driving), matching
        ``x[:, t] = sum_lag coupling[:, :, lag - 1] @ x[:, t - lag] + noise[:, t]``
        as simulated by `generate_var_process`.
    """
    if max_lag is None:
        max_lag = max(lag for _, _, lag, _ in edges)
    coupling = np.zeros((n_nodes, n_nodes, max_lag))
    for source, target, lag, gain in edges:
        coupling[target, source, lag - 1] = gain
    return coupling


def summarize_coupling_strength(coupling):
    """Collapse a lagged AR coefficient tensor into one strength matrix.

    Parameters
    ----------
    coupling : np.ndarray, shape (k, k, max_lag)
        AR coefficient tensor, see `edges_to_coupling`.

    Returns
    -------
    np.ndarray, shape (k, k)
        Sum of absolute per-lag gains for each (target, source) pair —
        a lag-agnostic "how strongly does source drive target" summary,
        directly comparable to a frequency-averaged ffDTF matrix.
    """
    return np.sum(np.abs(coupling), axis=2)


def generate_var_process(coupling, snr, n_samples, burn_in=200, seed=None):
    """Simulate a k-variable VAR process from a directed AR coefficient tensor.

    Implements
    ``x[:, t] = sum_{lag=1}^{p} coupling[:, :, lag - 1] @ x[:, t - lag] + noise[:, t]``,
    where ``coupling[target, source, lag - 1]`` is the linear gain by which
    node `source`'s value `lag` samples ago drives node `target`'s current
    value (row = target/driven, column = source/driving — build `coupling`
    with `edges_to_coupling` rather than by hand). Each channel's own
    innovation noise is independent Gaussian with standard deviation
    ``1 / snr``; the `coupling` gains should be scaled comparably (gains
    around 0.3-0.6 with `snr` in the 2-10 range give a clearly recoverable
    but not trivially noiseless ground truth).

    This function does not check `coupling` for stability before simulating;
    an unstable spec (gains/eigenvalues too large) will surface as
    non-finite output, caught by the assertion below rather than silently
    propagated.

    Parameters
    ----------
    coupling : np.ndarray, shape (k, k, max_lag)
        Directed AR coefficient tensor, see `edges_to_coupling`.
    snr : float
        Inverse innovation-noise scale: noise standard deviation is
        ``1 / snr``.
    n_samples : int
        Number of samples to return (after discarding `burn_in`).
    burn_in : int, optional
        Number of initial samples simulated and discarded, to let the
        process settle past its zero initial condition.
    seed : int or None, optional
        Seed for the random number generator (deterministic given a seed).

    Returns
    -------
    np.ndarray, shape (k, n_samples)
        Simulated multivariate process, variable order matching `coupling`'s
        first two axes.
    """
    k, k2, max_lag = coupling.shape
    assert k == k2, "coupling must be square in its first two axes (k, k, max_lag)"

    rng = np.random.default_rng(seed)
    noise_std = 1.0 / snr
    n_total = n_samples + burn_in

    x = np.zeros((k, n_total + max_lag))
    noise = rng.normal(0.0, noise_std, size=(k, n_total))
    for t in range(max_lag, n_total + max_lag):
        driven = np.zeros(k)
        for lag in range(1, max_lag + 1):
            driven += coupling[:, :, lag - 1] @ x[:, t - lag]
        x[:, t] = driven + noise[:, t - max_lag]

    x = x[:, max_lag + burn_in:]
    assert np.all(np.isfinite(x)), "simulated VAR process diverged - coupling spec is unstable"
    return x


def generate_coupled_oscillators(
    center_freqs, coupling, snr, fs, n_samples, bandwidth,
    envelope_fs=2.0, carrier_order=4, seed=None,
):
    """Generate k narrow-band oscillatory signals with coupled amplitude envelopes.

    Builds each node's signal as ``envelope_i(t) * unit_carrier_i(t)``:

    1. A slow, positive envelope per node comes from a `generate_var_process`
       log-amplitude VAR simulated at `envelope_fs` (`coupling` acts on
       log-amplitude; exponentiating keeps the envelope positive), upsampled
       to `fs` by polyphase resampling.
    2. Each node's `unit_carrier` is an independent, unit-amplitude
       narrow-band oscillation: white noise band-pass filtered at
       ``center_freqs[node] +/- bandwidth`` (`filter_individual_band`), then
       divided by its own Hilbert envelope (`hilbert_envelope`) so its
       instantaneous amplitude is exactly 1 and only its instantaneous
       phase/frequency (within the band) is random.
    3. Independent Gaussian observation noise is added to each node's
       finished carrier, with standard deviation ``std(signal) / snr`` — a
       second, distinct use of `snr` from the one inside
       `generate_var_process` (see Notes).

    Parameters
    ----------
    center_freqs : sequence of float, length k
        Carrier center frequency per node (Hz), e.g. child/caregiver rhythm
        center frequencies.
    coupling : np.ndarray, shape (k, k, max_lag)
        Directed AR coefficient tensor acting on log-amplitude, see
        `generate_var_process` / `edges_to_coupling`.
    snr : float
        Passed to `generate_var_process` for the envelope-coupling
        innovation noise, and separately used to scale final additive
        observation noise (``noise_std = std(signal) / snr``).
    fs : float
        Output sampling frequency (Hz) of the returned carrier signals.
    n_samples : int
        Number of samples to return, at `fs`.
    bandwidth : float
        Carrier half-bandwidth (Hz), passed to `filter_individual_band`.
    envelope_fs : float, optional
        Sampling rate (Hz) at which the coupled log-amplitude VAR is
        simulated, before upsampling to `fs`. Should be well below the rate
        at which the envelope coupling is meant to vary (default 2.0 Hz,
        matching the real pipeline's target envelope rate).
    carrier_order : int, optional
        Filter order passed to `filter_individual_band` for the carrier.
    seed : int or None, optional
        Seed for the random number generator (deterministic given a seed).

    Returns
    -------
    np.ndarray, shape (k, n_samples)
        Simulated narrow-band signals with coupled amplitude envelopes.

    Notes
    -----
    `snr` is reused for two conceptually different noise sources: the VAR
    innovation noise driving the envelope coupling (inside
    `generate_var_process`, ``noise_std = 1 / snr``), and the final additive
    observation noise on the finished carrier (``noise_std = std(signal) /
    snr``). Both make the ground-truth coupling structure harder to recover
    as `snr` decreases; call `generate_var_process` directly if independent
    control over the two is needed.
    """
    k = len(center_freqs)
    rng = np.random.default_rng(seed)

    # Small margin so the polyphase-resampled envelope is never shorter than n_samples.
    n_env_samples = int(round(n_samples * envelope_fs / fs)) + 2
    log_amplitude = generate_var_process(coupling, snr, n_env_samples, seed=seed)
    envelope_slow = np.exp(log_amplitude)

    up, down = int(round(fs)), int(round(envelope_fs))
    common = gcd(up, down)
    up, down = up // common, down // common
    envelope = np.stack([
        resample_poly(envelope_slow[node], up, down) for node in range(k)
    ])[:, :n_samples]

    signals = np.zeros((k, n_samples))
    for node in range(k):
        white_noise = rng.standard_normal(n_samples)
        narrowband = filter_individual_band(white_noise, fs, center_freqs[node], bandwidth, carrier_order)
        unit_carrier = narrowband / hilbert_envelope(narrowband)
        signals[node] = envelope[node] * unit_carrier

    for node in range(k):
        obs_noise_std = np.std(signals[node]) / snr
        signals[node] += rng.normal(0.0, obs_noise_std, size=n_samples)

    return signals
