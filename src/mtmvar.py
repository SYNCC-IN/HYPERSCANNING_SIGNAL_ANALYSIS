# -*- coding: utf-8 -*-
"""
MVAR Analysis Functions for EEG Connectivity Analysis

This module contains functions for Multivariate Autoregressive (MVAR) modeling
and Directed Transfer Function (DTF) analysis of EEG data.

ACKNOWLEDGMENTS:
Some functions in this module are based on algorithms and code originally 
developed by Prof. Maciej Kamiński from the Department of Biomedical Physics 
at the University of Warsaw. These contributions form part of the theoretical
and methodological foundation for MVAR-based connectivity analysis.

Original MVAR methodology references:
- Kamiński, M., & Blinowska, K. J. (1991). A new method of the description 
  of the information flow in the brain structures. Biological Cybernetics, 
  65(3), 203-210.
- Kamiński, M., Ding, M., Truccolo, W. A., & Bressler, S. L. (2001).
  Evaluating causal relations in neural systems: Granger causality, directed 
  transfer function and statistical assessment of significance. Biological 
  Cybernetics, 85(2), 145-157.

Modified and adapted for educational purposes by:
Jarosław Żygierewicz, University of Warsaw
SYNCCIN 2025 Summer School

License: Educational use
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def count_corr(x, ip, iwhat):
    """
    Internal procedure
    """
    sdt = np.shape(x)
    m = sdt[0]
    n = sdt[1]
    if len(sdt) > 2:
        trials = sdt[2]
    else:
        trials = 1
    mip = m * ip
    r_left = np.zeros((mip, mip))
    r_right = np.zeros((mip, m))
    r = np.zeros((m, m))
    r_left_tot = np.zeros((mip, mip))
    r_right_tot = np.zeros((mip, m))
    r_tot = np.zeros((m, m))

    for trial in range(trials):
        for k in range(ip):
            if iwhat == 1:
                corr_scale = 1 / n
                nn = n - k - 1
                r[:, :] = np.dot(x[:, :nn, trial], x[:, k + 1:k + nn + 1, trial].T) * corr_scale
            elif iwhat == 2:
                corr_scale = 1 / (n - k)
                nn = n - k - 1
                r[:, :] = np.dot(x[:, :nn, trial], x[:, k + 1:k + nn + 1, trial].T) * corr_scale

            r_right[k * m:k * m + m, :] = r[:, :]

            if k < ip:
                for i in range(1, ip - k):
                    r_left[(k + i) * m:(k + i) * m + m, (i - 1) * m:(i - 1) * m + m] = r[:, :]
                    r_left[(i - 1) * m:(i - 1) * m + m, (k + i) * m:(k + i) * m + m] = r[:, :].T

        corr_scale = 1 / n
        r[:, :] = np.dot(x[:, :, trial], x[:, :, trial].T) * corr_scale

        for k in range(ip):
            r_left[k * m:k * m + m, k * m:k * m + m] = r[:, :]

        r_left_tot = r_left_tot + r_left
        r_right_tot = r_right_tot + r_right
        r_tot = r_tot + r

    if trials > 1:
        r_left_tot = r_left_tot / trials
        r_right_tot = r_right_tot / trials
        r_tot = r_tot / trials

    return r_left_tot, r_right_tot, r_tot


def ar_coeff(data, model_order=5):
    """
    Estimate MVAR coefficients for multivariate / multi-trial data.

    Parameters:
    data : np.ndarray
        Input time series with shape (channels, samples) or (channels, samples, trials).
    model_order : int
        Model order.

    Returns:
    ar_coeffs : np.ndarray
        AR coefficients with shape (channels, channels, model_order).
    variance : np.ndarray
        Residual covariance matrix with shape (channels, channels).
    """
    # Ensure data has a trials dimension: (channels, samples, trials)
    if data.ndim < 3:
        data = data[:, :, None]

    n_channels = data.shape[0]

    # Compute correlation/block-correlation matrices
    r_left, r_right, r_zero = count_corr(data, model_order, 1)

    # Solve for stacked AR coefficients (shape: (n_channels * p, n_channels))
    x = np.linalg.solve(r_left, r_right).T

    # Residual covariance: r_zero - x @ r_right
    variance = r_zero - x.dot(r_right)

    # Reshape coefficients into (channels, channels, p) to match previous behavior
    ar_coeffs = x.reshape(n_channels, model_order, n_channels).transpose((0, 2, 1))
    return ar_coeffs, variance


def mvar_transfer_function(ar_coeffs, freqs, fs):
    """
    Calculate the transfer function H from multivariate autoregressive coefficients AR
    
    Parameters:
    ar_coeffs : numpy.ndarray
        AR coefficient matrix with shape (chan, chan, p), where p is the model order.
    freqs : numpy.ndarray
        Frequency vector.
    fs : float
        Sampling frequency.

    Returns:
    transfer_function_matrix : numpy.ndarray
        Transfer function matrix with shape (chan, chan, len(freqs)).
    ar_matrix : numpy.ndarray
        Frequency-dependent AR matrix with shape (chan, chan, len(freqs)).
    """
    model_order = ar_coeffs.shape[2]
    n_freqs = len(freqs)
    chan = ar_coeffs.shape[0]

    transfer_function_matrix = np.zeros((chan, chan, n_freqs), dtype=complex)
    ar_matrix = np.zeros((chan, chan, n_freqs), dtype=complex)

    z = np.zeros((model_order, n_freqs), dtype=complex)
    for m in range(1, model_order + 1):
        z[m - 1, :] = np.exp(-m * 2 * np.pi * 1j * freqs / fs)

    for fi in range(n_freqs):
        a = np.eye(chan, dtype=complex)
        for m in range(model_order):
            a -= ar_coeffs[:, :, m] * z[m, fi].item()
        transfer_function_matrix[:, :, fi] = np.linalg.inv(a)
        ar_matrix[:, :, fi] = a

    return transfer_function_matrix, ar_matrix


def multivariate_spectra(signals, freqs, fs, max_model_order=20, optimal_model_order=None, crit_type='AIC'):
    """
    Compute the multivariate spectra for all channels in signals.
    
    Parameters:
    signals : np.ndarray
        Input signals of shape (N_chan, N_samp).
    freqs : np.ndarray
        Frequency vector.
    fs : float
        Sampling frequency.
    max_model_order : int
        Maximum model order.
    optimal_model_order : int or None
        Optimal model order. If None, it will be computed.
    crit_type : str
        Criterion type for model order selection.
    
    Returns:
    spectra : np.ndarray
        Multivariate spectra of shape (N_chan, N_chan, N_f).
    """
    if optimal_model_order is None:
        _, _, optimal_model_order = mvar_criterion(signals, max_model_order, crit_type, True)
        print('Optimal model order for all channels: p = ', str(optimal_model_order))
    else:
        print('Using provided model order: p = ', str(optimal_model_order))
    # Estimate AR coefficients and residual variance
    ar_coeffs, variance = ar_coeff(signals, optimal_model_order)
    transfer_function_matrix, _ = mvar_transfer_function(ar_coeffs, freqs, fs)
    n_chan = signals.shape[0]
    n_freqs = freqs.shape[0]
    spectra = np.zeros((n_chan, n_chan, n_freqs), dtype=np.complex128)  # initialize the multivariate spectrum
    for fi in range(n_freqs):  # compute spectrum for all channels
        spectra[:, :, fi] = transfer_function_matrix[:, :, fi].dot(variance.dot(transfer_function_matrix[:, :, fi].T))

    return spectra

def box_cox_transform(x, box_cox_lambda):
    """Elementwise Box-Cox power transform, `(x**box_cox_lambda - 1) / box_cox_lambda`.

    Parameters
    ----------
    x : np.ndarray
        Non-negative array to transform (e.g. a Granger/DTF connectivity cube).
    box_cox_lambda : float
        Box-Cox exponent. `-1` is the sentinel for "no transform" (returns `x`
        unchanged) -- deliberately distinct from the mathematically valid
        `box_cox_lambda=1` (a pure shift, `x - 1`), so callers can request a
        no-op pass-through explicitly and keep prior behaviour by default.
        Any other value applies the transform as-is; `box_cox_lambda=0` is a
        singularity of the formula (division by zero) and is not special-cased
        to the log(x) limit -- left to surface as a real error.

    Returns
    -------
    np.ndarray
        `x` unchanged if `box_cox_lambda == -1`, else the transformed array.
    """
    if box_cox_lambda == -1:
        return x
    return (x ** box_cox_lambda - 1) / box_cox_lambda


def dtf_estimator(signals, freqs, fs, max_model_order=20, optimal_model_order=None, crit_type='AIC', ESTIMATOR="dDTF", box_cox_lambda=-1):
    """
    Select and compute the directed transfer function (DTF) or its variants for the multivariate case.

    Parameters:
    signals : np.ndarray
        Input signals of shape (N_chan, N_samp).
    freqs : np.ndarray
        Frequency vector.
    fs : float
        Sampling frequency.
    max_model_order : int
        Maximum model order.
    optimal_model_order : int or None
        Optimal model order. If None, it will be computed.
    crit_type : str
        Criterion type for model order selection.
    ESTIMATOR : {'dDTF', 'ffDTF', 'GPDC'}, optional
        Estimator type for the directed transfer function calculation (default is "dDTF").
    box_cox_lambda : float, optional
        Box-Cox exponent applied to the returned cube via `box_cox_transform`
        (default -1 = no transform, preserving prior behaviour).

    Returns:
    np.ndarray
        Multivariate DTF (or its variant) of shape (N_chan, N_chan, N_f).
    """
    if ESTIMATOR == "dDTF":
        result = direct_dtf(signals, freqs, fs, max_model_order, optimal_model_order, crit_type)
    elif ESTIMATOR == "ffDTF":
        result = full_freq_dtf(signals, freqs, fs, max_model_order, optimal_model_order, crit_type)
    elif ESTIMATOR == "GPDC":
        result = gen_partial_directed_coherence(signals, freqs, fs, max_model_order, optimal_model_order, crit_type)
    else:
        raise ValueError(f"Unknown ESTIMATOR: {ESTIMATOR}")
    return box_cox_transform(result, box_cox_lambda)

def dtf_multivariate(signals, freqs, fs, max_model_order=20, optimal_model_order=None, crit_type='AIC', comment=None):
    """
    Compute the directed transfer function (DTF) for the multivariate case.
    Parameters:
    signals : np.ndarray    
        Input signals of shape (N_chan, N_samp).
    freqs : np.ndarray
        Frequency vector.
    fs : float
        Sampling frequency.
    max_model_order : int
        Maximum model order.
    optimal_model_order : int or None
        Optimal model order. If None, it will be computed.
    crit_type : str
        Criterion type for model order selection.
    Returns:
    np.ndarray
        Multivariate DTF of shape (N_chan, N_chan, N_f).
    """
    if optimal_model_order is None:
        _, _, optimal_model_order = mvar_criterion(signals, max_model_order, crit_type, False)
        comment_str = '' if comment is None else comment + ' '
        print(f'Optimal model order for all {comment_str}channels: p = {optimal_model_order}')
    else:
        print(f'Using provided model order: p = {optimal_model_order}')
    ar_coeffs, _ = ar_coeff(signals, optimal_model_order)
    transfer_function_matrix, _ = mvar_transfer_function(ar_coeffs, freqs, fs)
    dtf = np.abs(transfer_function_matrix) ** 2

    return dtf


def full_freq_dtf(signals, freqs, fs, max_model_order=20, optimal_model_order=None, crit_type='AIC'):
    """
    Compute the full-frequency directed transfer function (ffDTF) for the multivariate case.
    
    The ffDTF modifies the standard DTF normalization to integrate over the entire frequency 
    range, making cross-frequency comparisons more interpretable. Unlike standard DTF which 
    normalizes by frequency-specific inflows, ffDTF normalizes by the sum over all frequencies.
    
    Mathematical formula:
    Standard DTF: DTF_ij(f) = |H_ij(f)|² / Σ_k |H_ik(f)|²
    Full-frequency DTF: ffDTF_ij(f) = |H_ij(f)|² / Σ_f Σ_k |H_ik(f)|²
    
    The normalization takes into account inflows to channel i across the entire frequency range,
    allowing for more meaningful comparison of information flow at different frequencies.
    
    Reference:
    Korzeniewska, A., Mańczak, M., Kamiński, M., Blinowska, K. J., & Kasicki, S. (2003).
    Determination of information flow direction among brain structures by a modified directed transfer function (dDTF) method. 
    Journal of neuroscience methods, 125(1-2), 195-207.
    https://doi.org/10.1016/S0165-0270(03)00052-9
    
    Parameters:
    signals : np.ndarray
        Input signals of shape (N_chan, N_samp).
    freqs : np.ndarray
        Frequency vector.
    fs : float
        Sampling frequency.
    max_model_order : int
        Maximum model order.
    optimal_model_order : int or None
        Optimal model order. If None, it will be computed.
    crit_type : str
        Criterion type for model order selection.
        
    Returns:
    np.ndarray
        Full-frequency DTF of shape (N_chan, N_chan, N_f).
    """
    dtf = dtf_multivariate(signals, freqs, fs, max_model_order, optimal_model_order, crit_type)

    n_chan, _, n_freqs = dtf.shape
    # Normalize DTF to get full-frequency DTF (ffDTF)
    ff_dtf = np.zeros((n_chan, n_chan, n_freqs))
    for i in range(n_chan):  # rows
        for j in range(n_chan):  # columns
            ff_dtf[i, j, :] = dtf[i, j, :] / np.sum(dtf[i, :, :])
    return ff_dtf


def partial_coherence(spectra):
    """
    Compute the partial coherence using efficient boolean indexing for minors.
    
    Parameters:
    spectra : np.ndarray
        Spectral density matrix of shape (N_chan, N_chan, N_f).

    Returns:
    np.ndarray
        Partial coherence of shape (N_chan, N_chan, N_f).
    """
    n_chan, _, n_f = spectra.shape

    # Compute minors of spectra using boolean indexing (Method 2)
    spectra_minors = np.zeros((n_chan, n_chan, n_f), dtype=np.complex128)

    for i in range(n_chan):  # rows to delete
        for j in range(n_chan):  # columns to delete
            for fi in range(n_f):  # for each frequency
                # Extract the (N_chan x N_chan) matrix at frequency fi
                spectra_freq = spectra[:, :, fi]

                # Create minor by deleting row i and column j using boolean indexing
                if n_chan > 1:  # Only compute minor if matrix is larger than 1x1
                    # Create boolean masks
                    row_mask = np.ones(n_chan, dtype=bool)
                    col_mask = np.ones(n_chan, dtype=bool)
                    row_mask[i] = False
                    col_mask[j] = False

                    # Extract submatrix using boolean indexing
                    minor_matrix = spectra_freq[np.ix_(row_mask, col_mask)]
                    spectra_minors[i, j, fi] = np.linalg.det(minor_matrix)
                else:
                    spectra_minors[i, j, fi] = 1.0  # For 1x1 matrix, minor is 1

    # Compute partial coherence
    kappa = np.zeros((n_chan, n_chan, n_f), dtype=np.complex128)
    for i in range(n_chan):
        for j in range(n_chan):
            if i != j:  # Only compute for off-diagonal elements
                # Avoid division by zero
                denominator = np.sqrt(spectra_minors[i, i, :] * spectra_minors[j, j, :])
                kappa[i, j, :] = np.where(denominator != 0,
                                          spectra_minors[i, j, :] / denominator,
                                          0)
            # Diagonal elements are set to 1 (perfect coherence with itself)
            else:
                kappa[i, j, :] = 1.0

    return kappa


def direct_dtf(signals, freqs, fs, max_model_order=20, optimal_model_order=None, crit_type='AIC'):
    """
    Compute the direct directed transfer function (dDTF) for the multivariate case.
    
    The dDTF combines DTF with partial coherence to distinguish direct from indirect 
    connections, filtering out spurious connections that may arise from common inputs
    or indirect pathways.
    
    Mathematical formula:
    dDTF_ij(f) = DTF_ij(f) × |partial_coherence_ij(f)|
    
    Reference:
    Korzeniewska, A., Mańczak, M., Kamiński, M., Blinowska, K. J., & Kasicki, S. (2003).
    Determination of information flow direction among brain structures by a modified 
    directed transfer function (dDTF) method. Journal of Neuroscience Methods, 125(1-2), 195-207.
    https://doi.org/10.1016/S0165-0270(03)00052-9
    
    Parameters:
    signals : np.ndarray    
        Input signals of shape (N_chan, N_samp).
    freqs : np.ndarray
        Frequency vector.
    fs : float
        Sampling frequency.
    max_model_order : int
        Maximum model order.
    optimal_model_order : int, optional
        Optimal model order (if known).
    crit_type : str
        Criterion type for model order selection (e.g., 'AIC', 'BIC').
        
    Returns:
    d_dtf : np.ndarray
        Direct directed transfer function (dDTF) of shape (N_chan, N_chan, N_f).
    """
    # compute spectral density matrix
    spectra = multivariate_spectra(signals, freqs, fs, max_model_order, optimal_model_order, crit_type)
    # Compute partial coherence
    kappa = partial_coherence(spectra)
    # compute the ffDTF
    ff_dtf = full_freq_dtf(signals, freqs, fs, max_model_order, optimal_model_order, crit_type)
    # Compute dDTF using the formula: dDTF = ff_DTF * |kappa|
    d_dtf = ff_dtf * np.abs(kappa)

    return d_dtf


def gen_partial_directed_coherence(signals, freqs, fs, max_model_order=20, optimal_model_order=None, crit_type='AIC'):
    """
    Compute the generalized partial directed coherence (GPDC) for the multivariate case.
    
    GPDC is a frequency-domain measure that quantifies the directed influence of one 
    signal on another, based on the autoregressive coefficients in the frequency domain.
    This generalized version addresses the limitations of standard PDC when dealing with 
    systems that have unbalanced noise variances across channels by incorporating 
    proper normalization using the noise covariance structure.
    
    Mathematical formula:
    GPDC_ij(f) = |A_ij(f)|/σ_i / sqrt(Σ_k (|A_kj(f)|² / σ²_k))

    Where:
    - A(f) is the frequency domain AR coefficient matrix (inverse of H)
    - |A_ij(f)| is the absolute value of the AR coefficient from source channel i to target channel j at frequency f
    - σ_i is the noise standard deviation for source channel i (sqrt of diagonal elements of noise covariance matrix V)
    - σ²_k is the noise variance for source channel k (diagonal elements of noise covariance matrix V)
    
    This formulation ensures that channels with different noise levels do not 
    artificially bias connectivity estimates, making GPDC more robust than standard PDC 
    for practical applications with heterogeneous noise characteristics.
    
    References:
    Original PDC: Baccala, L. A., & Sameshima, K. (2001). Partial directed coherence: a new concept 
    in neural structure determination. Biological Cybernetics, 84(6), 463-474.
    https://doi.org/10.1007/PL00007990
    
    Generalized PDC: Baccalá, L. A., & Sameshima, K. (2007). Generalized partial directed coherence. 
    15th International Conference on Digital Signal Processing (DSP 2007), 163-166.
    https://doi.org/10.1109/ICDSP.2007.4288544
    
    Parameters:
    signals : np.ndarray    
        Input signals of shape (N_chan, N_samp).
    freqs : np.ndarray
        Frequency vector.
    fs : float
        Sampling frequency.
    max_model_order : int
        Maximum model order.
    optimal_model_order : int or None
        Optimal model order. If None, it will be computed.
    crit_type : str
        Criterion type for model order selection.
        
    Returns:
    gpdc : np.ndarray
        Generalized partial directed coherence (GPDC) of shape (N_chan, N_chan, N_f).
    """
    if optimal_model_order is None:
        _, _, optimal_model_order = mvar_criterion(signals, max_model_order, crit_type, False)
        print('Optimal model order for all channels: p = ', str(optimal_model_order))
    else:
        print('Using provided model order: p = ', str(optimal_model_order))

    # Estimate AR coefficients and residual variance
    ar_coeffs, variance = ar_coeff(signals, optimal_model_order)
    _, ar_matrix = mvar_transfer_function(ar_coeffs, freqs, fs)  # A is the frequency domain AR matrix

    n_chan, _, n_f = ar_matrix.shape
    gpdc = np.zeros((n_chan, n_chan, n_f))

    # Extract noise variances (diagonal of V)
    sigma_squared = np.diag(variance)

    # Compute GPDC 
    for i in range(n_chan):  # source
        for j in range(n_chan):  # target
            # Numerator: |A_ij(f)| / σ_i
            numerator = np.abs(ar_matrix[i, j, :]) / np.sqrt(sigma_squared[i])

            # Denominator: sqrt(Σ_k (|A_kj(f)|² / σ²_k))
            denominator = np.sqrt(np.sum(np.abs(ar_matrix[:, j, :]) ** 2 / sigma_squared[:, np.newaxis], axis=0))

            # Avoid division by zero
            gpdc[i, j, :] = np.where(denominator != 0,
                                     numerator / denominator,
                                     0)

    return gpdc


# Plotting function for graph visualization
def mvar_plot(on_diag, off_diag, freqs, x_label, y_label, chan_names, top_title, scale='linear', fig_size=(8, 8),
              band_hz=None, fig=None, max_on_diag=None, max_off_diag=None):
    """
    Plot MVAR results using bar plots for diagonal (auto) and off-diagonal (cross) terms.

    Parameters:
    on_diag : np.ndarray
        Auto components (shape: N_chan x N_chan x len(freqs))
    off_diag : np.ndarray
        Cross components (shape: N_chan x N_chan x len(freqs))
    freqs : np.ndarray
        Frequency vector
    x_label : str
        Label for x-axis
    y_label : str
        Label for y-axis
    chan_names : list of str
        Names of channels
    top_title : str
        Main plot title
    scale : str
        'linear', 'sqrt', or 'log'
    fig_size : tuple of int, optional
        Figure size (width, height). Ignored when `fig` is given. Default is (8, 8).
    band_hz : tuple of float, optional
        `(low, high)` frequency band edges in Hz. When given, shaded as a
        background span behind every subplot to mark a coupling band of
        interest. Default None (no shading).
    fig : matplotlib.figure.Figure or matplotlib.figure.SubFigure, optional
        Draw the n_chan x n_chan grid into this figure (or subfigure) instead
        of creating a new standalone figure -- lets a caller embed this plot
        as one panel of a larger composite figure (e.g. via
        `fig.subfigures(...)`). Default None creates a new figure via
        `plt.subplots(..., figsize=fig_size)`, the original behaviour.
    max_on_diag : float, optional
        Maximum value for the on-diagonal plots. Default is None, which uses the actual max.
    max_off_diag : float, optional
        Maximum value for the off-diagonal plots. Default is None, which uses the actual max.
    """
    on_diag = np.abs(on_diag)
    off_diag = np.abs(off_diag)

    if scale == 'sqrt':
        on_diag = np.sqrt(on_diag)
        off_diag = np.sqrt(off_diag)
    elif scale == 'log':
        on_diag = np.log(on_diag + 1e-12)  # Avoid log(0)
        off_diag = np.log(off_diag + 1e-12)

    n_chan = on_diag.shape[0]

    # Zero-out irrelevant parts
    for i in range(n_chan):
        for j in range(n_chan):
            if i != j:
                on_diag[i, j, :] = 0
            else:
                off_diag[i, i, :] = 0

    max_on_diag = np.max(on_diag) if max_on_diag is None else max_on_diag
    max_off_diag = np.max(off_diag) if max_off_diag is None else max_off_diag # we want to have the same scale for all plots, so we can set max_val to a fixed value if desired

    if fig is None:
        _, axs = plt.subplots(n_chan, n_chan, figsize=fig_size,
                              gridspec_kw={'wspace': 0, 'hspace': 0})
    else:
        axs = fig.subplots(n_chan, n_chan, gridspec_kw={'wspace': 0, 'hspace': 0})

    for i in range(n_chan):
        for j in range(n_chan):
            ax = axs[i, j] if n_chan > 1 else axs
            if band_hz is not None:
                ax.axvspan(band_hz[0], band_hz[1], color='orange', alpha=0.2, zorder=0, lw=0)
            if i != j:
                y = np.real(off_diag[i, j, :])
                ax.plot(freqs, off_diag[i, j, :])
                ax.fill_between(freqs, y, 0, color='skyblue', alpha=0.4)
                ax.set_ylim([0, max_off_diag])
                ax.set_yticks([0, max_off_diag // 2])
            else:
                y = np.real(on_diag[i, j, :])
                ax.plot(freqs, y, color=[0.7, 0.7, 0.7])
                ax.fill_between(freqs, y, 0, color=[0.7, 0.7, 0.7], alpha=0.4)
                ax.set_ylim([0, max_on_diag])
                ax.set_yticks([0, max_on_diag // 2])

            # Set xtick and ytick labels only on leftmost and bottom plots
            ax.tick_params(labelleft=(j == 0), labelbottom=(i == n_chan - 1))

            if i == n_chan - 1:
                ax.set_xlabel(f"{x_label}{chan_names[j]}")
            if j == 0:
                ax.set_ylabel(f"{y_label}{chan_names[i]}")

    if n_chan > 1:
        axs[0, 0].set_title(top_title)
    else:
        axs.set_title(top_title)
    # plt.tight_layout()


def mvar_criterion(data, max_model_order, crit_type='AIC', plot=False):
    """
    Compute model order selection criteria (AIC, HQ, SC) for MVAR modeling.

    Parameters:
    data : np.ndarray
        Input data of shape (channels, samples).
    max_model_order : int
        Maximum model order to evaluate.
    crit_type : str
        Criterion type: 'AIC', 'HQ', or 'SC'.
    plot : bool
        Whether to plot the criterion values.

    Returns:
    crit : np.ndarray
        Criterion values for each model order.
    p_range : np.ndarray
        Evaluated model order range (1:max_model_order).
    optimal_model_order : int
        Optimal model order (minimizing the criterion).
    """
    n_channels, n_samples = data.shape
    model_order_range = np.arange(1, max_model_order + 1, dtype=int)
    crit = np.zeros(max_model_order)

    for model_order in model_order_range:
        _, variance = ar_coeff(data, model_order)  # You must define or import AR_coeff
        if crit_type == 'AIC':
            crit[model_order - 1] = np.log(np.linalg.det(variance)) + 2 * model_order * n_channels ** 2 / n_samples
        elif crit_type == 'HQ':
            crit[model_order - 1] = np.log(np.linalg.det(variance)) + 2 * np.log(
                np.log(n_samples)) * model_order * n_channels ** 2 / n_samples
        elif crit_type == 'SC':
            crit[model_order - 1] = np.log(np.linalg.det(variance)) + np.log(
                n_samples) * model_order * n_channels ** 2 / n_samples
        else:
            raise ValueError("Invalid criterion type. Choose from 'AIC', 'HQ', 'SC'.")

    optimal_model_range = model_order_range[np.argmin(crit)]
    if plot:
        plt.figure()
        plt.plot(model_order_range, crit, marker='o')
        plt.plot(optimal_model_range, np.min(crit), 'ro')
        plt.xlabel('Model order p')
        plt.ylabel(f'{crit_type} criterion')
        plt.title(f'MVAR Model Order Selection ({crit_type}). The best order = {optimal_model_range}')
        plt.grid(True)
        plt.show()

    return crit, model_order_range, optimal_model_range


# ── FAD decomposition ──────────────────────────────────────────────────────────


def fad_decomposition(signal, fs, model_order=None, max_model_order=20,
                      crit_type='AIC', plot=False, pair_conjugates=True,
                      imag_tol=1e-8):
    """
    FAD (Frequency-Amplitude-Damping) decomposition of a univariate AR model.

    Fits an AR model to the signal and decomposes the transfer function H(z) into
    oscillatory components. Each component is characterised by:

    - freq_hz      : resonance frequency (Hz)
    - B            : amplitude  B_j = 2|C_j|
    - beta         : damping coefficient (s⁻¹);  positive = exponential decay
    - bandwidth_hz : half-power bandwidth proxy in Hz, matching the MATLAB
                     FAD script convention: bandwidth_hz = beta / (2*pi)
    - phi          : initial phase (rad)

    The impulse response of the AR model is:
        h(t) = Σ_j  B_j · exp(−β_j · t) · cos(ω_j · t + φ_j)

    This corresponds to the partial-fraction expansion of the transfer function
    (Franaszczuk et al., and Blinowska & Żygierewicz, *Practical Biomedical
    Signal Analysis*, 2nd ed., §FAD):

        H(z) = Σ_j  C_j · z / (z − z_j)

        α_j = ln(z_j) / Δt,   β_j = −Re(α_j),   ω_j = Im(α_j)
        φ_j = arg(C_j),        B_j = 2|C_j|

    Parameters
    ----------
    signal : array_like, shape (N,) or (1, N)
        Univariate time series.
    fs : float
        Sampling frequency [Hz].
    model_order : int or None
        AR model order p.  If None the order is chosen automatically via
        `crit_type`.
    max_model_order : int
        Maximum order considered during automatic selection (default 20).
    crit_type : str
        Information criterion used for automatic order selection:
        ``'AIC'`` (default), ``'HQ'``, or ``'SC'``.
    plot : bool
        If True, show a two-panel figure: poles in the unit circle (left) and
        AR power spectrum annotated with FAD component frequencies (right).
    pair_conjugates : bool
        If True (default), expose one oscillator per complex-conjugate pole
        pair in ``paired_components`` (positive-imaginary pole branch).
        This is the standard FAD reporting style (p/2 oscillators for even p).
    imag_tol : float
        Numerical tolerance used to classify poles as complex vs. real.

    Returns
    -------
    dict with keys
        ``'model_order'``   – int, fitted AR order p
        ``'poles'``         – ndarray (p,), complex poles z_j
        ``'C'``             – ndarray (p,), complex residues C_j
        ``'alpha'``         – ndarray (p,), α_j = ln(z_j)/Δt
        ``'freq_hz'``       – ndarray (p,), resonance frequency [Hz]
        ``'omega_rad_s'``   – ndarray (p,), angular frequency [rad/s]
        ``'beta'``          – ndarray (p,), damping coefficient [s⁻¹]
        ``'phi'``           – ndarray (p,), phase [rad]
        ``'bandwidth_hz'``  – ndarray (p,), bandwidth in Hz
        ``'B'``             – ndarray (p,), amplitude B_j = 2|C_j|
        ``'noise_variance'``– float, residual noise variance σ²
        ``'osc_mask'``      – bool ndarray (p,), True for complex poles
        ``'ar_coeffs'``     – ndarray (p,), fitted AR coefficients a_1 … a_p
        ``'paired_components'`` – dict containing one oscillator per conjugate
                      pair (frequency, amplitude, damping, phase)
    """
    from scipy.signal import residuez

    x = np.atleast_2d(np.asarray(signal, dtype=float))
    if x.shape[0] != 1:
        raise ValueError(
            "fad_decomposition expects a univariate signal, shape (N,) or (1, N)."
        )

    if model_order is None:
        _, _, model_order = mvar_criterion(x, max_model_order, crit_type, plot)
        print(f'FAD: optimal AR model order ({crit_type}) = {model_order}')

    ar_co, variance = ar_coeff(x, model_order)
    a = ar_co[0, 0, :]  # a_1 … a_p

    # Partial-fraction decomposition of H(z⁻¹) = 1 / A(z⁻¹)
    # A(z⁻¹) = 1 − a_1·z⁻¹ − … − a_p·z⁻ᵖ
    a_denom = np.concatenate([[1.0], -a])          # [1, −a₁, …, −aₚ]
    C, z_poles, _ = residuez(np.array([1.0]), a_denom)
    # residuez gives:  H = Σ C_j / (1 − z_j·z⁻¹)  =  Σ C_j · z/(z−z_j)

    # FAD parameters (eq. FAD_params in the textbook)
    dt = 1.0 / fs
    alpha   = np.log(z_poles) / dt          # α_j = ln(z_j) / Δt
    beta    = -np.real(alpha)               # β_j = −Re(α_j)  [s⁻¹]
    omega   =  np.imag(alpha)               # ω_j = Im(α_j)   [rad/s]
    freq_hz = omega / (2.0 * np.pi)         # f_j              [Hz]
    phi     = np.angle(C)                   # φ_j = arg(C_j)  [rad]
    B       = 2.0 * np.abs(C)              # B_j = 2|C_j|
    bandwidth_hz = beta / (2.0 * np.pi)     # MATLAB-style bandwidth in Hz

    # Oscillatory poles are complex (exclude purely real roots).
    osc_mask = np.abs(np.imag(z_poles)) > imag_tol

    # One oscillator per conjugate pair: use positive-imaginary branch only.
    pos_idx = np.where(np.imag(z_poles) > imag_tol)[0]
    if pos_idx.size > 0:
        order = np.argsort(freq_hz[pos_idx])
        pos_idx = pos_idx[order]

    if pair_conjugates:
        paired_idx = pos_idx
    else:
        paired_idx = np.where(osc_mask)[0]

    paired_components = {
        'pole_index': paired_idx,
        'poles': z_poles[paired_idx],
        'C': C[paired_idx],
        'alpha': alpha[paired_idx],
        'freq_hz': freq_hz[paired_idx],
        'omega_rad_s': omega[paired_idx],
        'beta': beta[paired_idx],
        'bandwidth_hz': bandwidth_hz[paired_idx],
        'phi': phi[paired_idx],
        'B': B[paired_idx],
    }

    fad_params = {
        'model_order'   : model_order,
        'poles'         : z_poles,
        'C'             : C,
        'alpha'         : alpha,
        'freq_hz'       : freq_hz,
        'omega_rad_s'   : omega,
        'beta'          : beta,
        'bandwidth_hz'  : bandwidth_hz,
        'phi'           : phi,
        'B'             : B,
        'noise_variance': float(np.real(variance[0, 0])),
        'osc_mask'      : osc_mask,
        'ar_coeffs'     : a,
        'paired_components': paired_components,
    }

    if plot:
        _fad_plot(x.ravel(), fs, fad_params)

    return fad_params


def fad_components_table(fad_params, output='dataframe', decimals=None):
    """
    Return compact FAD components table for export.

    The function uses ``fad_params['paired_components']`` so each row
    corresponds to one oscillator (one complex-conjugate pole pair).

    Parameters
    ----------
    fad_params : dict
        Output dictionary returned by ``fad_decomposition``.
    output : str
        Output format:
        - ``'dataframe'`` (default): pandas DataFrame
        - ``'ndarray'``: numeric ndarray with columns in fixed order
    decimals : int or None
        If provided, round numeric values to this number of decimals.

    Returns
    -------
    pandas.DataFrame or np.ndarray
        Table-like representation with columns:
        [component, freq_hz, omega_rad_s, beta_s_1, bandwidth_hz, B, phi_rad,
         pole_real, pole_imag, residue_real, residue_imag]

    Notes
    -----
    If ``output='dataframe'`` and pandas is not installed, an ImportError
    is raised. Use ``output='ndarray'`` for a dependency-free export path.
    """
    paired = fad_params.get('paired_components', None)
    if paired is None:
        raise ValueError("fad_params does not contain 'paired_components'.")

    n = len(paired['freq_hz'])
    component = np.arange(1, n + 1, dtype=int)

    table = np.column_stack([
        component,
        paired['freq_hz'],
        paired['omega_rad_s'],
        paired['beta'],
        paired['bandwidth_hz'],
        paired['B'],
        paired['phi'],
        np.real(paired['poles']),
        np.imag(paired['poles']),
        np.real(paired['C']),
        np.imag(paired['C']),
    ])

    if decimals is not None:
        table = np.round(table, decimals=decimals)

    if output == 'ndarray':
        return table

    if output == 'dataframe':
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "pandas is required for output='dataframe'. "
                "Use output='ndarray' instead."
            ) from exc

        columns = [
            'component',
            'freq_hz',
            'omega_rad_s',
            'beta_s_1',
            'bandwidth_hz',
            'B',
            'phi_rad',
            'pole_real',
            'pole_imag',
            'residue_real',
            'residue_imag',
        ]
        df = pd.DataFrame(table, columns=columns)
        df['component'] = df['component'].astype(int)
        return df

    raise ValueError("output must be 'dataframe' or 'ndarray'.")


def _fad_plot(signal, fs, fad_params):
    """Two-panel FAD plot: poles in the unit circle and annotated AR spectrum."""
    a           = fad_params['ar_coeffs']
    model_order = fad_params['model_order']
    var         = fad_params['noise_variance']
    z_poles     = fad_params['poles']
    osc_mask    = fad_params['osc_mask']
    freq_hz     = fad_params['freq_hz']
    B           = fad_params['B']
    beta        = fad_params['beta']
    bandwidth_hz = fad_params['bandwidth_hz']
    paired      = fad_params.get('paired_components', None)

    # AR power spectrum: S(f) = σ² / |A(f)|²
    n_fft  = 2048
    f_axis = np.linspace(0, fs / 2.0, n_fft // 2 + 1)
    A_f    = np.ones(len(f_axis), dtype=complex)
    for m in range(model_order):
        A_f -= a[m] * np.exp(-(m + 1) * 2j * np.pi * f_axis / fs)
    ar_spectrum = var / np.abs(A_f) ** 2

    # Colours for oscillatory components (one per conjugate pair)
    # Always restrict to the positive-imaginary branch to avoid duplicates
    # when pair_conjugates=False includes both halves of each conjugate pair.
    if paired is not None and paired['pole_index'].size > 0:
        pos_mask = np.imag(z_poles[paired['pole_index']]) > 1e-8
        pair_idx = paired['pole_index'][pos_mask]
        pair_f = paired['freq_hz'][pos_mask]
        pair_B = paired['B'][pos_mask]
        pair_beta = paired['beta'][pos_mask]
        pair_bw_hz = paired['bandwidth_hz'][pos_mask]
    else:
        pair_idx = np.where(np.imag(z_poles) > 1e-8)[0]
        pair_f = freq_hz[pair_idx]
        pair_B = B[pair_idx]
        pair_beta = beta[pair_idx]
        pair_bw_hz = bandwidth_hz[pair_idx]

    n_osc  = int(len(pair_idx))
    colors = (plt.cm.viridis(np.linspace(0.15, 0.9, n_osc))
              if n_osc > 0 else np.empty((0, 4)))

    theta = np.linspace(0, 2 * np.pi, 300)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'FAD decomposition  (AR order p = {model_order})', fontsize=13)

    # ── Panel 1: poles in the complex plane ───────────────────────────────────
    ax0 = axes[0]
    ax0.plot(np.cos(theta), np.sin(theta), 'k-', lw=0.9, alpha=0.4)
    ax0.axhline(0, color='k', lw=0.5, alpha=0.4)
    ax0.axvline(0, color='k', lw=0.5, alpha=0.4)

    ci = 0
    pair_set = set(pair_idx.tolist())
    for j, z in enumerate(z_poles):
        if j in pair_set:
            c = colors[ci]
            ax0.scatter(np.real(z), np.imag(z), color=c, s=70, zorder=5,
                        label=f'{pair_f[ci]:.2f} Hz')
            ax0.scatter(np.real(z), -np.imag(z), color=c, s=70, zorder=5,
                        marker='x')
            ci += 1
        elif not osc_mask[j]:
            ax0.scatter(np.real(z), np.imag(z), color='gray', s=40, zorder=4,
                        marker='s')

    ax0.set_xlim(-1.25, 1.25)
    ax0.set_ylim(-1.25, 1.25)
    ax0.set_aspect('equal')
    ax0.set_xlabel('Re(z)')
    ax0.set_ylabel('Im(z)')
    ax0.set_title('Poles in the complex plane')
    ax0.legend(fontsize=8, loc='upper left', title='freq [Hz]')
    ax0.grid(True, alpha=0.3)

    # ── Panel 2: AR power spectrum with FAD annotations ───────────────────────
    ax1 = axes[1]
    ax1.plot(f_axis, ar_spectrum, 'k-', lw=1.5, label='AR spectrum')

    for ci, (f, b_val, bt, bw_hz) in enumerate(zip(pair_f, pair_B, pair_beta, pair_bw_hz)):
        c = colors[ci]
        peak_idx = np.argmin(np.abs(f_axis - f))
        peak_power = ar_spectrum[peak_idx]
        half_power = peak_power / 2.0
        left_bw = max(0.0, f - bw_hz)
        right_bw = min(fs / 2.0, f + bw_hz)

        ax1.plot([f, f], [0, peak_power], color='r', lw=1.2, alpha=0.9)
        ax1.plot([left_bw, right_bw], [half_power, half_power], color='m', lw=1.8, alpha=0.9)
        ax1.scatter([f], [peak_power], color=c, s=28, zorder=6)
        ax1.annotate(
            f'{f:.1f} Hz\nBW={bw_hz:.2f} Hz\nB={b_val:.3g}',
            xy=(f, peak_power), xytext=(6, 8),
            textcoords='offset points', fontsize=7, color=c,
        )

    ax1.set_xlabel('Frequency [Hz]')
    ax1.set_ylabel('Power spectral density')
    ax1.set_title('AR power spectrum with peak positions and bandwidth')
    ax1.set_xlim(0, fs / 2.0)
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    plt.tight_layout()
    plt.show()


# Compute linewidths
def get_linewidths(graph):
    weights = np.array([d['weight'] for u, v, d in graph.edges(data=True)])
    return 5 * weights / weights.max()


def graph_plot(connectivity_matrix, ax, freqs, freq_range, chan_names, title):
    """
    Plot connectivity matrix as a graph.
    Parameters:
    connectivity_matrix : np.ndarray, shape (N_chan, N_chan, N_f)
        Connectivity matrix with complex values. e.g. Directed Transfer Function (DTF).
    ax : matplotlib.axes.Axes
        Axes on which to plot the graph.
    freqs : np.ndarray
        Frequency vector.
    f_range : tuple
        Frequency range for the plot (min, max).
    chan_names : list
        List of channel names.
    title : str
        Title for the plot. 
    Returns:
    G : networkx.DiGraph
        Directed graph created from the connectivity matrix.

    """
    # Convert complex DTF to real for visualization
    connectivity = connectivity_matrix.real
    # Sum over the frequencies in f_range and transpose to match the directionality of edges (from row to column) in the directed graph
    # find the indices of the frequency range
    f_indices = np.where((freqs >= freq_range[0]) & (freqs <= freq_range[1]))[0]
    if len(f_indices) == 0:
        raise ValueError("No frequencies found in the specified range.")
    adj = np.sum(connectivity[:, :, f_indices], axis=2).T
    # Remove self-loops by setting diagonal to zero
    np.fill_diagonal(adj, 0)
    # Create directed graphs
    graph = nx.from_numpy_array(adj, create_using=nx.DiGraph)
    # Plotting
    pos = nx.spring_layout(graph)  # use the same layout for both
    # Map chan_names to node labels
    labels = {i: chan_names[i] for i in range(len(chan_names))}
    nx.draw(graph, pos, ax=ax, with_labels=True, labels=labels, arrows=True,
            width=get_linewidths(graph), node_size=500,
            arrowstyle='->',
            arrowsize=35,
            connectionstyle='arc3,rad=0.2')
    ax.set_title(title)

    return graph


def compute_and_plot_mvar(
    ncdf_path,
    channel_subset=None,
    max_model_order=20,
    optimal_model_order=None,
    crit_type="AIC",
    freq_min=1.0,
    freq_max=40.0,
    freq_step=0.5,
    low_cutoff_hz=None,
    high_cutoff_hz=None,
    plot=True,
    plot_loaded_signal=False,
    loaded_signal_max_channels=19,
    loaded_signal_spacing=8.0,
    loaded_signal_figsize=(16.0, 9.0),
):
    """Load one EEG NetCDF file, compute ffDTF and multivariate spectra, and
    optionally plot results.

    This is a high-level pipeline that chains :func:`~src.export.load_eeg_signals`,
    :func:`mvar_criterion`, :func:`full_freq_dtf`, :func:`multivariate_spectra`,
    and :func:`mvar_plot`.

    Args:
        ncdf_path: Path to the exported EEG NetCDF file.
        channel_subset: Optional list of channel names to keep (``None`` = all).
        max_model_order: Upper bound for the model-order search.
        optimal_model_order: Fixed model order.  If ``None``, the order is
            chosen automatically using ``crit_type``.
        crit_type: Information criterion for model-order selection:
            ``'AIC'``, ``'HQ'``, or ``'SC'``.
        freq_min: Minimum frequency of the analysis axis (Hz).
        freq_max: Maximum frequency of the analysis axis (Hz).
        freq_step: Frequency resolution (Hz).
        low_cutoff_hz: High-pass filter cutoff (Hz).  ``None`` disables it.
        high_cutoff_hz: Low-pass filter cutoff (Hz).  ``None`` disables it.
        plot: If ``True``, display the MVAR spectra / ffDTF matrix plot.
        plot_loaded_signal: If ``True``, display the stacked EEG trace plot
            before running the MVAR analysis.
        loaded_signal_max_channels: Max channels shown in the signal preview.
        loaded_signal_spacing: Vertical spacing between traces in the preview.
        loaded_signal_figsize: Figure size for the signal preview.

    Returns:
        tuple:
            - **ff_dtf** (*np.ndarray*): Full-frequency DTF array.
            - **spectra** (*np.ndarray*): Multivariate spectra array.
            - **chan_names** (*list[str]*): Channel labels.
            - **crit** (*np.ndarray*): Information-criterion values per model
              order (empty array when ``optimal_model_order`` is fixed).
            - **model_order_range** (*np.ndarray*): Tested model orders (empty
              when ``optimal_model_order`` is fixed).
            - **p_opt** (*int*): Selected (or provided) model order.
    """
    from .multimodal_io import load_eeg_signals, plot_loaded_eeg_signals

    signals, chan_names, fs, time_s, event_duration_s = load_eeg_signals(
        ncdf_path,
        channel_subset=channel_subset,
        low_cutoff_hz=low_cutoff_hz,
        high_cutoff_hz=high_cutoff_hz,
    )

    if plot_loaded_signal:
        plot_loaded_eeg_signals(
            time_s=time_s,
            signals=signals,
            channel_names=chan_names,
            max_channels=loaded_signal_max_channels,
            spacing=loaded_signal_spacing,
            figsize=loaded_signal_figsize,
            event_duration_s=event_duration_s,
            title=f"Loaded EEG signal — {ncdf_path.stem if hasattr(ncdf_path, 'stem') else ncdf_path}",
        )

    freqs = np.arange(freq_min, freq_max + freq_step, freq_step)
    attrs_label = (
        f"{ncdf_path.stem if hasattr(ncdf_path, 'stem') else ncdf_path}"
        f"  (fs={fs:.0f} Hz, {signals.shape[0]} ch, {signals.shape[1]} samp)"
    )
    print(f"\n  Channels : {chan_names}")
    print(f"  Signals  : {signals.shape}  fs={fs} Hz")

    if optimal_model_order is None:
        crit, model_order_range, p_opt = mvar_criterion(signals, max_model_order, crit_type, plot=False)
        print(f"  {crit_type} optimal model order: p = {p_opt}")
    else:
        p_opt = optimal_model_order
        print(f"  Using fixed model order: p = {p_opt}")
        crit = np.array([])
        model_order_range = np.array([])

    print("  Computing ffDTF ...")
    ff_dtf = full_freq_dtf(
        signals, freqs, fs,
        max_model_order=max_model_order,
        optimal_model_order=p_opt,
        crit_type=crit_type,
    )

    print("  Computing multivariate spectra ...")
    spectra = multivariate_spectra(
        signals, freqs, fs,
        max_model_order=max_model_order,
        optimal_model_order=p_opt,
        crit_type=crit_type,
    )

    if plot:
        plt.figure(figsize=(10, 10))
        mvar_plot(
            spectra, ff_dtf, freqs,
            x_label="", y_label="",
            chan_names=chan_names,
            top_title=f"MVAR Spectra and ff_DTF — {ncdf_path.stem if hasattr(ncdf_path, 'stem') else ncdf_path}",
            scale="linear",
        )
        plt.suptitle(attrs_label, fontsize=8, y=1.01)
        plt.tight_layout()
        plt.show()

    return ff_dtf, spectra, chan_names, crit, model_order_range, p_opt
