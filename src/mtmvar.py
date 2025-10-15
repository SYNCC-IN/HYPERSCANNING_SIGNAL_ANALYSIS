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
import pandas as pd
import scipy.signal as signal
import matplotlib.pyplot as plt
import networkx as nx

def countCorr(x, ip, iwhat):
    '''
    Internal procedure 
    '''
    sdt = np.shape(x)
    m = sdt[0]
    n = sdt[1]
    if len(sdt) > 2:
        trials = sdt[2]
    else:
        trials = 1
    mip = m * ip
    rleft = np.zeros((mip, mip))
    rright = np.zeros((mip, m))
    r = np.zeros((m, m))
    rleft_tot = np.zeros((mip, mip))
    rright_tot = np.zeros((mip, m))
    r_tot = np.zeros((m, m))

    for trial in range(trials):
        for k in range(ip):
            if iwhat == 1:
                corrscale = 1 / n
                nn = n - k - 1
                r[:, :] = np.dot(x[:, :nn, trial], x[:, k + 1:k + nn + 1, trial].T) * corrscale
            elif iwhat == 2:
                corrscale = 1 / (n - k)
                nn = n - k - 1
                r[:, :] = np.dot(x[:, :nn, trial], x[:, k + 1:k + nn + 1, trial].T) * corrscale
        
            rright[k * m:k * m + m, :] = r[:, :]
        
            if k < ip:
                for i in range(1, ip - k):
                    rleft[(k + i) * m:(k + i) * m + m, (i - 1) * m:(i - 1) * m + m] = r[:, :]
                    rleft[(i - 1) * m:(i - 1) * m + m, (k + i) * m:(k + i) * m + m] = r[:, :].T
        
        corrscale = 1 / n
        r[:, :] = np.dot(x[:, :, trial], x[:, :, trial].T) * corrscale
    
        for k in range(ip):
            rleft[k * m:k * m + m, k * m:k * m + m] = r[:, :]
    
        rleft_tot = rleft_tot + rleft
        rright_tot = rright_tot + rright
        r_tot = r_tot + r

    if trials > 1:
        rleft_tot = rleft_tot / trials
        rright_tot = rright_tot / trials
        r_tot = r_tot / trials

    return rleft_tot, rright_tot, r_tot

def AR_coeff(dat0, p=5, method=1):
    '''
    Estimates AR model coefficients
    for multivariate/multitrial data.

    dat - input time series of shape (trials,chans,timepoints)
    p - model order
    method - unused

    returns:
    ARr - model coefficients of size (p,chans,chans)
    Vr - residual data variance matrix
    '''
    
    ds = dat0.shape
    if len(ds) < 3:
        dat = np.zeros((ds[0], ds[1], 1))
        dat[:, :, 0] = dat0[:, :]
    else:
        dat = dat0
    chans = ds[0]

    rleft, rright, r = countCorr(dat, p, 1)
    xres = np.linalg.solve(rleft, rright)  
    xres= xres.T  
    Vr = -np.dot(xres, rright) + r
    AR = np.reshape(xres, ( chans, p,chans)) #(chans, p, chans))# ( chans, chans,p,)) #
    AR = AR.transpose((0, 2, 1))
    return AR, Vr


def mvar_H(Ar, f, Fs):
    """
    Calculate the transfer function H from multivariate autoregressive coefficients Ar.
    
    Parameters:
    Ar : numpy.ndarray
        AR coefficient matrix with shape (chan, chan, p), where p is model order.
    f : numpy.ndarray
        Frequency vector.
    Fs : float
        Sampling frequency.

    Returns:
    H : numpy.ndarray
        Transfer function matrix with shape (chan, chan, len(f)).
    A_out : numpy.ndarray
        Frequency-dependent AR matrix with shape (chan, chan, len(f)).
    """
    p = Ar.shape[2]
    Nf = len(f)
    chan = Ar.shape[0]
    
    H = np.zeros((chan, chan, Nf), dtype=complex)
    A_out = np.zeros((chan, chan, Nf), dtype=complex)

    z = np.zeros((p, Nf), dtype=complex)
    for m in range(1, p + 1):
        z[m-1, :] = np.exp(-m * 2 * np.pi * 1j * f / Fs)

    for fi in range(Nf):
        A = np.eye(chan, dtype=complex)
        for m in range(p):
            A -= Ar[:, :,m] * z[m, fi].item()
        H[:, :, fi] = np.linalg.inv(A)
        A_out[:, :, fi] = A

    return H, A_out


def bivariate_spectra(signals, f, Fs, max_p, p_opt = None, crit_type='AIC'):
    """
    Compute the bivariate spectra for each pair of channels in signals.
    Parameters:
    signals : np.ndarray
        Input signals of shape (N_chan, N_samp).
    f : np.ndarray
        Frequency vector.
    Fs : float
        Sampling frequency.
    max_p : int
        Maximum model order.
    p_opt : int or None
        Optimal model order. If None, it will be computed.
    crit_type : str
        Criterion type for model order selection.   
    Returns:
    np.ndarray
        Bivariate spectra of shape (N_chan, N_chan, N_f).
    """
    N_chan = signals.shape[0]
    N_f = f.shape[0]
    S_bivariate = np.zeros((N_chan,N_chan, N_f), dtype=np.complex128   ) #initialize the bivariate spectrum
    for ch1 in range(N_chan):
        for ch2 in range(ch1+1,N_chan):
            x = np.vstack((signals[ch1,:],
                           signals[ch2,:]))
            if p_opt is None:
                _, _, p_opt = mvar_criterion(x, max_p, crit_type, False)
                print('Optimal model order for channel pair: ', str(ch1), ' and ', str(ch2), ' p = ', str(p_opt))   
            else:
                print('Using provided model order: p = ', str(p_opt))   
            # Estimate AR coefficients and residual variance
            Ar, V = AR_coeff(x, p_opt)
            H, _ = mvar_H(Ar, f, Fs)
            S_2chan = np.zeros((2,2, N_f), dtype=np.complex128) #initialize the bivariate spectrum for the pair of channels
            for fi in range(N_f): #compute spectrum for the pair ch1, ch2
                S_2chan[:,:,fi] = H[:,:,fi].dot(V.dot(H[:,:,fi].T))
            S_bivariate[ch1,ch1,:] = S_2chan[0,0,:]
            S_bivariate[ch2,ch2,:] = S_2chan[1,1,:]
            S_bivariate[ch1,ch2,:] = S_2chan[0,1,:]
            S_bivariate[ch2,ch1,:] = S_2chan[1,0,:]
    return S_bivariate

def multivariate_spectra(signals, f, Fs, max_p, p_opt = None, crit_type='AIC'):
    """
    Compute the multivariate spectra for all channels in signals.
    
    Parameters:
    signals : np.ndarray
        Input signals of shape (N_chan, N_samp).
    f : np.ndarray
        Frequency vector.
    Fs : float
        Sampling frequency.
    max_p : int
        Maximum model order.
    p_opt : int or None
        Optimal model order. If None, it will be computed.
    crit_type : str
        Criterion type for model order selection.
    
    Returns:
    np.ndarray
        Multivariate spectra of shape (N_chan, N_chan, N_f).
    """
    x = signals
    if p_opt is None:
        _, _, p_opt = mvar_criterion(x, max_p, crit_type, True)
        print('Optimal model order for all channels: p = ', str(p_opt))
    else:
        print('Using provided model order: p = ', str(p_opt))
    # Estimate AR coefficients and residual variance
    Ar, V = AR_coeff(x, p_opt)
    H, _ = mvar_H(Ar, f, Fs)
    N_chan = x.shape[0]
    N_f = f.shape[0]
    S_multivariate = np.zeros((N_chan,N_chan, N_f), dtype=np.complex128) #initialize the multivariate spectrum
    for fi in range(N_f): #compute spectrum for all channels
        S_multivariate[:,:,fi] = H[:,:,fi].dot(V.dot(H[:,:,fi].T))      
    
    return S_multivariate

def DTF_bivariate(signals, f, Fs, max_p = 20, p_opt = None, crit_type='AIC'):
    """
    Compute the directed transfer function (DTF) for the bivariate case.
    Parameters:
    signals : np.ndarray
        Input signals of shape (N_chan, N_samp).
    f : np.ndarray
        Frequency vector.
    Fs : float
        Sampling frequency.
    max_p : int
        Maximum model order.
    p_opt : int or None
        Optimal model order. If None, it will be computed.
    crit_type : str
        Criterion type for model order selection.
    Returns:
    np.ndarray
        Bivariate DTF of shape (N_chan, N_chan, N_f).
    """
    N_chan = signals.shape[0]
    N_f = f.shape[0]
    DTF_bivariate = np.zeros((N_chan,N_chan, N_f),dtype=np.complex128) #initialize the bivariate DTF; 
    for ch1 in range(N_chan):
        for ch2 in range(ch1+1,N_chan):
            x = np.vstack( (signals[ch1,:],
                            signals[ch2,:]) )  
            if p_opt is None:  
                _, _, p_opt = mvar_criterion(x, max_p, crit_type, False)
            Ar, _ = AR_coeff(x, p_opt)
            H, _ = mvar_H(Ar, f, Fs)
            
            DTF_2chan = np.abs(H)**2
            
            DTF_bivariate[ch1,ch2,:] = DTF_2chan[0,1,:]
            DTF_bivariate[ch2,ch1,:] = DTF_2chan[1,0,:]
    return DTF_bivariate

def DTF_multivariate(signals, f, Fs, max_p = 20, p_opt = None, crit_type='AIC'):
    """
    Compute the directed transfer function (DTF) for the multivariate case.
    Parameters:
    signals : np.ndarray    
        Input signals of shape (N_chan, N_samp).
    f : np.ndarray
        Frequency vector.
    Fs : float
        Sampling frequency.
    max_p : int
        Maximum model order.
    p_opt : int or None
        Optimal model order. If None, it will be computed.
    crit_type : str
        Criterion type for model order selection.
    Returns:
    np.ndarray
        Multivariate DTF of shape (N_chan, N_chan, N_f).
    """
    if p_opt is None:
        _, _, p_opt = mvar_criterion(signals, max_p, crit_type, False)
        print('Optimal model order for all channels: p = ', str(p_opt))
    else:
        print('Using provided model order: p = ', str(p_opt))
    Ar, _ = AR_coeff(signals, p_opt)
    H, _ = mvar_H(Ar, f, Fs)
    DTF = np.abs(H)**2

    return DTF

def ffDTF(signals, f, Fs, max_p = 20, p_opt = None, crit_type='AIC'):
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
    f : np.ndarray
        Frequency vector.
    Fs : float
        Sampling frequency.
    max_p : int
        Maximum model order.
    p_opt : int or None
        Optimal model order. If None, it will be computed.
    crit_type : str
        Criterion type for model order selection.
        
    Returns:
    np.ndarray
        Full-frequency DTF of shape (N_chan, N_chan, N_f).
    """
    if p_opt is None:
        _, _, p_opt = mvar_criterion(signals, max_p, crit_type, False)
        print('Optimal model order for all channels: p = ', str(p_opt))
    else:
        print('Using provided model order: p = ', str(p_opt))
    Ar, _ = AR_coeff(signals, p_opt)
    H, _ = mvar_H(Ar, f, Fs)
    DTF = np.abs(H)**2
    N_chan, _, N_f = DTF.shape
    # Normalize DTF to get full-frequency DTF (ffDTF)
    ffDTF = np.zeros((N_chan,N_chan, N_f))
    for i in range(N_chan):  # rows
        for j in range(N_chan):  # columns
            ffDTF[i,j,:] = DTF[i,j,:] / np.sum(DTF[i,:,:])
    return ffDTF

def partial_coherence(S):
    """
    Compute the partial coherence using efficient boolean indexing for minors.
    
    Parameters:
    S : np.ndarray
        Spectral density matrix of shape (N_chan, N_chan, N_f).

    Returns:
    np.ndarray
        Partial coherence of shape (N_chan, N_chan, N_f).
    """
    N_chan, _, N_f = S.shape
    
    # Compute minors of S using boolean indexing (Method 2)
    M = np.zeros((N_chan, N_chan, N_f), dtype=np.complex128)
    
    for i in range(N_chan):  # rows to delete
        for j in range(N_chan):  # columns to delete
            for fi in range(N_f):  # for each frequency
                # Extract the (N_chan x N_chan) matrix at frequency fi
                S_freq = S[:, :, fi]
                
                # Create minor by deleting row i and column j using boolean indexing
                if N_chan > 1:  # Only compute minor if matrix is larger than 1x1
                    # Create boolean masks
                    row_mask = np.ones(N_chan, dtype=bool)
                    col_mask = np.ones(N_chan, dtype=bool)
                    row_mask[i] = False
                    col_mask[j] = False
                    
                    # Extract submatrix using boolean indexing
                    minor_matrix = S_freq[np.ix_(row_mask, col_mask)]
                    M[i, j, fi] = np.linalg.det(minor_matrix)
                else:
                    M[i, j, fi] = 1.0  # For 1x1 matrix, minor is 1
    
    # Compute partial coherence
    kappa = np.zeros((N_chan, N_chan, N_f), dtype=np.complex128)
    for i in range(N_chan):
        for j in range(N_chan):
            if i != j:  # Only compute for off-diagonal elements
                # Avoid division by zero
                denominator = np.sqrt(M[i, i, :] * M[j, j, :])
                kappa[i, j, :] = np.where(denominator != 0, 
                                        M[i, j, :] / denominator, 
                                        0)
            # Diagonal elements are set to 1 (perfect coherence with itself)
            else:
                kappa[i, j, :] = 1.0
    
    return kappa

def dDTF(signals, f, Fs, max_p = 20, p_opt = None, crit_type='AIC'):
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
    f : np.ndarray
        Frequency vector.
    Fs : float
        Sampling frequency.
    max_p : int
        Maximum model order.
    p_opt : int, optional
        Optimal model order (if known).
    crit_type : str
        Criterion type for model order selection (e.g., 'AIC', 'BIC').
        
    Returns:
    np.ndarray
        Direct directed transfer function (dDTF) of shape (N_chan, N_chan, N_f).
    """
    if p_opt is None:
        _, _, p_opt = mvar_criterion(signals, max_p, crit_type, False)
        print('Optimal model order for all channels: p = ', str(p_opt))
    else:
        print('Using provided model order: p = ', str(p_opt))  
    # compute spectral density matrix
    S = multivariate_spectra(signals, f, Fs, max_p, p_opt, crit_type)
    # Compute partial coherence
    kappa = partial_coherence(S)
    # compute the ffDTF
    ff_DTF = ffDTF(signals, f, Fs, max_p, p_opt, crit_type)
    # Compute dDTF using the formula: dDTF = ff_DTF * kappa
    dDTF = ff_DTF * np.abs(kappa)

    return dDTF

def GPDC(signals, f, Fs, max_p = 20, p_opt = None, crit_type='AIC'):
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
    f : np.ndarray
        Frequency vector.
    Fs : float
        Sampling frequency.
    max_p : int
        Maximum model order.
    p_opt : int or None
        Optimal model order. If None, it will be computed.
    crit_type : str
        Criterion type for model order selection.
        
    Returns:
    np.ndarray
        Generalized partial directed coherence (GPDC) of shape (N_chan, N_chan, N_f).
    """
    if p_opt is None:
        _, _, p_opt = mvar_criterion(signals, max_p, crit_type, False)
        print('Optimal model order for all channels: p = ', str(p_opt))
    else:
        print('Using provided model order: p = ', str(p_opt))
    
    # Estimate AR coefficients and residual variance
    Ar, V = AR_coeff(signals, p_opt)
    H, A = mvar_H(Ar, f, Fs)  # A is the frequency domain AR matrix
    
    N_chan, _, N_f = A.shape
    GPDC = np.zeros((N_chan, N_chan, N_f))
    
    # Extract noise variances (diagonal of V)
    sigma_squared = np.diag(V)
    
    # Compute GPDC 
    for i in range(N_chan):  # source
        for j in range(N_chan):  # target
            # Numerator: |A_ij(f)| / σ_i
            numerator = np.abs(A[i, j, :]) / np.sqrt(sigma_squared[i])
            
            # Denominator: sqrt(Σ_k (|A_kj(f)|² / σ²_k))
            denominator = np.sqrt(np.sum(np.abs(A[:, j, :])**2 / sigma_squared[:, np.newaxis], axis=0))
            
            # Avoid division by zero
            GPDC[i, j, :] = np.where(denominator != 0, 
                                   numerator / denominator, 
                                   0)

    return GPDC

# Plotting function for graph visualization
def mvar_plot(onDiag, offDiag, f, xlab, ylab, ChanNames, Top_title, scale='linear'):
    """
    Plot MVAR results using bar plots for diagonal (auto) and off-diagonal (cross) terms.

    Parameters:
    onDiag : np.ndarray
        Auto components (shape: N_chan x N_chan x len(f))
    offDiag : np.ndarray
        Cross components (shape: N_chan x N_chan x len(f))
    f : np.ndarray
        Frequency vector
    xlab : str
        Label for x-axis
    ylab : str
        Label for y-axis
    ChanNames : list of str
        Names of channels
    Top_title : str
        Main plot title
    scale : str
        'linear', 'sqrt', or 'log'
    """
    onDiag = np.abs(onDiag)
    offDiag = np.abs(offDiag)

    if scale == 'sqrt':
        onDiag = np.sqrt(onDiag)
        offDiag = np.sqrt(offDiag)
    elif scale == 'log':
        onDiag = np.log(onDiag + 1e-12)  # Avoid log(0)
        offDiag = np.log(offDiag + 1e-12)

    N_chan = onDiag.shape[0]

    # Zero-out irrelevant parts
    for i in range(N_chan):
        for j in range(N_chan):
            if i != j:
                onDiag[i, j, :] = 0
            else:
                offDiag[i, i, :] = 0

    MaxonDiag = np.max(onDiag)
    MaxoffDiag = np.max(offDiag)

    fig, axs = plt.subplots(N_chan, N_chan, figsize=(6,6), constrained_layout=True)
    for i in range(N_chan):
        for j in range(N_chan):
            ax = axs[i, j] if N_chan > 1 else axs
            if i != j:
                y = np.real(offDiag[i, j, :])
                ax.plot(f, offDiag[i, j, :])
                ax.fill_between(f, y, 0, color='skyblue', alpha=0.4)
                ax.set_ylim([0, MaxoffDiag])
            else:
                y = np.real(onDiag[i, j, :])
                ax.plot(f, y, color=[0.7, 0.7, 0.7])
                ax.fill_between(f, y, 0, color=[0.7, 0.7, 0.7], alpha=0.4)
                ax.set_ylim([0, MaxonDiag])

            if i == N_chan - 1:
                ax.set_xlabel(f"{xlab}{ChanNames[j]}")
            if j == 0:
                ax.set_ylabel(f"{ylab}{ChanNames[i]}")

    axs[0, 0].set_title(Top_title)
    #plt.tight_layout()

def mvar_plot_dense(onDiag, offDiag, f, xlab, ylab,ChanNames , Top_title, scale='linear'):
    """
    Plot MVAR results for diagonal (auto) and off-diagonal (cross) terms.

    Parameters:
    onDiag : np.ndarray
        Auto components (shape: N_chan x N_chan x len(f))
    offDiag : np.ndarray
        Cross components (shape: N_chan x N_chan x len(f))
    f : np.ndarray
        Frequency vector
    xlab : str
        Label for x-axis
    ylab : str
        Label for y-axis
    ChanNames : list of str
        Names of channels
    Top_title : str
        Main plot title
    scale : str
        'linear', 'sqrt', or 'log'
    """
    import numpy as np
    import matplotlib.pyplot as plt

    onDiag = np.abs(onDiag)
    offDiag = np.abs(offDiag)

    if scale == 'sqrt':
        onDiag = np.sqrt(onDiag)
        offDiag = np.sqrt(offDiag)
    elif scale == 'log':
        onDiag = np.log(onDiag + 1e-12)  # Avoid log(0)
        offDiag = np.log(offDiag + 1e-12)

    N_chan = onDiag.shape[0]

    # Zero-out irrelevant parts
    for i in range(N_chan):
        for j in range(N_chan):
            if i != j:
                onDiag[i, j, :] = 0
            else:
                offDiag[i, i, :] = 0

    MaxonDiag = np.max(onDiag)
    MaxoffDiag = np.max(offDiag)

    fig, axs = plt.subplots(N_chan, N_chan, figsize=(8,8),
                            gridspec_kw={'wspace': 0, 'hspace': 0})

    for i in range(N_chan):
        for j in range(N_chan):
            ax = axs[i, j] if N_chan > 1 else axs
            if i != j:
                y = np.real(offDiag[i, j, :])
                ax.plot(f, offDiag[i, j, :])
                ax.fill_between(f, y, 0, color='skyblue', alpha=0.4)
                ax.set_ylim([0, MaxoffDiag])
                ax.set_yticks([0, MaxoffDiag // 2])
            else:
                y = np.real(onDiag[i, j, :])
                ax.plot(f, y, color=[0.7, 0.7, 0.7])
                ax.fill_between(f, y, 0, color=[0.7, 0.7, 0.7], alpha=0.4)
                ax.set_ylim([0, MaxonDiag])
                ax.set_yticks([0, MaxonDiag // 2])

            ax.set_xticks([f[0], int(f[len(f)//2]) ])
            ax.tick_params(labelleft=(j == 0), labelbottom=(i == N_chan - 1))

            if i == N_chan - 1:
                ax.set_xlabel(f"{xlab}{ ChanNames[j]}")
            if j == 0:
                ax.set_ylabel(f"{ylab}{ChanNames[i]}")

    if N_chan > 1:
        axs[0, 0].set_title(Top_title)
    else:
        axs.set_title(Top_title)
    #plt.tight_layout()

def mvar_criterion(dat, max_p, crit_type='AIC', do_plot=False):
    """
    Compute model order selection criteria (AIC, HQ, SC) for MVAR modeling.

    Parameters:
    dat : np.ndarray
        Input data of shape (channels, samples).
    max_p : int
        Maximum model order to evaluate.
    crit_type : str
        Criterion type: 'AIC', 'HQ', or 'SC'.
    do_plot : bool
        Whether to plot the criterion values.

    Returns:
    crit : np.ndarray
        Criterion values for each model order.
    p_range : np.ndarray
        Evaluated model order range (1:max_p).
    p_opt : int
        Optimal model order (minimizing the criterion).
    """
    k, N = dat.shape
    p_range = np.arange(1, max_p + 1)
    crit = np.zeros(max_p)

    for p in p_range:
        _, Vr = AR_coeff(dat, p)  # You must define or import AR_coeff
        if crit_type == 'AIC':
            crit[p-1] = np.log(np.linalg.det(Vr)) + 2 * p * k**2 / N
        elif crit_type == 'HQ':
            crit[p-1] = np.log(np.linalg.det(Vr)) + 2 * np.log(np.log(N)) * p * k**2 / N
        elif crit_type == 'SC':
            crit[p-1] = np.log(np.linalg.det(Vr)) + np.log(N) * p * k**2 / N
        else:
            raise ValueError("Invalid criterion type. Choose from 'AIC', 'HQ', 'SC'.")

    p_opt = p_range[np.argmin(crit)]
    if do_plot:
        plt.figure()
        plt.plot(p_range, crit, marker='o')
        plt.plot(p_opt, np.min(crit), 'ro')
        plt.xlabel('Model order p')
        plt.ylabel(f'{crit_type} criterion')
        plt.title(f'MVAR Model Order Selection ({crit_type}). The best order = {p_opt}')
        plt.grid(True)
        plt.show()

    return crit, p_range, p_opt


def mvar_spectra(H, V, f  ):
    """
    Calculate the MVAR spectra from transfer function H and noise covariance V. 
    Parameters:
    H : np.ndarray
        Transfer function matrix with shape (chan, chan, len(f)).
    V : np.ndarray      
        Noise covariance matrix with shape (chan, chan).
    f : np.ndarray  
        Frequency vector.           
    Returns:    
    S : np.ndarray
        MVAR spectra with shape (chan, chan, len(f)).
    """ 
    N_chan, _, N_f = H.shape
    # check if f has the same length as the third dimension of H
    if len(f) != N_f:
        raise ValueError("Frequency vector f must match the third dimension of H.") 
    S_multivariate = np.zeros((N_chan,N_chan, N_f), dtype=np.complex128) #initialize the multivariate spectrum
    for fi in range(N_f): #compute spectrum for all channels
        S_multivariate[:,:,fi] = H[:,:,fi].dot(V.dot(H[:,:,fi].T)) 
    return S_multivariate, f

# Compute linewidths
def get_linewidths(G):
    weights = np.array([d['weight'] for u, v, d in G.edges(data=True)])
    return 5 * weights / weights.max()

def graph_plot(connectivity_matrix, ax, f, f_range, ChanNames, title):
    
    """
    Plot connectivity matrix as a graph.
    Parameters:
    connectivity_matrix : np.ndarray, shape (N_chan, N_chan, N_f)
        Connectivity matrix with complex values. e.g. Directed Transfer Function (DTF).
    ax : matplotlib.axes.Axes
        Axes on which to plot the graph.
    f : np.ndarray
        Frequency vector.
    f_range : tuple
        Frequency range for the plot (min, max).
    ChanNames : list
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
    f_indices = np.where((f >= f_range[0]) & (f <= f_range[1]))[0]
    if len(f_indices) == 0:
        raise ValueError("No frequencies found in the specified range.")
    adj  = np.sum(connectivity[:,:,f_indices], axis=2).T
    # Remove self-loops by setting diagonal to zero
    np.fill_diagonal(adj, 0)
    # Create directed graphs
    G = nx.from_numpy_array(adj, create_using=nx.DiGraph)
    # Plotting
    pos = nx.spring_layout(G)  # use the same layout for both
    # Map ChanNames to node labels
    labels = {i: ChanNames[i] for i in range(len(ChanNames))}
    nx.draw(G, pos, ax=ax, with_labels=True, labels=labels, arrows=True,
            width=get_linewidths(G), node_size=500,
            arrowstyle='->',
            arrowsize=35,
            connectionstyle='arc3,rad=0.2')
    ax.set_title(title)

    return G

