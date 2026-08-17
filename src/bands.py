"""Functional band assignment (posterior dominant rhythm, frontal midline theta, mu)."""

POSTERIOR_CHANNELS = ['P3', 'Pz', 'P4', 'O1', 'O2']
CENTRAL_CHANNELS = ['C3', 'Cz', 'C4']


def assign_bands(peaks_df, participant_id, age=None):
    """Assign functional band labels to a participant's detected peaks.

    Implements a simple decision sequence:

    1. The strongest peak (by power) at posterior channels is the posterior
       dominant rhythm ("alpha").
    2. A peak at Fz at least 1.5 Hz below the posterior peak is frontal midline
       theta. Without a posterior reference, an Fz peak below 7 Hz is still
       called theta; otherwise it is flagged as ambiguous.
    3. A central (C3/Cz/C4) peak in 7-13 Hz that differs from the posterior
       peak by more than 1 Hz is called mu; within 1 Hz it is treated as the
       same generator as the posterior rhythm.
    4. For participants with no posterior peak, an age-adjusted expected
       frequency (5.5 + 0.5 x age_years) is noted when ``age`` is given.

    Parameters
    ----------
    peaks_df : pandas.DataFrame
        Peak inventory table with columns ``participant_id``, ``channel``,
        ``center_freq``, ``power``, ``bandwidth``.
    participant_id : str
        Participant to assign bands for.
    age : float or None, optional
        Age in years, used only for the posterior-peak fallback note.

    Returns
    -------
    dict
        ``{participant_id, posterior_alpha_cf, posterior_alpha_bw,
        frontal_theta_cf, frontal_theta_bw, mu_cf, mu_bw, assignment_notes}``.
        Frequency/bandwidth fields are ``None`` where no peak was assigned.
    """
    subset = peaks_df[peaks_df['participant_id'] == participant_id]
    notes = []

    # 1. Posterior dominant rhythm ("alpha")
    posterior = subset[subset['channel'].isin(POSTERIOR_CHANNELS)]
    posterior_cf, posterior_bw = None, None
    if len(posterior):
        top = posterior.loc[posterior['power'].idxmax()]
        posterior_cf, posterior_bw = float(top['center_freq']), float(top['bandwidth'])
    elif age is not None:
        expected = 5.5 + 0.5 * age
        notes.append(f'no posterior peak; age-adjusted fallback ~{expected:.1f} Hz')
    else:
        notes.append('no posterior peak')

    # 2. Frontal midline theta
    frontal = subset[subset['channel'] == 'Fz']
    frontal_cf, frontal_bw = None, None
    if len(frontal):
        top_f = frontal.loc[frontal['power'].idxmax()]
        if posterior_cf is not None:
            if top_f['center_freq'] <= posterior_cf - 1.5:
                frontal_cf, frontal_bw = float(top_f['center_freq']), float(top_f['bandwidth'])
            else:
                notes.append('frontal peak within 1.5 Hz of posterior alpha; not assigned as separate theta generator')
        elif top_f['center_freq'] < 7:
            frontal_cf, frontal_bw = float(top_f['center_freq']), float(top_f['bandwidth'])
        else:
            notes.append('frontal peak present but ambiguous (no posterior reference, cf >= 7 Hz)')

    # 3. Mu rhythm
    central = subset[subset['channel'].isin(CENTRAL_CHANNELS)]
    central_in_range = central[(central['center_freq'] >= 7) & (central['center_freq'] <= 13)]
    mu_cf, mu_bw = None, None
    if len(central_in_range):
        top_c = central_in_range.loc[central_in_range['power'].idxmax()]
        if posterior_cf is None or abs(top_c['center_freq'] - posterior_cf) > 1:
            mu_cf, mu_bw = float(top_c['center_freq']), float(top_c['bandwidth'])
        else:
            notes.append('central peak within 1 Hz of posterior alpha; treated as same generator')

    return {
        'participant_id': participant_id,
        'posterior_alpha_cf': posterior_cf,
        'posterior_alpha_bw': posterior_bw,
        'frontal_theta_cf': frontal_cf,
        'frontal_theta_bw': frontal_bw,
        'mu_cf': mu_cf,
        'mu_bw': mu_bw,
        'assignment_notes': '; '.join(notes),
    }
