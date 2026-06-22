"""Rescaled hydrogen atomic form factor for a neutral dark-hydrogen state.

The fit is the standard four-Gaussian SM hydrogen X-ray atomic form factor

    f_H(q) = sum_i a_i exp[-b_i (q_Ainv / 4 pi)^2] + c,

with q_Ainv in inverse Angstrom.  A dark atom with Bohr momentum
q_B,D = alpha_D * mu_D is obtained by evaluating the same fit at
q -> q * q_B,SM / q_B,D.

The quantity relevant for Coulomb scattering of a neutral atom is the
screened charge amplitude

    F_atom(q) = Z_fit - f_H,D(q),

where Z_fit = f_H(0).  This exactly mirrors the convention used by
FormFactors.wl, where Zeff = Z - AFF and Z is reset to the fitted f(0).
"""
from __future__ import annotations

import numpy as np

ALPHA_EM = 1.0 / 137.035999084
M_E_EV = 510_998.95069
M_P_EV = 938.27208816e6
HBARC_EV_ANGSTROM = 1973.269804

H_A = np.array([0.489918, 0.262003, 0.196767, 0.049879], dtype=float)
H_B = np.array([20.6593, 7.74039, 49.5519, 2.20159], dtype=float)
H_C = 0.001305
H_Z_FIT = float(np.sum(H_A) + H_C)


def reduced_mass(m1_eV, m2_eV):
    """Reduced mass in eV, accepting scalars or NumPy arrays."""
    m1 = np.asarray(m1_eV, dtype=float)
    m2 = np.asarray(m2_eV, dtype=float)
    out = m1 * m2 / (m1 + m2)
    return float(out) if out.ndim == 0 else out


MU_H_SM_EV = reduced_mass(M_E_EV, M_P_EV)
Q_BOHR_H_SM_EV = ALPHA_EM * MU_H_SM_EV


def hydrogen_aff_sm(q_eV):
    """SM hydrogen atomic form factor f_H(q), with q supplied in eV."""
    q = np.asarray(q_eV, dtype=float)
    x2 = (q / (4.0 * np.pi * HBARC_EV_ANGSTROM)) ** 2
    out = np.sum(H_A * np.exp(-x2[..., None] * H_B), axis=-1) + H_C
    return float(out) if out.ndim == 0 else out


def dark_hydrogen_aff(q_eV, alpha_D, m_eD_eV, m_pD_eV):
    """Rescaled bound-electron form factor f_D(q)."""
    if alpha_D <= 0.0 or m_eD_eV <= 0.0 or m_pD_eV <= 0.0:
        raise ValueError("alpha_D, m_eD_eV, and m_pD_eV must all be positive")

    mu_D = reduced_mass(m_eD_eV, m_pD_eV)
    q_bohr_D = alpha_D * mu_D
    q_rescaled = np.asarray(q_eV, dtype=float) * (Q_BOHR_H_SM_EV / q_bohr_D)
    return hydrogen_aff_sm(q_rescaled)


def dark_hydrogen_effective_charge(q_eV, alpha_D, m_eD_eV, m_pD_eV):
    """Dimensionless neutral-atom charge amplitude Z_fit - f_D(q)."""
    return H_Z_FIT - dark_hydrogen_aff(q_eV, alpha_D, m_eD_eV, m_pD_eV)


def dark_hydrogen_screening_squared(q_eV, alpha_D, m_eD_eV, m_pD_eV):
    """Squared atomic screening factor entering a rate or cross section."""
    zeff = dark_hydrogen_effective_charge(q_eV, alpha_D, m_eD_eV, m_pD_eV)
    return np.asarray(zeff) ** 2
