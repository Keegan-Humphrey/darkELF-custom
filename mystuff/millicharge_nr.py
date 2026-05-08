
"""
Millicharged nuclear-recoil material/rate utilities extracted from the Mathematica notebook
`directD2 for Keegan (1).nb` and adapted for Python/DarkELF-style use.

What is taken directly from the notebook
---------------------------------------
- The pure millicharge differential cross section uses the "MC" process flag:
      dσ/dER = 8 π α_em^2 (kappa * Q_dm)^2 m_T / (v^2 q^4) * F_pp(q)
  with q^2 = 2 m_T ER.
- The target isotope lists, approximate isotope masses, isotope abundances, and recoil windows
  for Xe / Ge / Si / He come from the notebook.

What is approximated here
-------------------------
- The notebook encodes isotope-specific proton-proton nuclear response polynomials FM[p,p](y).
  To keep this implementation clean and portable, this module uses a standard Helm charge form factor
  approximation:
      F_pp(q) ≈ Z^2 * F_Helm(q)^2
  This preserves the correct millicharge/charge-coupling structure and the q^{-4} Rutherford behavior.
- If you want, the exact FM[p,p](y) polynomials can be wired in later as a drop-in replacement.

Units
-----
- Energies and masses are in eV.
- Cross sections are in cm^2 / eV for dσ/dER and cm^2 for integrated reference cross sections.
- Rates are in events / kg / year / eV or events / kg / year.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import math
import numpy as np

PI = math.pi
ALPHA_EM = 1.0 / 137.0
AMU_EV = 0.9315e9       # atomic mass unit in eV
HC_EV_FM = 197.3269804e6  # (eV * fm)
EV_CM = 1.9732e-5       # eV * cm
YEAR_S = 365.0 * 24.0 * 3600.0
EV_TO_INV_S = 1.5192674e15
EV_TO_INV_YR = EV_TO_INV_S * YEAR_S


@dataclass(frozen=True)
class Isotope:
    name: str
    A: int
    Z: int
    abundance: float
    mass_eV: float

@dataclass(frozen=True)
class NRMaterial:
    name: str
    recoil_window_eV: tuple[float, float]
    isotopes: tuple[Isotope, ...]


# Extracted from the notebook's isotope/threshold blocks.
# The notebook appears to contain a typo Anum["Ge74"] = 4; that is corrected here to 74.
NR_MATERIALS: Dict[str, NRMaterial] = {
    "Xe": NRMaterial(
        name="Xe",
        recoil_window_eV=(6.6e3, 43.3e3),  # 6.6 keV to 43.3 keV
        isotopes=(
            Isotope("Xe128", 128, 54, 0.019, 119 * AMU_EV),
            Isotope("Xe129", 129, 54, 0.260, 120 * AMU_EV),
            Isotope("Xe130", 130, 54, 0.041, 121 * AMU_EV),
            Isotope("Xe131", 131, 54, 0.210, 122 * AMU_EV),
            Isotope("Xe132", 132, 54, 0.270, 123 * AMU_EV),
            Isotope("Xe134", 134, 54, 0.100, 125 * AMU_EV),
            Isotope("Xe136", 136, 54, 0.089, 126 * AMU_EV),
        ),
    ),
    "Ge": NRMaterial(
        name="Ge",
        recoil_window_eV=(40.0, 43.3e3),   # 40 eV to 43.3 keV
        isotopes=(
            Isotope("Ge70", 70, 32, 0.200, 65 * AMU_EV),
            Isotope("Ge72", 72, 32, 0.270, 67 * AMU_EV),
            Isotope("Ge74", 74, 32, 0.370, 69 * AMU_EV),
            Isotope("Ge76", 76, 32, 0.078, 71 * AMU_EV),
            Isotope("Ge73", 73, 32, 0.078, 68 * AMU_EV),
        ),
    ),
    "Si": NRMaterial(
        name="Si",
        recoil_window_eV=(78.0, 43.3e3),   # 78 eV to 43.3 keV
        isotopes=(
            Isotope("Si28", 28, 14, 0.920, 26 * AMU_EV),
        ),
    ),
    "He": NRMaterial(
        name="He",
        recoil_window_eV=(0.0, 43.3e3),    # notebook did not show an explicit He threshold block nearby
        isotopes=(
            Isotope("He4", 4, 2, 1.0, 4 * AMU_EV),
        ),
    ),
}


def helm_form_factor_squared(q_eV: np.ndarray | float, A: int) -> np.ndarray | float:
    """
    Standard Helm form factor squared, dimensionless.

    q_eV: momentum transfer in eV
    A: mass number
    """
    q = np.asarray(q_eV, dtype=float)

    # Standard Helm parameters in fm.
    s = 0.9
    a = 0.52
    c = 1.23 * A**(1.0 / 3.0) - 0.60
    rn = math.sqrt(max(c * c + (7.0 / 3.0) * PI * PI * a * a - 5.0 * s * s, 1e-12))

    q_fm = q / HC_EV_FM
    qr = q_fm * rn

    # spherical bessel j1(x) = sin(x)/x^2 - cos(x)/x
    out = np.ones_like(qr)
    small = np.abs(qr) < 1e-6
    xs = qr[~small]
    j1 = np.sin(xs) / xs**2 - np.cos(xs) / xs
    out[~small] = (3.0 * j1 / xs) ** 2 * np.exp(-(q_fm[~small] * s) ** 2)
    out[small] = 1.0 #  (3.0 * j1 / xs) requires high numerical precision at small xs, and is undefined at xs=0, so just take limiting value for small xs
    return out if isinstance(q_eV, np.ndarray) else float(out)


def nuclear_response_pp_approx(q_eV: np.ndarray | float, isotope: Isotope) -> np.ndarray | float:
    """
    Approximate proton-proton response entering the notebook's MC cross section:
        F_pp(q) ~ Z^2 F_Helm(q)^2
    """
    return isotope.Z**2 * helm_form_factor_squared(q_eV, isotope.A)


def dsigma_dER_mc(
    
    ER_eV: np.ndarray | float,
    v: np.ndarray | float,
    isotope: Isotope,
    kappa: float,
    q_dm: float = 1.0,
) -> np.ndarray | float:
    r"""
    Pure millicharge ("MC") differential cross section from the notebook, in cm^2 / eV:

        dσ/dER = 8 π α_em^2 (kappa * q_dm)^2 m_T / (v^2 q^4) * F_pp(q)

    with q^2 = 2 m_T ER.

    Here v is dimensionless (units of c).
    """
    ER = np.asarray(ER_eV, dtype=float)
    vv = np.asarray(v, dtype=float)
    mT = isotope.mass_eV
    q2 = 2.0 * mT * ER
    q = np.sqrt(np.maximum(q2, 1e-300))

    pref = 8.0 * PI * (ALPHA_EM**2) * (kappa * q_dm) ** 2 * mT
    resp = nuclear_response_pp_approx(q, isotope)

    # natural-units expression gives eV^{-3}; convert by (eV*cm)^2 -> cm^2 and leave /eV
    out = pref * resp / (np.maximum(vv, 1e-300) ** 2 * np.maximum(q, 1e-300) ** 4)
    out = out * (EV_CM ** 2)

    return out if (np.ndim(ER) or np.ndim(vv)) else float(out)

# C_CM_S = 2.99792458e10

# def dsigma_dER_mc(
#     ER_eV: np.ndarray | float,
#     isotope: Isotope,
#     kappa: float,
#     q_dm: float = 1.0,
# ) -> np.ndarray | float:
#     r"""
#     Returns v^2 * dσ/dER in cm^2/eV:

#         v^2 dσ/dER =
#             8π α^2 (kappa q_dm)^2 m_T / q^4
#             × Z^2 F_Helm(q)^2
#     """
#     ER = np.asarray(ER_eV, dtype=float)
#     mT = isotope.mass_eV

#     q2 = 2.0 * mT * ER
#     q = np.sqrt(np.maximum(q2, 1e-300))

#     pref = 8.0 * PI * ALPHA_EM**2 * (kappa * q_dm)**2 * mT
#     resp = nuclear_response_pp_approx(q, isotope)

#     out = pref * resp / np.maximum(q, 1e-300)**4

#     # eV^-3 -> cm^2/eV
#     out = out * EV_CM**2

#     return out if np.ndim(ER) else float(out)


def vmin_nr(ER_eV: np.ndarray | float, mT_eV: float, mX_eV: float) -> np.ndarray | float:
    """
    Elastic nuclear-recoil v_min in units of c:
        v_min = q / (2 mu_{XT})
              = sqrt(mT ER / (2 mu_{XT}^2))
    """
    ER = np.asarray(ER_eV, dtype=float)
    muXT = mX_eV * mT_eV / (mX_eV + mT_eV)
    out = np.sqrt(np.maximum(mT_eV * ER / (2.0 * muXT * muXT), 0.0))
    return out if np.ndim(ER) else float(out)


def dRdER_mc_isotope(
    delfobj,
    ER_eV: np.ndarray | float,
    isotope: Isotope,
    kappa: float,
    q_dm: float = 1.0,
    vdist: str = "halo",
) -> np.ndarray | float:
    """
    Differential rate for one isotope in events / kg / year / eV, using the velocity
    integral methods already present on the supplied DarkELF-like object.

    Required attributes on delfobj:
      - mX, rhoX, eVtoInvYr
      - etav(vmin) and/or etav_disk(vmin)

    The number of nuclei per kg is computed from the isotope mass.
    """
    ER = np.asarray(ER_eV, dtype=float)
    mT = isotope.mass_eV
    vmin = vmin_nr(ER, mT, delfobj.mX)

    if vdist == "halo":
        eta = delfobj.etav(vmin)
    elif vdist == "disk":
        eta = delfobj.etav_disk(vmin)
    else:
        raise ValueError("vdist must be 'halo' or 'disk'")

    NTkg = 1.0 / (mT * delfobj.eVtokg)
    ds = dsigma_dER_mc(ER, np.maximum(vmin, 1e-300), isotope, kappa=kappa, q_dm=q_dm)

    out = (delfobj.rhoX / delfobj.mX) * NTkg * delfobj.eVtoInvYr * eta * ds
    return out if np.ndim(ER) else float(out)

# def dRdER_mc_isotope(
#     delfobj,
#     ER_eV: np.ndarray | float,
#     isotope: Isotope,
#     kappa: float,
#     q_dm: float = 1.0,
#     vdist: str = "halo",
# ) -> np.ndarray | float:
#     """
#     Differential rate for one isotope in events / kg / year / eV.

#     Assumes delfobj.etav(vmin) is the inverse-speed integral appropriate
#     for dσ/dER ∝ 1/v^2.
#     """
#     ER = np.asarray(ER_eV, dtype=float)
#     mT = isotope.mass_eV
#     vmin = vmin_nr(ER, mT, delfobj.mX)

#     if vdist == "halo":
#         eta = delfobj.etav(vmin)
#     elif vdist == "disk":
#         eta = delfobj.etav_disk(vmin)
#     else:
#         raise ValueError("vdist must be 'halo' or 'disk'")

#     NTkg = 1.0 / (mT * delfobj.eVtokg)

#     # A(ER), not dσ/dER evaluated at v=vmin.
#     ds_vindep = dsigma_dER_mc(
#         ER, isotope, kappa=kappa, q_dm=q_dm
#     )

#     out = (delfobj.rhoX / delfobj.mX) * NTkg * delfobj.eVtoInvYr * eta * ds_vindep
#     return out if np.ndim(ER) else float(out)


def dRdER_mc_material(
    delfobj,
    ER_eV: np.ndarray | float,
    material: str,
    kappa: float,
    q_dm: float = 1.0,
    vdist: str = "halo",
) -> np.ndarray | float:
    """
    Isotopically averaged differential recoil spectrum in events / kg / year / eV.
    """
    mat = NR_MATERIALS[material]
    ER = np.asarray(ER_eV, dtype=float)
    total = np.zeros_like(ER, dtype=float)
    for iso in mat.isotopes:
        total += iso.abundance * dRdER_mc_isotope(
            delfobj, ER, iso, kappa=kappa, q_dm=q_dm, vdist=vdist
        )
    return total if np.ndim(ER) else float(total)


def R_mc_material(
    delfobj,
    material: str,
    kappa: float,
    q_dm: float = 1.0,
    vdist: str = "halo",
    ER_min_eV: Optional[float] = None,
    ER_max_eV: Optional[float] = None,
    npts: int = 800,
) -> float:
    """
    Total event rate in events / kg / year integrated over the material's recoil window
    (or user-specified bounds).
    """
    mat = NR_MATERIALS[material]
    lo, hi = mat.recoil_window_eV
    if ER_min_eV is not None:
        lo = max(lo, ER_min_eV)
    if ER_max_eV is not None:
        hi = min(hi, ER_max_eV)
    if not (hi > lo > 0.0):
        return 0.0

    grid = np.geomspace(lo, hi, npts)
    spec = dRdER_mc_material(delfobj, grid, material, kappa=kappa, q_dm=q_dm, vdist=vdist)
    return float(np.trapz(spec, x=grid))


def attach_mc_nr_methods(darkelf_cls) -> None:
    """
    Monkey-patch convenience methods onto a DarkELF-like class.

    After calling:
        attach_mc_nr_methods(darkelf)

    you can do:
        delf.dRdER_mc(ER, material="Ge", kappa=1e-9, q_dm=2.0, vdist="disk")
        delf.R_mc(material="Ge", kappa=1e-9, q_dm=2.0, vdist="disk")
    """
    def _dRdER_mc(self, ER_eV, material="Ge", kappa=1e-9, q_dm=1.0, vdist="halo"):
        return dRdER_mc_material(self, ER_eV, material=material, kappa=kappa, q_dm=q_dm, vdist=vdist)

    def _R_mc(self, material="Ge", kappa=1e-9, q_dm=1.0, vdist="halo",
              ER_min_eV=None, ER_max_eV=None, npts=800):
        return R_mc_material(self, material=material, kappa=kappa, q_dm=q_dm,
                             vdist=vdist, ER_min_eV=ER_min_eV, ER_max_eV=ER_max_eV, npts=npts)

    setattr(darkelf_cls, "dRdER_mc", _dRdER_mc)
    setattr(darkelf_cls, "R_mc", _R_mc)


if __name__ == "__main__":
    print("Available NR materials:", ", ".join(NR_MATERIALS.keys()))
