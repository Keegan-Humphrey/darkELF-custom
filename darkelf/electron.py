import numpy as np
from numpy import linspace, sqrt, array, pi, cos, sin, dot, exp, sinh, log, log10, cosh, sinh
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy import integrate

from atomic_dark_hydrogen_ff import dark_hydrogen_screening_squared

############################################################################################

def dRdomegadk_electron(
    self, omega, k, sigmae=1e-38, withscreening=True, method="grid", vdist="halo",
    with_atomic_ff=False, alpha_D=None, m_eD_eV=None, m_pD_eV=None,
):
    """
    Returns double differential rate for DM-electron scattering in 1/kg/year/eV^2.

    Set with_atomic_ff=True to multiply the point-particle rate by the rescaled
    neutral dark-hydrogen screening factor |Z_fit - f_D(k)|^2.  In that case
    alpha_D, m_eD_eV, and m_pD_eV are required.
    """
    etav_val = self.etav(self.vmin(omega,k)) if vdist == "halo" else self.etav_disk(self.vmin(omega,k))
    temp_eps1 = self.eps1(omega,k,method=method)
    temp_eps2 = self.eps2(omega,k,method=method)

    if method not in ("grid", "Lindhard"):
        raise ValueError("method must be 'grid' or 'Lindhard'")

    atomic_factor2 = 1.0
    if with_atomic_ff:
        if alpha_D is None or m_eD_eV is None or m_pD_eV is None:
            raise ValueError(
                "with_atomic_ff=True requires alpha_D, m_eD_eV, and m_pD_eV"
            )
        atomic_factor2 = dark_hydrogen_screening_squared(
            k, alpha_D=alpha_D, m_eD_eV=m_eD_eV, m_pD_eV=m_pD_eV
        )

    dR = etav_val * self.rhoX/self.mX * 1000./self.rhoT * self.eVtoInvYr \
            * 1/(2*pi)**2 * sigmae/self.eVcm**2/ self.muXe**2 \
            * 1.0/(2.0*self.alphaEM) * k**3 * self.Fmed_electron(k)**2 \
            * atomic_factor2 * temp_eps2
    if withscreening:
        dR = dR/(temp_eps1**2 + temp_eps2**2)
    return dR


def dRdomega_electron(
    self, omega, sigmae=1e-38, kcut=0, withscreening=True, method="grid", vdist="halo",
    with_atomic_ff=False, alpha_D=None, m_eD_eV=None, m_pD_eV=None,
):
    """Returns differential rate for DM-electron scattering in 1/kg/yr/eV."""
    if kcut == 0:
        kcut = self.kmax

    scalar_input = np.isscalar(omega)
    omega = np.atleast_1d(omega)
    dRdomega = np.zeros_like(omega, dtype=float)

    for i in range(len(omega)):
        if method == "Lindhard":
            kmin_eps = np.sqrt(2*self.me * omega[i] + self.kF**2) - self.kF
            kmax_eps = np.sqrt(2*self.me * omega[i] + self.kF**2) + self.kF
        elif method == "grid":
            kmin_eps = 0.0
            kmax_eps = kcut
        else:
            raise ValueError("method must be 'grid' or 'Lindhard'")

        if vdist == "halo":
            kmin = max(kmin_eps, self.qmin(omega[i]))
            kmax = min(kmax_eps, self.qmax(omega[i]))
        elif vdist == "disk":
            kmin = max(kmin_eps, 0.0)
            kmax = kmax_eps
        else:
            raise ValueError("vdist must be 'halo' or 'disk'")

        if kmin >= kmax:
            continue
        if kmin == 0.0:
            kmin = 1e-12

        dRdomega[i] = self.eVtoInvYr * integrate.quad(
            lambda x: self.dRdomegadk_electron(
                omega[i], x, sigmae,
                withscreening=withscreening, method=method, vdist=vdist,
                with_atomic_ff=with_atomic_ff, alpha_D=alpha_D,
                m_eD_eV=m_eD_eV, m_pD_eV=m_pD_eV,
            ) / self.eVtoInvYr,
            kmin, kmax, limit=50,
        )[0]

    return dRdomega[0] if scalar_input else dRdomega


def R_electron(
    self, threshold=-1.0, Emax=-1.0, sigmae=1e-38, kcut=0,
    withscreening=True, method="grid", vdist="halo",
    with_atomic_ff=False, alpha_D=None, m_eD_eV=None, m_pD_eV=None,
):
    """Returns total number of events per kg-year."""
    if threshold < 0.0:
        if hasattr(self, "e0"):
            threshold = self.E_gap + self.e0
        else:
            threshold = np.max([2.0*self.E_gap, 1e-3])

    if vdist == "halo":
        kin_emax = 0.5*(self.vesc+self.veavg)**2*self.mX
    elif vdist == "disk":
        kin_emax = self.ommax
    else:
        raise ValueError("vdist must be 'halo' or 'disk'")

    if Emax < 1.0:
        Emax = np.min([self.ommax, kin_emax])
    else:
        Emax = np.min([self.ommax, kin_emax, Emax])

    if Emax <= threshold:
        return 0.0

    olist = np.linspace(threshold, Emax, 200)
    return np.trapz(
        self.dRdomega_electron(
            olist, sigmae=sigmae, kcut=kcut, withscreening=withscreening,
            method=method, vdist=vdist, with_atomic_ff=with_atomic_ff,
            alpha_D=alpha_D, m_eD_eV=m_eD_eV, m_pD_eV=m_pD_eV,
        ),
        x=olist,
    )

############################################################################################

def electron_yield(omega):
    """Number of ionization electrons for a given energy omega."""
    if hasattr(self, "e0") and hasattr(self, "E_gap"):
        return 1+np.floor((omega-self.E_gap)/self.e0)
    print("This function is not available for "+self.target)
    return 0.0


def dRdQ_electron(
    self, Q, sigmae=1e-38, kcut=0, withscreening=True, method="grid", vdist="halo",
    with_atomic_ff=False, alpha_D=None, m_eD_eV=None, m_pD_eV=None,
):
    """Differential rate in ionization-electron number, in 1/kg/yr."""
    assert hasattr(self, "e0") and hasattr(self, "E_gap"), \
        "This function is not available for "+self.target+" due to missing e0 and E_gap"

    Qlist = np.atleast_1d(Q)
    scalar_input = np.isscalar(Q)
    dRdQ = np.zeros_like(Qlist, dtype=float)
    for i in range(len(Qlist)):
        if Qlist[i] < 1.0:
            dRdQ[i] = 0.0
        else:
            olist = np.linspace(
                self.E_gap+(Qlist[i]-1.0)*self.e0,
                self.E_gap+Qlist[i]*self.e0,
                20,
            )
            dRdQ[i] = np.trapz(
                self.dRdomega_electron(
                    olist, sigmae=sigmae, kcut=kcut,
                    withscreening=withscreening, method=method, vdist=vdist,
                    with_atomic_ff=with_atomic_ff, alpha_D=alpha_D,
                    m_eD_eV=m_eD_eV, m_pD_eV=m_pD_eV,
                ),
                x=olist,
            )
    return dRdQ[0] if scalar_input else dRdQ
