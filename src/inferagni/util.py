from __future__ import annotations

from copy import deepcopy

import numpy as np
from cmcrameri import cm

# Physical constants (SI units)
Gconst = 6.6743e-11
M_jup = 1.898e27
R_jup = 69.911e3
R_earth = 6.371e6
M_earth = 5.972e24
bar = 1e5  # Pa
S_earth = 1361.0

# Other constants
print_sep_min: str = "-" * 50
print_sep_maj: str = "=" * 50


# Get closest value
def getclose(arr, val):
    iclose = np.argmin(np.abs(np.array(arr) - val))
    return float(arr[iclose])


class GridVar:
    def __init__(self, scale, log, label, cmap):
        self.scale = float(scale)
        self.log = bool(log)
        self.label = str(label)

        if cmap:
            self.cmap = deepcopy(cmap)
        else:
            self.cmap = deepcopy(cm.batlow)


# scale factor, log axis true/false, unit string, colormap
# fmt: off
varprops = {
    # input vars
    "frac_core"     : GridVar(1.0,       False  ,r"Core frac, $f_c$",  cm.batlow),
    "frac_atm"      : GridVar(1.0,       True   ,r"Atmos frac, $f_a$",  cm.glasgow),
    "mass_tot"      : GridVar(1.0,       False  ,r"Planet mass, $M_p$ [$M_\oplus$]",  cm.batlow),
    "instellation"  : GridVar(1.0,       True   ,r"Instellation [$S_\oplus$]", cm.batlow),

    "metal_C"       : GridVar(1.0,       True   ,"C/H mol", cm.hawaii_r),
    "metal_S"       : GridVar(1.0,       True   ,"S/H mol", cm.hawaii_r),
    "metal_O"       : GridVar(1.0,       True   ,"O/H mol", cm.hawaii_r),

    "logZ"          : GridVar(1.0,       False   ,r"log$_{10}(Z_a)$", cm.hawaii_r),
    "logCO"         : GridVar(1.0,       False   ,r"log$_{10}(\text{C/O})$", cm.managua_r),

    "Teff"          : GridVar(1.0,       False  ,r"Star $T_\text{eff}$ [K]", cm.batlow),

    "flux_int"      : GridVar(1.0,       False  ,r"$F_\text{net}$ [W/m$^2$]", cm.tokyo),

    # output vars
    "p_surf"        : GridVar(1e-5,      True   ,r"Surface pressure [bar]",cm.glasgow), # output, Pa -> bar
    "t_surf"        : GridVar(1,         False  ,r"Surface temperature [K]",cm.glasgow),
    "r_surf"        : GridVar(1/R_earth, False  ,r"Interior radius, $R_s$ [$R_\oplus$]", cm.batlow),
    "μ_surf"        : GridVar(1e3,       False  ,r"Surface MMW [g/mol]", cm.hawaii_r),
    "g_surf"        : GridVar(1,         False  ,r"$g_s$ [m$^2$/s]", cm.devon_r),

    "r_bound"       : GridVar(1/R_earth, False  ,r"$R_b$ [$R_\oplus$]", cm.batlow),

    "r_phot"        : GridVar(1/R_earth, True  , r"Planet radius, $R_p$ [$R_\oplus$]",  cm.batlow),
    "μ_phot"        : GridVar(1e3,       False  ,r"$\mu_p$ [g/mol]",    cm.hawaii_r),
    "t_phot"        : GridVar(1,         False  ,r"Photo. temperature, $T_p$ [K]",         cm.glasgow),
    "g_phot"        : GridVar(1,         False  ,r"Photo. gravity, $g_p$ [m$^2$/s]",    cm.devon_r),

    "Kzz_max"       : GridVar(1e4,       True   ,r"Maximum $K_{zz}$ [cm$^2$/s]",cm.acton),
    "conv_pbot"     : GridVar(1e-5,      True   ,r"Convection $p_c^b$ [bar]",      cm.acton),
    "conv_ptop"     : GridVar(1e-5,      True   ,r"Convection $p_c^t$ [bar]",      cm.acton),

    "flux_loss"     : GridVar(1.0,       False  ,r"$F_\text{loss}$ [W/m$^2$]", cm.roma),
    "succ"          : GridVar(1,         False  ,"Success", cm.roma), # succ=1, fail=-1
    "worker"        : GridVar(1,         False  ,"Worker", cm.nuuk),

    "dom_gas"       : GridVar(1,         False, "_dominant_gas", None)
}
# fmt: on


def undimen(arr, key):
    """Return the un-dimensionalised form of an array"""

    if key not in varprops.keys():
        raise KeyError(f"Unknown key {key}")

    if isinstance(arr, list):
        arr = np.array(arr)

    # if varprops[key].log:
    #     return np.log10(arr*varprops[key].scale)
    # else:
    return arr * varprops[key].scale


def redimen(arr, key):

    if key not in varprops.keys():
        raise KeyError(f"Unknown key {key}")

    if isinstance(arr, list):
        arr = np.array(arr)

    # if varprops[key].log:
    #     return np.pow(10, arr/varprops[key].scale)
    # else:
    return arr / varprops[key].scale
