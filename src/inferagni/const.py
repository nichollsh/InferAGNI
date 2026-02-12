from __future__ import annotations

import matplotlib.pyplot as plt
from cmcrameri import cm


unit_Rjup = 69.911e3 # m
unit_Rear = 6.371e3  # m
unit_Mjup = 1.898e27 # kg
unit_Mear = 5.972e24 # kg

Gconst = 6.6743e-11

R_earth = 6.371e6
M_earth = 5.972e24
bar = 1e5 # Pa
S_earth =1361.0


# scale factor, log axis true/false, unit string, colormap
units = {
    # input vars
    "frac_core"     : [1.0,       False  ,r"Core frac, $f_c$",  plt.get_cmap("viridis")],
    "frac_atm"      : [1.0,       True   ,r"Atmos frac, $f_a$",  cm.glasgow],
    "mass_tot"      : [1.0,       True   ,r"Planet mass, $M_p$ [$M_\oplus$]",  cm.batlow],
    "instellation"  : [1.0,       True   ,r"Instellation [$S_\oplus$]", cm.batlow],

    "metal_C"       : [1.0,       True   ,"C/H mol", cm.hawaii_r],
    "metal_S"       : [1.0,       True   ,"S/H mol", cm.hawaii_r],
    "metal_O"       : [1.0,       True   ,"O/H mol", cm.hawaii_r],

    "logZ"          : [1.0,       False   ,r"log$_{10}(Z_a)$", cm.hawaii_r],
    "logCO"         : [1.0,       False   ,r"log$_{10}(\text{C/O})$", cm.managua_r],

    "Teff"          : [1.0,       False  ,r"Star $T_\text{eff}$ [K]", cm.batlow],

    "flux_int"      : [1.0,       False  ,r"$F_\text{net}$ [W/m$^2$]", cm.tokyo],

    # output vars
    "p_surf"        : [1e-5,      True   ,r"Surface pressure [bar]",cm.glasgow], # output, Pa -> bar
    "t_surf"        : [1,         False  ,r"Surface temperature [K]",cm.glasgow],
    "r_surf"        : [1/R_earth, False  ,r"Interior radius, $R_s$ [$R_\oplus$]", cm.batlow],
    "μ_surf"        : [1e3,       False  ,r"Surface MMW [g/mol]", cm.hawaii_r],
    "g_surf"        : [1,         False  ,r"$g_s$ [m$^2$/s]", cm.devon_r],

    "r_bound"       : [1/R_earth, False  ,r"$R_b$ [$R_\oplus$]", cm.batlow],

    "r_phot"        : [1/R_earth, True  , r"Planet radius, $R_p$ [$R_\oplus$]",  cm.batlow],
    "μ_phot"        : [1e3,       False  ,r"$\mu_p$ [g/mol]",    cm.hawaii_r],
    "t_phot"        : [1,         False  ,r"Photo. temperature, $T_p$ [K]",         cm.glasgow],
    "g_phot"        : [1,         False  ,r"Photo. gravity, $g_p$ [m$^2$/s]",    cm.devon_r],

    "Kzz_max"       : [1e4,       True   ,r"Maximum $K_{zz}$ [cm$^2$/s]",cm.acton],
    "conv_pbot"     : [1e-5,      True   ,r"Convection $p_c^b$ [bar]",      cm.acton],
    "conv_ptop"     : [1e-5,      True   ,r"Convection $p_c^t$ [bar]",      cm.acton],

    "flux_loss"     : [1.0,       False  ,r"$F_\text{loss}$ [W/m$^2$]", cm.roma],
    "succ"          : [1,         False  ,"Success", cm.roma], # succ=1, fail=-1
    "worker"        : [1,         False  ,"Worker", cm.nuuk],

    "dom_gas"       : [1,         False, "_dominant_gas", None]
}
