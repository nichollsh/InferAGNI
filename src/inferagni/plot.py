from __future__ import annotations

from copy import deepcopy

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from cmcrameri import cm

from .grid import Grid
from .planets import exoplanets, solarsys
from .util import GridVar, R_earth, getclose, varprops

DPI = 120


def latexify(s):
    latex = ""
    for c in s:
        if c.isnumeric():
            latex += f"$_{c}$"
        else:
            latex += c
    return latex


gas_colors = {
    # H rich
    "H2": "#008C01",
    "H2O": "#027FB1",
    "CH4": "#C720DD",
    # C rich
    "CO": "#D1AC02",
    # S rich
    "H2S": "#640deb",
    "S2": "#FF8FA1",
    "SO2": "#00008B",
    # O rich
    "OH": "#00ffd4",
    "CO2": "#D24901",
    "O2": "#00ff00",
    # N rich
    "NO2": "#eb0076",
    "H2SO4": "#46eba4",
    "N2": "#870036",
    "NH3": "#675200",
}

gases = list(gas_colors.keys())
for gas in gases:
    latex = latexify(gas)
    varprops[f"vmr_{gas}"] = GridVar(1, True, f"{latex} VMR", cm.glasgow)

# ZENG+19 MASS RADIUS CURVES
#    key = iron mass fraction
zeng = {}
# pure silicate
zeng_sil = [
    [0.0094, 0.2533],
    [0.0133, 0.2831],
    [0.0188, 0.3167],
    [0.0263, 0.3533],
    [0.0363, 0.3926],
    [0.0502, 0.436],
    [0.0668, 0.4778],
    [0.0883, 0.5214],
    [0.1151, 0.5659],
    [0.1453, 0.6064],
    [0.1882, 0.6542],
    [0.2467, 0.7083],
    [0.3252, 0.7682],
    [0.4289, 0.8332],
    [0.5643, 0.9031],
    [0.7393, 0.9772],
    [0.9634, 1.0552],
    [1.2479, 1.1368],
    [1.6061, 1.2214],
    [2.0541, 1.3089],
    [2.6105, 1.3987],
    [3.2973, 1.4906],
    [4.1404, 1.5844],
    [5.1702, 1.6797],
    [6.4220, 1.7763],
    [7.9376, 1.8739],
    [9.7656, 1.9725],
    [11.9628, 2.0718],
    [14.5957, 2.1716],
    [17.7419, 2.2719],
    [21.4589, 2.3716],
    [25.6899, 2.4665],
    [30.5051, 2.5564],
    [35.9972, 2.6420],
    [42.2655, 2.7237],
    [49.4164, 2.8016],
    [57.5637, 2.8759],
    [66.8285, 2.9467],
    [77.339, 3.0138],
    [89.2314, 3.0774],
    [102.6513, 3.1372],
    [117.7562, 3.1933],
    [134.7189, 3.2455],
    [153.7329, 3.2941],
    [175.0178, 3.3389],
    [198.8262, 3.3801],
    [225.4501, 3.4179],
    [255.2280, 3.4524],
    [288.5492, 3.484],
    [325.8570, 3.5128],
    [367.6497, 3.539],
    [414.4773, 3.5629],
    [466.9373, 3.5847],
    [525.6648, 3.6044],
    [591.3243, 3.6222],
    [664.5985, 3.6379],
]
zeng[0.0] = np.array(zeng_sil).T
# Earth like (32.5% Fe + silicate)
zeng_mix = [
    [0.003, 0.1648],
    [0.0042, 0.1831],
    [0.0059, 0.2036],
    [0.0082, 0.2267],
    [0.0114, 0.2524],
    [0.0159, 0.281],
    [0.0221, 0.3128],
    [0.0306, 0.3476],
    [0.042, 0.3854],
    [0.0575, 0.4265],
    [0.0779, 0.4697],
    [0.1046, 0.515],
    [0.1393, 0.5625],
    [0.1831, 0.61],
    [0.2402, 0.6608],
    [0.3142, 0.7150],
    [0.4093, 0.7725],
    [0.5304, 0.8330],
    [0.6835, 0.8964],
    [0.8756, 0.9625],
    [1.115, 1.0309],
    [1.4114, 1.1015],
    [1.7763, 1.1741],
    [2.2233, 1.2485],
    [2.7682, 1.3245],
    [3.4297, 1.4019],
    [4.2296, 1.4806],
    [5.1932, 1.5604],
    [6.3505, 1.6412],
    [7.7363, 1.7228],
    [9.3912, 1.8052],
    [11.3628, 1.8883],
    [13.7066, 1.9719],
    [16.4870, 2.0559],
    [19.7797, 2.1404],
    [23.6585, 2.2246],
    [28.152, 2.3063],
    [33.3138, 2.3848],
    [39.2487, 2.4602],
    [46.0693, 2.5325],
    [53.8965, 2.6019],
    [62.8692, 2.6683],
    [73.1339, 2.7319],
    [84.8337, 2.7924],
    [98.1197, 2.8497],
    [113.1545, 2.9034],
    [130.1162, 2.9536],
    [149.2054, 3.0002],
    [170.6534, 3.0431],
]
zeng[0.325] = np.array(zeng_mix).T
# pure iron
zeng_irn = [
    [0.0021, 0.1175],
    [0.0029, 0.1314],
    [0.0041, 0.1469],
    [0.0057, 0.1641],
    [0.008, 0.1832],
    [0.0111, 0.2041],
    [0.0154, 0.2272],
    [0.0213, 0.2525],
    [0.0293, 0.2801],
    [0.0401, 0.31],
    [0.0546, 0.3422],
    [0.0739, 0.3768],
    [0.0993, 0.4138],
    [0.1326, 0.4529],
    [0.1758, 0.4943],
    [0.2313, 0.5376],
    [0.3023, 0.5828],
    [0.3921, 0.6297],
    [0.5051, 0.6781],
    [0.6462, 0.7279],
    [0.8212, 0.7789],
    [1.0371, 0.831],
    [1.3018, 0.884],
    [1.625, 0.9377],
    [2.0176, 0.9922],
    [2.4927, 1.0471],
    [3.0655, 1.1025],
    [3.7538, 1.1582],
    [4.5782, 1.2142],
    [5.5631, 1.2705],
    [6.7367, 1.3269],
    [8.1322, 1.3834],
    [9.7879, 1.4400],
    [11.7489, 1.4966],
    [14.0676, 1.5532],
    [16.805, 1.6099],
    [20.0062, 1.6660],
    [23.7165, 1.7209],
    [28.0181, 1.7747],
    [33.0006, 1.8275],
    [38.7619, 1.8791],
    [45.4074, 1.9294],
    [53.0498, 1.9785],
    [61.8097, 2.0260],
    [71.8161, 2.0719],
    [83.2082, 2.116],
    [96.1382, 2.1581],
    [110.7759, 2.1982],
    [127.3145, 2.2361],
]
zeng[1.0] = np.array(zeng_irn).T


left_align = ["T1", "L98-59"]
mass_ticks = [0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7, 8.0, 10.0]
radius_ticks = [0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0]


def massrad_2d(
    gr: Grid,
    key1=None,
    key2=None,
    controls=None,
    show_isolines=True,
    show_scatter=True,
    show_controls=True,
    echo_gridpoints=False,
    vmr_min=1e-4,
    show=True,
    save="massrad2d.pdf",
):
    # copy dataframe
    subdata = deepcopy(gr.data)

    # crop to successful runs (succ==1)
    subdata = subdata[subdata["succ"] > 0.0]

    # crop to bound atmospheres
    subdata = subdata[(subdata["r_bound"] < 0.0) | (subdata["r_bound"] > subdata["r_phot"])]

    print("Plot mass-radius 2D plane")

    if key1 and key1 not in subdata.keys():
        raise KeyError(f"Z-value {key1} not found in dataframe")
    else:
        print(f"    Zkey 1 (colour): {key1}")

    if key2 and key2 not in subdata.keys():
        raise KeyError(f"Z-value {key2} not found in dataframe")
    else:
        print(f"    Zkey 2 (alpha):  {key2}")

    # crop data by control variables
    print("    Control variables")
    for c in controls.keys():
        if c == key1:
            raise KeyError(f"Z-value {key1} cannot also be a control variable")
        if c == key2:
            raise KeyError(f"Z-value {key2} cannot also be a control variable")
        if c in subdata.columns:
            closest = getclose(subdata[c].values, controls[c])
            if not np.isclose(closest, controls[c]):
                print(f"      Filter by {c:12s} = {closest}, adjusted from {controls[c]}")
                controls[c] = closest
            else:
                print(f"      Filter by {c:12s} = {controls[c]}")
            subdata = subdata[np.isclose(subdata[c], controls[c])]
        if len(subdata) < 1:
            print("No data remaining! \n")
            return False

    print(f"    Number of grid points: {len(subdata)}")

    if key2:
        if len(controls) < len(gr.input_keys) - 2:
            show_isolines = False
            print("    Too few control variables to show isolines; showing scatter points")
    else:
        if len(controls) < len(gr.input_keys) - 2:
            show_isolines = False
            print("    Too few control variables to show isolines; showing scatter points")

    # create figure object
    figscale = 1.2
    fig, ax = plt.subplots(
        1, 1, figsize=(5 * figscale, 4 * figscale), dpi=DPI, num="Mass-Radius 2D"
    )

    # configure...
    xlim = [mass_ticks[0], mass_ticks[-1]]
    ylim = [radius_ticks[0], radius_ticks[-1]]
    s = 16
    sn = 35
    ec = "grey"
    lw = 0.8
    sm2_min = 0.3
    sm2_max = 1.0
    text_pe = [pe.Stroke(linewidth=0.9, foreground="black"), pe.Normal()]
    text_fs = 9
    planet_legend = False

    # choose cmap
    cmap = varprops[key1].cmap

    # colorbar map, for key1
    zunit = varprops[key1].scale
    vmin, vmax = np.amin(subdata[key1]) * zunit, np.amax(subdata[key1]) * zunit
    if "vmr_" in key1:
        vmin = max(vmin, vmr_min)
    print(f"    Colorbar shows '{key1}': vmin {vmin} - vmax {vmax}")
    num_z = len(np.unique(subdata[key1]))
    if num_z < 2:
        print(f"    Warning: only 1 unique value for {key1} \n")
        return False
    if varprops[key1].log:  # log scale
        nm1 = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        nm1 = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm1 = mpl.cm.ScalarMappable(norm=nm1, cmap=cmap)

    # opacity map, for key2
    if key2:
        v2_uni = np.unique(subdata[key2].values)
        if varprops[key2].log:
            v2_uni = np.log10(v2_uni)
        if len(v2_uni) < 2:
            print(f"    Warning: only 1 unique value for {key2} \n")
            return False

        def sm2(_v):
            _vn = (_v - np.amin(v2_uni)) / (np.amax(v2_uni) - np.amin(v2_uni))
            return _vn * (sm2_max - sm2_min) + sm2_min

    # echo
    if echo_gridpoints:
        print("    Matching grid points:")
        for row in subdata.iterrows():
            row = row[1]
            i = int(row["index"])
            x = row["mass_tot"]
            y = row["r_phot"] / R_earth
            print(f"     index={i:7d}:  m={x:.2f}    r={y:.2f}")
            if (xlim[0] <= x <= xlim[1]) and (ylim[0] <= y <= ylim[1]):
                ax.text(x, y, str(i), fontsize=8, color="red")

    # Grid results
    if show_scatter:
        c = np.array([sm1.to_rgba(z) for z in subdata[key1] * zunit])

        rast = bool(len(c) > 500)

        # bound atmospheres
        # bnd = np.array(() | (subdata['r_bound'] > subdata['r_phot']), dtype=bool)
        bnd = np.array(subdata["r_bound"] < 0.0)
        ax.scatter(
            subdata["mass_tot"][bnd],
            subdata["r_phot"][bnd] / R_earth,
            c=c[bnd],
            rasterized=rast,
            zorder=10,
            alpha=0.7,
            edgecolors=ec,
            label="Models",
            s=s,
            marker="o",
        )

        # partially unbound
        ax.scatter(
            subdata["mass_tot"][~bnd],
            subdata["r_phot"][~bnd] / R_earth,
            c=c[~bnd],
            rasterized=rast,
            zorder=10,
            alpha=0.7,
            edgecolors=ec,
            s=s,
            marker="X",
        )

    # Plot grid as lines of constant key
    isolines = np.unique(subdata[key1])
    if show_isolines:
        ax.plot([], [], label="Isolines", lw=lw, color="grey")
        # for each key1
        for u in isolines:
            df = subdata[subdata[key1] == u]  # get rows with key= this value
            u1 = u * zunit
            col = sm1.to_rgba(u1)

            # for each key2
            if key2:
                for u2 in np.unique(df[key2]):
                    df2 = df[df[key2] == u2]
                    xx = np.array(df2["mass_tot"].values)
                    yy = np.array(df2["r_phot"].values) / R_earth

                    mask_srt = np.argsort(xx)  # sort them in ascending mass order
                    if varprops[key2].log:
                        alp = sm2(np.log10(u2))
                    else:
                        alp = sm2(u2)
                    ax.plot(xx[mask_srt], yy[mask_srt], zorder=300, color=col, alpha=alp, lw=lw)
            else:
                alp = 0.7
                xx = np.array(df["mass_tot"].values)
                yy = np.array(df["r_phot"].values) / R_earth
                mask_srt = np.argsort(xx)  # sort them in ascending mass order
                ax.plot(xx[mask_srt], yy[mask_srt], zorder=300, color=col, alpha=alp, lw=lw)

    # Zeng2019 curves without atmospheres
    col = "blue"
    ax.plot([], [], ls="dashed", color=col, label=r"Zeng airless", lw=lw)
    for k, (x, y) in zeng.items():
        ax.plot(x, y, zorder=7, ls="dashed", color=col, lw=lw)
        # x0 = xlim[1] * 1.01
        # y0 = y[np.argmin(np.abs(x - x0))]
        # ax.text(x0, y0, k, ha='left', va='center', color=col, fontsize=text_fs)

    # Exoplanets
    ax.scatter(
        exoplanets.mass().value,
        exoplanets.radius().value,
        s=s / 2,
        label="Exoplanets",
        color="gray",
        edgecolors="none",
        alpha=0.5,
        zorder=0,
        rasterized=True,
    )

    # Named exoplanets
    exo_named = [
        "TRAPPIST-1",
        "L 98-59",
        "TOI-561",
        "K2-18",
        "GJ 1214",
        "55 Cnc",
        "TOI-270",
        "GJ 9827",
        "LP 791-18",
    ]
    exo_colors = [mpl.colormaps.get_cmap("tab10")(x) for x in np.linspace(0, 1, len(exo_named))]

    for j, s in enumerate(exo_named):
        col = exo_colors[j]
        system = exoplanets[s]

        x = system.mass().value
        y = system.radius().value

        for i in range(len(x)):
            if i == 0:
                lbl = s
            else:
                lbl = None
            if not planet_legend:
                lbl = None

            pl = str(system.name()[i])
            pl = pl.replace("TRAPPIST-1", "T1")
            pl = pl.replace("Cnc A", "Cnc")
            pl = pl.replace(" ", "")

            if not (xlim[0] < x[i] < xlim[1]) or not (ylim[0] < y[i] < ylim[1]):
                continue

            a = ax.scatter(
                x[i],
                y[i],
                s=sn,
                label=lbl,
                alpha=0.8,
                zorder=901,
                color=col,
                ec="k",
                marker="D",
            )
            if not planet_legend:
                if np.any([pl.startswith(la) for la in left_align]):
                    ha = "left"
                else:
                    ha = "right"
                a = ax.text(
                    x[i],
                    y[i],
                    pl,
                    color=col,
                    fontsize=text_fs,
                    ha=ha,
                    va="bottom",
                    zorder=902,
                )
                a.set_path_effects(text_pe)

    # Solar system bodies
    ss_colors = {
        "Mercury": "grey",
        "Venus": "tab:orange",
        "Earth": "tab:blue",
        "Mars": "tab:red",
    }
    for p in solarsys:
        x = p.mass().value[0]
        y = p.radius().value[0]
        if not (xlim[0] < x < xlim[1]) or not (ylim[0] < y < ylim[1]):
            continue

        v = p.name()[0]
        c = getattr(ss_colors, v, None)
        ax.scatter(
            x,
            y,
            label=v if planet_legend else None,
            zorder=902,
            s=sn,
            marker="D",
            edgecolors="k",
            color=c,
        )
        if not planet_legend:
            a = ax.text(
                x,
                y,
                v,
                color=c,
                fontsize=text_fs,
                ha="right",
                va="bottom",
                zorder=902,
            )
            a.set_path_effects(text_pe)

    ax.set_xlabel(varprops["mass_tot"].label)
    ax.set_xlim(xlim)

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%g"))
    ax.xaxis.set_ticks(mass_ticks)

    ax.set_ylabel(varprops["r_phot"].label)
    ax.set_ylim(ylim)

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.1f"))
    ax.yaxis.set_ticks(radius_ticks)

    # grid
    # ax.grid(alpha=0.3, zorder=-2)

    # colorbar
    if num_z > 2:
        cbkwargs = {}
        if (not varprops[key1].log) and not (len(isolines) > 30):
            cbkwargs = {"values": isolines * zunit, "ticks": isolines * zunit}
        cb = fig.colorbar(
            sm1,
            ax=ax,
            label=varprops[key1].label,
            location="top",
            pad=0.01,
            aspect=30,
            **cbkwargs,
        )

        if varprops[key1].log:
            cb.ax.set_xscale("log")
            cb.ax.xaxis.set_major_locator(mpl.ticker.LogLocator(numticks=9999))

    # controls annotation
    if show_controls:
        if key2:
            anno = r"$\bf{Opacity} \rightarrow$" + f"{varprops[key2].label} \n"
        else:
            anno = ""
        for k, v in controls.items():
            anno += f"{varprops[k].label}={v}\n"
        anno = anno[:-1]
        ax.text(
            0.02,
            0.98,
            anno,
            transform=ax.transAxes,
            fontsize=text_fs + 1,
            ha="left",
            va="top",
            zorder=9999,
            bbox=dict(facecolor="w", alpha=0.5, edgecolor="none"),
        )

    # legend
    leg = ax.legend(
        framealpha=1,
        loc="lower right",
        ncol=2,
        fontsize=text_fs,
        columnspacing=0.8,
        handletextpad=0.3,
        labelspacing=0.1,
    )
    leg.set_zorder(99999)

    # show and save
    fig.tight_layout()
    if save:
        print(f"    Saving plot to '{save}'")
        fig.savefig(save)
    if show:
        print("    Showing plot GUI")
        plt.show()
    return fig
