"""
Gráficas combinadas al estilo de la guía / ejemplos del informe:
  1) Curvas I–V de los cuatro LEDs
  2) Ajustes Shockley (log I) + residuales normalizados
  3) E vs f con ajuste experimental y recta teórica
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from analisis_laboratorio import (
    C_LIGHT,
    H_TEORICO,
    LAMBDA_NM,
    Q_E,
    SIGMA_I,
    SIGMA_V,
    datos_amarillo,
    datos_azul,
    datos_rojo,
    datos_verde,
    voltaje_encendido,
)

OUT = Path(__file__).resolve().parent / "figuras"
OUT.mkdir(exist_ok=True)

T = 290.0  # K (como en el ejemplo)
K_B = 1.380649e-23
VT = K_B * T / Q_E  # ≈ 0.0250 V

SERIES = {
    "Rojo": (datos_rojo, "#d62728"),
    "Amarillo": (datos_amarillo, "#ff7f0e"),
    "Verde": (datos_verde, "#2ca02c"),
    "Azul": (datos_azul, "#1f77b4"),
}

MARKERS = {"Rojo": "o", "Amarillo": "s", "Verde": "^", "Azul": "D"}


def shockley(v, i_s, n):
    return i_s * (np.exp(v / (n * VT)) - 1.0)


def fit_shockley(df: pd.DataFrame, i_lo=0.01, i_hi=1.5):
    sub = df[(df["I_mA"] >= i_lo) & (df["I_mA"] <= i_hi)].copy()
    if len(sub) < 4:
        sub = df[df["I_mA"] >= i_lo].nsmallest(12, "I_mA")
    if len(sub) < 4:
        return None
    v = sub["V"].to_numpy(float)
    i = sub["I_mA"].to_numpy(float)
    si = np.full_like(i, SIGMA_I)
    popt, pcov = curve_fit(
        shockley,
        v,
        i,
        p0=(1e-12, 2.0),
        sigma=np.maximum(si, 1e-12),
        absolute_sigma=True,
        bounds=([1e-20, 0.5], [1e-2, 8.0]),
        maxfev=30000,
    )
    i_fit = shockley(v, *popt)
    res_norm = (i - i_fit) / si
    dof = max(len(sub) - 2, 1)
    chi2_nu = float(np.sum(res_norm**2) / dof)
    return {
        "df": sub,
        "popt": popt,
        "perr": np.sqrt(np.diag(pcov)),
        "res_norm": res_norm,
        "chi2_nu": chi2_nu,
        "i_fit": i_fit,
    }


def grafica_iv_combinada(series_data: dict) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for nombre, (df, color) in series_data.items():
        ax.errorbar(
            df["V"],
            df["I_mA"],
            xerr=SIGMA_V,
            yerr=SIGMA_I,
            fmt=MARKERS[nombre] + "-",
            color=color,
            markersize=4,
            lw=1,
            capsize=2,
            elinewidth=0.7,
            label=nombre,
        )
    ax.set_xlabel(r"Voltaje en el LED, $V_D$ (V)")
    ax.set_ylabel(r"Corriente, $I$ (mA)")
    ax.set_title("Curvas características I–V de los cuatro LEDs")
    ax.legend(title="Color del LED")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    path = OUT / "combinada_IV_cuatro_LEDs.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def grafica_shockley_combinada(series_data: dict) -> Path:
    fits = {}
    for nombre, (df, color) in series_data.items():
        res = fit_shockley(df)
        if res is None:
            print(f"Sin ajuste Shockley para {nombre}")
            continue
        fits[nombre] = (res, color)
        n, sn = res["popt"][1], res["perr"][1]
        print(f"{nombre}: n={n:.3f}±{sn:.3f}, χ²ν={res['chi2_nu']:.2f}, N={len(res['df'])}")

    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(8.5, 7.5), sharex=True, gridspec_kw={"height_ratios": [2.6, 1], "hspace": 0.05}
    )

    for nombre, (res, color) in fits.items():
        df = res["df"]
        v = df["V"].to_numpy()
        i = df["I_mA"].to_numpy()
        v_line = np.linspace(v.min(), v.max(), 200)
        ax.errorbar(
            v,
            i,
            xerr=SIGMA_V,
            yerr=SIGMA_I,
            fmt=MARKERS[nombre],
            color=color,
            markersize=5,
            capsize=2,
            elinewidth=0.8,
            label=f"{nombre} datos",
            zorder=3,
        )
        ax.plot(
            v_line,
            shockley(v_line, *res["popt"]),
            "-",
            color=color,
            lw=1.8,
            label=f"{nombre} ajuste",
            zorder=2,
        )
        axr.axhline(0, color="0.5", lw=0.8)
        axr.plot(v, res["res_norm"], MARKERS[nombre], color=color, markersize=5)

    ax.set_yscale("log")
    ax.set_ylabel(r"Corriente, $I$ (mA)")
    ax.set_title(rf"Ajustes al modelo de Shockley para los cuatro LEDs ($T = {T:.0f}$ K)")
    ax.grid(True, which="both", alpha=0.35)
    ax.legend(fontsize=8, ncol=2, loc="upper left")

    texto = (
        r"$I = I_s\left[\exp\!\left(\dfrac{eV_D}{n k_B T}\right)-1\right]$"
        + "\n"
        + "\n".join(rf"$\chi_\nu^2$ {n}:" for n in fits)
    )
    ax.text(
        0.98,
        0.03,
        texto,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.6", alpha=0.95),
    )

    axr.set_xlabel(r"Voltaje en el LED, $V_D$ (V)")
    axr.set_ylabel("Residual\nnormalizado")
    axr.grid(True, alpha=0.35)
    axr.set_ylim(-4, 4)

    fig.tight_layout()
    path = OUT / "combinada_shockley_residuales.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def grafica_planck() -> Path:
    v0 = voltaje_encendido()
    # map names
    key = {
        "Rojo": "LED rojo",
        "Amarillo": "LED amarillo",
        "Verde": "LED verde",
        "Azul": "LED azul",
    }
    fs, es, sfs, ses, colors, names = [], [], [], [], [], []
    for short, full in key.items():
        lam_nm, sig_lam = LAMBDA_NM[full]
        lam = lam_nm * 1e-9
        f = C_LIGHT / lam
        sf = f * (sig_lam * 1e-9 / lam)
        e = Q_E * v0[full]
        se = Q_E * SIGMA_V
        fs.append(f)
        es.append(e)
        sfs.append(sf)
        ses.append(se)
        colors.append(SERIES[short][1])
        names.append(short)

    f = np.array(fs)
    e = np.array(es)
    se = np.array(ses)
    w = 1.0 / se**2
    h_exp = np.sum(w * f * e) / np.sum(w * f * f)
    sh = np.sqrt(1.0 / np.sum(w * f * f))

    f_line = np.linspace(4.0e14, 6.8e14, 200)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for i, name in enumerate(names):
        ax.errorbar(
            f[i],
            e[i],
            xerr=sfs[i],
            yerr=se[i],
            fmt="o",
            color=colors[i],
            markersize=8,
            capsize=3,
            label=name,
        )
    ax.plot(f_line, h_exp * f_line, "k-", lw=1.8, label=r"Ajuste $E = h_{\rm exp} f$")
    ax.plot(f_line, H_TEORICO * f_line, color="0.45", ls="--", lw=1.6, label=r"Teórico $E = h f$")
    ax.set_xlabel(r"Frecuencia $f$ (Hz)")
    ax.set_ylabel(r"Energía $E = qV_0$ (J)")
    ax.set_title(
        rf"$h_{{\rm exp}} = ({h_exp:.3e} \pm {sh:.3e})$ J·s"
    )
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    path = OUT / "combinada_planck_E_vs_f.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"h_exp = {h_exp:.6e} ± {sh:.6e}")
    return path


def main():
    series_data = {}
    for nombre, (loader, color) in SERIES.items():
        series_data[nombre] = (loader(), color)

    p1 = grafica_iv_combinada(series_data)
    p2 = grafica_shockley_combinada(series_data)
    p3 = grafica_planck()
    print(p1)
    print(p2)
    print(p3)


if __name__ == "__main__":
    main()
