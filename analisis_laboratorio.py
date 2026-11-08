"""
Análisis de la práctica de diodos (hoja de trabajo Uniandes).

Incluye:
  1) Curvas I–V con barras de error
  2) Ajuste Shockley rango bajo:  I = Is (exp(VD/(n VT)) - 1)   [0 → 0.2 mA]
  3) Ajuste exponencial rango alto: I = Is exp(VD/(n VT))       [0.2 → 1.8 mA]
  4) Residuales de cada ajuste
  5) Constante de Planck: E vs f, pendiente = h_exp ± σ_m

Incertidumbres instrumentales:
  σ_V = 0.001 V   (voltímetro)
  σ_I = 0.001 mA  (amperímetro)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

EXCEL = Path(__file__).resolve().parent / "Datos Diodos.xlsx"
OUT = Path(__file__).resolve().parent / "figuras"
OUT.mkdir(exist_ok=True)

SIGMA_V = 0.001  # V
SIGMA_I = 0.001  # mA
VT = 0.02585  # V a 300 K
Q_E = 1.602176634e-19  # C
C_LIGHT = 2.99792458e8  # m/s
H_TEORICO = 6.62607015e-34  # J·s

# Longitudes de onda nominales típicas de laboratorio (nm).
# Ajusta estos valores si la guía te da otros.
LAMBDA_NM = {
    "LED rojo": (650.0, 10.0),
    "LED amarillo": (590.0, 10.0),
    "LED verde": (565.0, 10.0),
    "LED azul": (470.0, 10.0),
}

COLORES = {
    "LED verde": "#2ca02c",
    "LED rojo": "#d62728",
    "LED amarillo": "#c9a000",
    "LED azul": "#1f77b4",
}


# ---------------------------------------------------------------------------
# Lectura de datos
# ---------------------------------------------------------------------------

def _clean_iv(v: np.ndarray, i: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"V": v, "I_mA": i})
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    # corrige tipografía 0.1767 → 1.767 en amarillo
    bad = (df["V"] < 0.5) & (df["I_mA"] > 0.05)
    df.loc[bad, "V"] = df.loc[bad, "V"] + 1.5
    df = df.sort_values("V").drop_duplicates(subset=["V", "I_mA"])
    df["sigma_V"] = SIGMA_V
    df["sigma_I"] = SIGMA_I
    return df.reset_index(drop=True)


def datos_amarillo() -> pd.DataFrame:
    raw = pd.read_excel(EXCEL, sheet_name="LED amarillo", header=None)
    start = raw.index[raw.iloc[:, 0].astype(str).str.strip() == "Voltaje (V)"][0] + 1
    return _clean_iv(raw.iloc[start:, 0].to_numpy(), raw.iloc[start:, 1].to_numpy())


def datos_azul() -> pd.DataFrame:
    raw = pd.read_excel(EXCEL, sheet_name="LED azul", header=None)
    start = raw.index[raw.iloc[:, 0].astype(str).str.strip() == "Voltaje (V)"][0] + 1
    return _clean_iv(raw.iloc[start:, 0].to_numpy(), raw.iloc[start:, 1].to_numpy())


def datos_verde() -> pd.DataFrame:
    raw = pd.read_excel(EXCEL, sheet_name="LED verde", header=None)
    s1 = raw.iloc[1:16, [1, 2]].to_numpy()
    s2 = raw.iloc[1:16, [5, 6]].to_numpy()
    stacked = np.vstack([s1, s2])
    return _clean_iv(stacked[:, 1], stacked[:, 0])


def datos_rojo() -> pd.DataFrame:
    raw = pd.read_excel(EXCEL, sheet_name="LED rojo", header=None)
    s1 = raw.iloc[2:17, [1, 2]].to_numpy()
    s2 = raw.iloc[2:17, [5, 6]].to_numpy()
    stacked = np.vstack([s1, s2])
    return _clean_iv(stacked[:, 1], stacked[:, 0])


def voltaje_encendido() -> dict[str, float]:
    """V0 anotado en el Excel / hoja de datos."""
    return {
        "LED rojo": 1.44,
        "LED amarillo": 1.69,
        "LED verde": 1.92,
        "LED azul": 2.26,
    }


# ---------------------------------------------------------------------------
# Modelos Shockley
# ---------------------------------------------------------------------------

def shockley_bajo(v: np.ndarray, i_s: float, n: float) -> np.ndarray:
    """I [mA] = Is [mA] * (exp(V/(n VT)) - 1)."""
    return i_s * (np.exp(v / (n * VT)) - 1.0)


def shockley_alto(v: np.ndarray, i_s: float, n: float) -> np.ndarray:
    """I [mA] = Is [mA] * exp(V/(n VT))."""
    return i_s * np.exp(v / (n * VT))


def ajustar(
    modelo,
    v: np.ndarray,
    i: np.ndarray,
    sigma_i: np.ndarray,
    p0: tuple[float, float],
    bounds: tuple,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    popt, pcov = curve_fit(
        modelo,
        v,
        i,
        p0=p0,
        sigma=np.maximum(sigma_i, 1e-12),
        absolute_sigma=True,
        bounds=bounds,
        maxfev=20000,
    )
    perr = np.sqrt(np.diag(pcov))
    i_fit = modelo(v, *popt)
    residuales = i - i_fit
    return popt, perr, residuales


def filtrar_rango(df: pd.DataFrame, i_min: float, i_max: float) -> pd.DataFrame:
    return df[(df["I_mA"] >= i_min) & (df["I_mA"] <= i_max)].copy()


# ---------------------------------------------------------------------------
# Gráficas
# ---------------------------------------------------------------------------

def graficar_iv(nombre: str, df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(
        df["V"],
        df["I_mA"],
        xerr=df["sigma_V"],
        yerr=df["sigma_I"],
        fmt="o",
        color=COLORES[nombre],
        markersize=3.5,
        capsize=2,
        elinewidth=0.8,
        label=nombre,
    )
    ax.set_xlabel(r"$V_D$ (V)")
    ax.set_ylabel(r"$I$ (mA)")
    ax.set_title(f"Curva característica I–V — {nombre}")
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    path = OUT / f"IV_{nombre.replace(' ', '_')}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def graficar_ajuste_y_residuales(
    nombre: str,
    etiqueta_rango: str,
    modelo,
    df: pd.DataFrame,
    popt: np.ndarray,
    perr: np.ndarray,
    residuales: np.ndarray,
) -> Path:
    i_s, n = popt
    si_s, sn = perr
    v = df["V"].to_numpy()
    i = df["I_mA"].to_numpy()
    v_fine = np.linspace(v.min(), v.max(), 300)
    i_fine = modelo(v_fine, *popt)

    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(7.5, 7.2), sharex=True, gridspec_kw={"height_ratios": [2.4, 1]}
    )
    ax.errorbar(
        v,
        i,
        xerr=df["sigma_V"],
        yerr=df["sigma_I"],
        fmt="o",
        color=COLORES[nombre],
        markersize=4,
        capsize=2,
        label="Datos",
    )
    ax.plot(v_fine, i_fine, "k-", lw=1.6, label="Ajuste")
    ax.set_ylabel(r"$I$ (mA)")
    ax.set_title(
        f"{nombre} — {etiqueta_rango}\n"
        rf"$I_s = ({i_s:.3e} \pm {si_s:.3e})$ mA,  $n = {n:.3f} \pm {sn:.3f}$"
    )
    ax.grid(True, alpha=0.35)
    ax.legend()

    axr.axhline(0, color="k", lw=1)
    axr.errorbar(
        v,
        residuales,
        yerr=df["sigma_I"],
        fmt="o",
        color=COLORES[nombre],
        markersize=4,
        capsize=2,
    )
    axr.set_xlabel(r"$V_D$ (V)")
    axr.set_ylabel(r"Residuales (mA)")
    axr.grid(True, alpha=0.35)
    fig.tight_layout()

    slug = etiqueta_rango.replace(" ", "_").replace("→", "a").replace(".", "p")
    path = OUT / f"ajuste_{nombre.replace(' ', '_')}_{slug}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def analizar_shockley(series: dict[str, pd.DataFrame]) -> pd.DataFrame:
    filas = []
    rangos = [
        ("bajo 0–0.2 mA", 0.0, 0.2, shockley_bajo, (1e-9, 1.5), (1e-20, 1e-2), (0.3, 8.0)),
        ("alto 0.2–1.8 mA", 0.2, 1.8, shockley_alto, (1e-9, 1.5), (1e-20, 1e-2), (0.3, 8.0)),
    ]

    for nombre, df in series.items():
        for etiqueta, i_lo, i_hi, modelo, p0, b_is, b_n in rangos:
            sub = filtrar_rango(df, i_lo, i_hi)
            if len(sub) < 4:
                print(f"  [omitido] {nombre} {etiqueta}: solo {len(sub)} puntos")
                continue
            try:
                popt, perr, res = ajustar(
                    modelo,
                    sub["V"].to_numpy(),
                    sub["I_mA"].to_numpy(),
                    sub["sigma_I"].to_numpy(),
                    p0=p0,
                    bounds=([b_is[0], b_n[0]], [b_is[1], b_n[1]]),
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  [falló] {nombre} {etiqueta}: {exc}")
                continue

            path = graficar_ajuste_y_residuales(
                nombre, etiqueta, modelo, sub, popt, perr, res
            )
            print(
                f"  {nombre} | {etiqueta}: n = {popt[1]:.3f} ± {perr[1]:.3f}, "
                f"Is = {popt[0]:.3e} ± {perr[0]:.3e} mA → {path.name}"
            )
            filas.append(
                {
                    "LED": nombre,
                    "rango": etiqueta,
                    "N": len(sub),
                    "Is_mA": popt[0],
                    "sigma_Is": perr[0],
                    "n": popt[1],
                    "sigma_n": perr[1],
                }
            )
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Constante de Planck
# ---------------------------------------------------------------------------

def tabla_planck() -> pd.DataFrame:
    v0 = voltaje_encendido()
    filas = []
    for color, (lam_nm, sig_lam) in LAMBDA_NM.items():
        lam = lam_nm * 1e-9
        sig_lam_m = sig_lam * 1e-9
        f = C_LIGHT / lam
        sig_f = f * (sig_lam_m / lam)
        vv = v0[color]
        e = Q_E * vv
        sig_e = Q_E * SIGMA_V
        filas.append(
            {
                "Color": color,
                "lambda_nm": lam_nm,
                "sigma_lambda_nm": sig_lam,
                "f_Hz": f,
                "sigma_f_Hz": sig_f,
                "V0_V": vv,
                "sigma_V0_V": SIGMA_V,
                "E_J": e,
                "sigma_E_J": sig_e,
            }
        )
    return pd.DataFrame(filas)


def regresion_lineal_ponderada(
    x: np.ndarray, y: np.ndarray, sy: np.ndarray
) -> tuple[float, float, float, float]:
    """y = m x + b con pesos 1/sy^2. Devuelve m, σm, b, σb."""
    w = 1.0 / np.maximum(sy, 1e-40) ** 2
    sw = np.sum(w)
    sx = np.sum(w * x)
    syw = np.sum(w * y)
    sxx = np.sum(w * x * x)
    sxy = np.sum(w * x * y)
    delta = sw * sxx - sx * sx
    m = (sw * sxy - sx * syw) / delta
    b = (sxx * syw - sx * sxy) / delta
    sm = np.sqrt(sw / delta)
    sb = np.sqrt(sxx / delta)
    return m, sm, b, sb


def graficar_planck(tabla: pd.DataFrame) -> tuple[Path, float, float]:
    f = tabla["f_Hz"].to_numpy()
    e = tabla["E_J"].to_numpy()
    sf = tabla["sigma_f_Hz"].to_numpy()
    se = tabla["sigma_E_J"].to_numpy()

    # Ajuste forzado por el origen teórico E = h f  (como en la hoja: E = hf)
    # Usamos ajuste con intercepto libre y también reportamos pendiente pura E=hf.
    m_free, sm_free, b, sb = regresion_lineal_ponderada(f, e, se)

    # Ajuste E = h f (b = 0), ponderado
    w = 1.0 / se**2
    h_exp = np.sum(w * f * e) / np.sum(w * f * f)
    # incertidumbre aproximada propagando solo σ_E
    sh = np.sqrt(1.0 / np.sum(w * f * f))

    f_line = np.linspace(f.min() * 0.98, f.max() * 1.02, 200)
    e_line = h_exp * f_line
    residuales = e - h_exp * f

    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(7.5, 7.2), sharex=True, gridspec_kw={"height_ratios": [2.4, 1]}
    )
    for _, row in tabla.iterrows():
        ax.errorbar(
            row["f_Hz"],
            row["E_J"],
            xerr=row["sigma_f_Hz"],
            yerr=row["sigma_E_J"],
            fmt="o",
            color=COLORES[row["Color"]],
            markersize=7,
            capsize=3,
            label=row["Color"],
        )
    ax.plot(
        f_line,
        e_line,
        "k-",
        lw=1.6,
        label=rf"$E = h_{{\rm exp}} f$" "\n"
        rf"$h_{{\rm exp}} = ({h_exp:.3e} \pm {sh:.3e})$ J·s",
    )
    ax.set_ylabel(r"$E$ (J)")
    ax.set_title("Constante de Planck — $E$ vs $f$")
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=8)

    axr.axhline(0, color="k", lw=1)
    for i, row in tabla.iterrows():
        axr.errorbar(
            row["f_Hz"],
            residuales[tabla.index.get_loc(i)],
            yerr=row["sigma_E_J"],
            fmt="o",
            color=COLORES[row["Color"]],
            markersize=7,
            capsize=3,
        )
    axr.set_xlabel(r"$f$ (Hz)")
    axr.set_ylabel(r"Residuales (J)")
    axr.grid(True, alpha=0.35)
    fig.tight_layout()
    path = OUT / "planck_E_vs_f.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)

    print(
        f"  Ajuste libre E = m f + b: m = {m_free:.3e} ± {sm_free:.3e}, "
        f"b = {b:.3e} ± {sb:.3e}"
    )
    return path, h_exp, sh


def main() -> None:
    series = {
        "LED rojo": datos_rojo(),
        "LED amarillo": datos_amarillo(),
        "LED verde": datos_verde(),
        "LED azul": datos_azul(),
    }

    print("=== 1) Curvas I–V ===")
    for nombre, df in series.items():
        path = graficar_iv(nombre, df)
        print(f"  {nombre}: {len(df)} puntos → {path.name}")

    print("\n=== 2–3) Ajustes Shockley + residuales ===")
    resumen = analizar_shockley(series)
    resumen_path = OUT / "resumen_coeficientes_n.csv"
    resumen.to_csv(resumen_path, index=False)
    print(f"  Resumen n → {resumen_path.name}")

    print("\n=== 4) Constante de Planck ===")
    tabla = tabla_planck()
    tabla_path = OUT / "tabla_planck.csv"
    tabla.to_csv(tabla_path, index=False)
    print(tabla.to_string(index=False))
    path, h_exp, sh = graficar_planck(tabla)
    err_pct = 100 * abs(h_exp - H_TEORICO) / H_TEORICO
    print(f"  h_exp = ({h_exp:.6e} ± {sh:.6e}) J·s")
    print(f"  h_teo = {H_TEORICO:.6e} J·s  (|error| = {err_pct:.2f} %)")
    print(f"  Gráfica → {path.name}")


if __name__ == "__main__":
    main()
