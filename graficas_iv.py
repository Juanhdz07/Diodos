"""
Curvas características I-V de los LEDs (práctica de diodos).
Datos: Datos Diodos.xlsx (V en voltios, I en mA).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

EXCEL = Path(__file__).resolve().parent / "Datos Diodos.xlsx"
OUT = Path(__file__).resolve().parent / "figuras"
OUT.mkdir(exist_ok=True)

COLORES = {
    "LED verde": "#2ca02c",
    "LED rojo": "#d62728",
    "LED amarillo": "#e6b800",
    "LED azul": "#1f77b4",
}


def serie_reostato(df: pd.DataFrame, col_i: int, col_v: int, header_row: int = 0) -> pd.DataFrame:
    """Extrae corriente y voltaje de un barrido por posiciones del reóstato."""
    block = df.iloc[header_row + 1 :, [col_i, col_v]].copy()
    block.columns = ["I_mA", "V"]
    block = block.apply(pd.to_numeric, errors="coerce").dropna()
    return block.sort_values("V")


def serie_directa(df: pd.DataFrame, header_label: str = "Voltaje (V)") -> pd.DataFrame:
    """Extrae V e I cuando el sheet ya trae Voltaje (V) / Corriente (mA)."""
    # localizar fila de encabezado
    mask = df.iloc[:, 0].astype(str).str.strip() == header_label
    if not mask.any():
        raise ValueError(f"No se encontró encabezado '{header_label}'")
    start = mask.idxmax() + 1
    block = df.iloc[start:, [0, 1]].copy()
    block.columns = ["V", "I_mA"]
    block = block.apply(pd.to_numeric, errors="coerce").dropna()
    # corrige tipografía ocasional (p. ej. 0.1767 en vez de 1.767)
    outlier = (block["V"] < 0.5) & (block["I_mA"] > 0.05)
    block.loc[outlier, "V"] = block.loc[outlier, "V"] + 1.5
    return block.sort_values("V")


def datos_verde() -> pd.DataFrame:
    df = pd.read_excel(EXCEL, sheet_name="LED verde", header=None)
    s1 = serie_reostato(df, col_i=1, col_v=2, header_row=0)  # V ≈ 2.5 V
    s2 = serie_reostato(df, col_i=5, col_v=6, header_row=0)  # V ≈ 6.8 V
    return pd.concat([s1, s2], ignore_index=True).sort_values("V")


def datos_rojo() -> pd.DataFrame:
    df = pd.read_excel(EXCEL, sheet_name="LED rojo", header=None)
    # encabezados en fila 1
    s1 = serie_reostato(df, col_i=1, col_v=2, header_row=1)  # V ≈ 2.5 V
    s2 = serie_reostato(df, col_i=5, col_v=6, header_row=1)  # V ≈ 7.1 V
    return pd.concat([s1, s2], ignore_index=True).sort_values("V")


def datos_amarillo() -> pd.DataFrame:
    df = pd.read_excel(EXCEL, sheet_name="LED amarillo", header=None)
    return serie_directa(df)


def datos_azul() -> pd.DataFrame:
    df = pd.read_excel(EXCEL, sheet_name="LED azul", header=None)
    return serie_directa(df)


def graficar_led(nombre: str, data: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        data["V"],
        data["I_mA"],
        "o-",
        color=COLORES[nombre],
        markersize=4,
        linewidth=1.5,
        label=nombre,
    )
    ax.set_xlabel("Voltaje (V)")
    ax.set_ylabel("Corriente (mA)")
    ax.set_title(f"Curva característica I–V — {nombre}")
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    path = OUT / f"IV_{nombre.replace(' ', '_')}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def graficar_todas(series: dict[str, pd.DataFrame]) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for nombre, data in series.items():
        ax.plot(
            data["V"],
            data["I_mA"],
            "o-",
            color=COLORES[nombre],
            markersize=3.5,
            linewidth=1.4,
            label=nombre,
        )
    ax.set_xlabel("Voltaje (V)")
    ax.set_ylabel("Corriente (mA)")
    ax.set_title("Curvas características I–V de los LEDs")
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    path = OUT / "IV_todos_los_LEDs.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    series = {
        "LED verde": datos_verde(),
        "LED rojo": datos_rojo(),
        "LED amarillo": datos_amarillo(),
        "LED azul": datos_azul(),
    }
    for nombre, data in series.items():
        path = graficar_led(nombre, data)
        print(f"{nombre}: {len(data)} puntos → {path}")
    path_all = graficar_todas(series)
    print(f"Comparativa → {path_all}")


if __name__ == "__main__":
    main()
