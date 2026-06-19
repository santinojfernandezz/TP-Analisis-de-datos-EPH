import os
import sys

import pandas as pd


# ============================================================
# CONFIGURACIÓN
# ============================================================

ARCHIVO_IPC_MENSUAL = "ipc_gba_mensual.csv"

ANIOS = list(range(2016, 2026))
MES_REFERENCIA = 9  # Septiembre: tercer trimestre

# IPC-GBA oficial de septiembre de 2016.
# Base diciembre de 2016 = 100.
IPC_BASE_SEPTIEMBRE_2016 = 95.0014


# ============================================================
# VALIDAR ARCHIVO
# ============================================================

if not os.path.exists(ARCHIVO_IPC_MENSUAL):
    print(f"ERROR: No se encontró '{ARCHIVO_IPC_MENSUAL}'.")
    print("Guardalo en la misma carpeta que este script.")
    sys.exit(1)


# ============================================================
# CARGAR IPC MENSUAL
# ============================================================

print("=" * 75)
print("DEFLACTORES IPC - PESOS CONSTANTES DE SEPTIEMBRE DE 2016")
print("=" * 75)

ipc = pd.read_csv(
    ARCHIVO_IPC_MENSUAL,
    low_memory=False
)

columnas_necesarias = [
    "indice_tiempo",
    "ipc_ng_nacional"
]

faltantes = [
    columna
    for columna in columnas_necesarias
    if columna not in ipc.columns
]

if faltantes:
    print("\nERROR: Faltan columnas necesarias:")
    print(faltantes)
    sys.exit(1)

ipc["indice_tiempo"] = pd.to_datetime(
    ipc["indice_tiempo"],
    errors="coerce"
)

ipc["ipc_ng_nacional"] = pd.to_numeric(
    ipc["ipc_ng_nacional"],
    errors="coerce"
)

ipc = ipc.dropna(
    subset=[
        "indice_tiempo",
        "ipc_ng_nacional"
    ]
)


# ============================================================
# FUNCIÓN AUXILIAR
# ============================================================

def obtener_ipc_nacional(anio, mes):
    fila = ipc[
        (ipc["indice_tiempo"].dt.year == anio)
        & (ipc["indice_tiempo"].dt.month == mes)
    ]

    if fila.empty:
        print(
            f"\nERROR: No se encontró el IPC nacional para "
            f"{mes:02d}/{anio}."
        )
        sys.exit(1)

    return float(
        fila.iloc[0]["ipc_ng_nacional"]
    )


# ============================================================
# GENERAR DEFLACTORES
# ============================================================

resultados = []

for anio in ANIOS:
    if anio == 2016:
        indice_ipc = IPC_BASE_SEPTIEMBRE_2016
        fuente = "IPC-GBA oficial"

    else:
        indice_ipc = obtener_ipc_nacional(
            anio=anio,
            mes=MES_REFERENCIA
        )

        fuente = "IPC nacional oficial"

    factor_acumulado = (
        indice_ipc
        / IPC_BASE_SEPTIEMBRE_2016
    )

    deflactor = (
        IPC_BASE_SEPTIEMBRE_2016
        / indice_ipc
    )

    resultados.append(
        {
            "Anio": anio,
            "Mes_referencia": "Septiembre",
            "Indice_IPC": indice_ipc,
            "Factor_acumulado_desde_sep_2016": factor_acumulado,
            "Deflactor_a_pesos_constantes_sep_2016": deflactor,
            "Fuente": fuente
        }
    )


# ============================================================
# EXPORTAR
# ============================================================

df_deflactores = pd.DataFrame(
    resultados
)

df_deflactores.to_csv(
    "deflactores_ipc_sep_2016_2025.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 75)
print("DEFLACTORES GENERADOS")
print("=" * 75)

print(
    df_deflactores.to_string(
        index=False,
        formatters={
            "Indice_IPC": "{:.4f}".format,
            "Factor_acumulado_desde_sep_2016": "{:.6f}".format,
            "Deflactor_a_pesos_constantes_sep_2016": "{:.8f}".format
        }
    )
)

print("\nArchivo generado:")
print("- deflactores_ipc_sep_2016_2025.csv")
print("=" * 75)