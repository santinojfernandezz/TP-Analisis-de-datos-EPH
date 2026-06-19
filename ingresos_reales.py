import os
import sys

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# CONFIGURACIÓN
# ============================================================

ANIOS = list(range(2016, 2026))

AGLOMERADOS = {
    33: "Partidos del GBA",
    4: "Gran Rosario"
}

ARCHIVO_DEFLACTORES = "deflactores_ipc_sep_2016_2025.csv"

COLUMNAS_NECESARIAS = [
    "AGLOMERADO",
    "ESTADO",
    "P21",
    "PONDIIO",
    "PONDERA"
]


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def nombre_archivo(anio):
    sufijo = str(anio)[-2:]
    return f"usu_individual_T3{sufijo}.txt"


def promedio_ponderado(valores, ponderadores):
    valores = pd.to_numeric(valores, errors="coerce")
    ponderadores = pd.to_numeric(ponderadores, errors="coerce")

    mascara = (
        valores.notna()
        & ponderadores.notna()
        & (ponderadores > 0)
    )

    if mascara.sum() == 0:
        return None

    return (valores[mascara] * ponderadores[mascara]).sum() / ponderadores[mascara].sum()


def validar_archivos():
    faltantes = []

    for anio in ANIOS:
        archivo = nombre_archivo(anio)
        if not os.path.exists(archivo):
            faltantes.append(archivo)

    if not os.path.exists(ARCHIVO_DEFLACTORES):
        faltantes.append(ARCHIVO_DEFLACTORES)

    if faltantes:
        print("\nERROR: Faltan archivos necesarios:")
        for archivo in faltantes:
            print(f"- {archivo}")

        print("\nGuardalos en la misma carpeta que este script.")
        sys.exit(1)


def cargar_deflactores():
    df_deflactores = pd.read_csv(ARCHIVO_DEFLACTORES)

    columnas_necesarias = [
        "Anio",
        "Deflactor_a_pesos_constantes_sep_2016"
    ]

    faltantes = [
        columna for columna in columnas_necesarias
        if columna not in df_deflactores.columns
    ]

    if faltantes:
        print("\nERROR: El archivo de deflactores no tiene estas columnas:")
        print(faltantes)
        sys.exit(1)

    df_deflactores["Anio"] = pd.to_numeric(
        df_deflactores["Anio"],
        errors="coerce"
    )

    df_deflactores["Deflactor_a_pesos_constantes_sep_2016"] = pd.to_numeric(
        df_deflactores["Deflactor_a_pesos_constantes_sep_2016"],
        errors="coerce"
    )

    return df_deflactores


def cargar_base(anio):
    archivo = nombre_archivo(anio)

    df = pd.read_csv(
        archivo,
        sep=";",
        decimal=",",
        usecols=lambda col: col.strip() in COLUMNAS_NECESARIAS,
        low_memory=False
    )

    df.columns = df.columns.str.strip()

    for columna in COLUMNAS_NECESARIAS:
        if columna not in df.columns:
            print(f"\nERROR: Falta la columna {columna} en {archivo}")
            sys.exit(1)

        df[columna] = pd.to_numeric(
            df[columna],
            errors="coerce"
        )

    return df


# ============================================================
# PROCESAMIENTO
# ============================================================

print("=" * 75)
print("INGRESOS REALES DE LA OCUPACIÓN PRINCIPAL - EPH TERCER TRIMESTRE")
print("=" * 75)

validar_archivos()

deflactores = cargar_deflactores()

resultados = []

for anio in ANIOS:
    print(f"\nProcesando {nombre_archivo(anio)}...")

    df = cargar_base(anio)

    fila_deflactor = deflactores[
        deflactores["Anio"] == anio
    ]

    if fila_deflactor.empty:
        print(f"\nERROR: No se encontró deflactor para el año {anio}.")
        sys.exit(1)

    deflactor = float(
        fila_deflactor.iloc[0]["Deflactor_a_pesos_constantes_sep_2016"]
    )

    for codigo, nombre in AGLOMERADOS.items():
        df_aglo = df[
            (df["AGLOMERADO"] == codigo)
            & (df["ESTADO"] == 1)
            & (df["P21"] > 0)
        ].copy()

        df_aglo["P21_REAL"] = df_aglo["P21"] * deflactor

        ingreso_real_medio = promedio_ponderado(
            df_aglo["P21_REAL"],
            df_aglo["PONDIIO"]
        )

        ingreso_nominal_medio = promedio_ponderado(
            df_aglo["P21"],
            df_aglo["PONDIIO"]
        )

        resultados.append({
            "Anio": anio,
            "Codigo_aglomerado": codigo,
            "Aglomerado": nombre,
            "Casos_ocupados_con_ingreso": len(df_aglo),
            "Deflactor": deflactor,
            "Ingreso_ocup_principal_nominal_medio": ingreso_nominal_medio,
            "Ingreso_ocup_principal_real_medio": ingreso_real_medio
        })


df_resultados = pd.DataFrame(resultados)

df_resultados["Ingreso_ocup_principal_nominal_medio"] = (
    df_resultados["Ingreso_ocup_principal_nominal_medio"].round(2)
)

df_resultados["Ingreso_ocup_principal_real_medio"] = (
    df_resultados["Ingreso_ocup_principal_real_medio"].round(2)
)


# ============================================================
# EXPORTAR TABLAS
# ============================================================

df_resultados.to_csv(
    "ingresos_reales_ocupacion_principal.csv",
    index=False,
    encoding="utf-8-sig"
)

tabla_presentacion = df_resultados[
    [
        "Anio",
        "Aglomerado",
        "Ingreso_ocup_principal_real_medio"
    ]
].copy()

tabla_presentacion = tabla_presentacion.rename(
    columns={
        "Anio": "Año",
        "Ingreso_ocup_principal_real_medio": "Ingreso Ocup. Principal Real Medio ($)"
    }
)

tabla_presentacion.to_csv(
    "tabla_ingresos_reales_para_informe.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# UNIR CON TABLA DE MERCADO LABORAL, SI EXISTE
# ============================================================

if os.path.exists("tabla_mercado_laboral_para_informe.csv"):
    mercado = pd.read_csv("tabla_mercado_laboral_para_informe.csv")

    tabla_final = mercado.merge(
        tabla_presentacion,
        on=["Año", "Aglomerado"],
        how="left"
    )

    tabla_final.to_csv(
        "tabla_final_mercado_laboral_ingresos.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nTabla final generada:")
    print("- tabla_final_mercado_laboral_ingresos.csv")

else:
    print("\nNo se encontró tabla_mercado_laboral_para_informe.csv.")
    print("Se generó solo la tabla de ingresos reales.")


# ============================================================
# GRÁFICO DE INGRESOS REALES
# ============================================================

plt.figure(figsize=(9, 5))

for aglomerado in AGLOMERADOS.values():
    datos = df_resultados[
        df_resultados["Aglomerado"] == aglomerado
    ]

    plt.plot(
        datos["Anio"],
        datos["Ingreso_ocup_principal_real_medio"],
        marker="o",
        label=aglomerado
    )

plt.title("Evolución del ingreso real de la ocupación principal (2016-2025)")
plt.xlabel("Año")
plt.ylabel("Ingreso real medio ($)")
plt.xticks(ANIOS)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    "grafico_ingreso_real_ocupacion_principal.png",
    dpi=300
)

plt.close()


# ============================================================
# MOSTRAR RESULTADOS
# ============================================================

print("\n" + "=" * 75)
print("RESULTADOS")
print("=" * 75)

print(
    tabla_presentacion.to_string(
        index=False
    )
)

print("\nArchivos generados:")
print("- ingresos_reales_ocupacion_principal.csv")
print("- tabla_ingresos_reales_para_informe.csv")
print("- grafico_ingreso_real_ocupacion_principal.png")

if os.path.exists("tabla_final_mercado_laboral_ingresos.csv"):
    print("- tabla_final_mercado_laboral_ingresos.csv")

print("=" * 75)