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

COLUMNAS_NECESARIAS = [
    "AGLOMERADO",
    "ESTADO",
    "PONDERA"
]


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def nombre_archivo(anio):
    """
    Construye el nombre esperado del archivo individual
    correspondiente al tercer trimestre de cada año.
    Ejemplo:
    2016 -> usu_individual_T316.txt
    2025 -> usu_individual_T325.txt
    """
    sufijo = str(anio)[-2:]
    return f"usu_individual_T3{sufijo}.txt"


def validar_archivos():
    """
    Verifica que estén disponibles los diez archivos necesarios.
    """
    faltantes = []

    for anio in ANIOS:
        archivo = nombre_archivo(anio)

        if not os.path.exists(archivo):
            faltantes.append(archivo)

    if faltantes:
        print("\nERROR: Faltan los siguientes archivos:")
        for archivo in faltantes:
            print(f"- {archivo}")

        print("\nGuardalos en la misma carpeta que este script.")
        sys.exit(1)


def cargar_base(anio):
    """
    Carga solamente las columnas necesarias de la base individual.
    """
    archivo = nombre_archivo(anio)

    df = pd.read_csv(
        archivo,
        sep=";",
        decimal=",",
        usecols=COLUMNAS_NECESARIAS,
        low_memory=False
    )

    df.columns = df.columns.str.strip()

    for columna in COLUMNAS_NECESARIAS:
        df[columna] = pd.to_numeric(
            df[columna],
            errors="coerce"
        )

    return df


def calcular_indicadores(df, anio, codigo_aglomerado, nombre_aglomerado):
    """
    Calcula las tasas laborales utilizando PONDERA.

    ESTADO:
    1 = Ocupado
    2 = Desocupado
    3 = Inactivo
    4 = Menor de 10 años
    """

    df_aglomerado = df[
        df["AGLOMERADO"] == codigo_aglomerado
    ].copy()

    poblacion_total = df_aglomerado["PONDERA"].sum()

    poblacion_activa = df_aglomerado[
        df_aglomerado["ESTADO"].isin([1, 2])
    ]["PONDERA"].sum()

    poblacion_ocupada = df_aglomerado[
        df_aglomerado["ESTADO"] == 1
    ]["PONDERA"].sum()

    poblacion_desocupada = df_aglomerado[
        df_aglomerado["ESTADO"] == 2
    ]["PONDERA"].sum()

    tasa_actividad = (
        poblacion_activa / poblacion_total * 100
        if poblacion_total > 0
        else None
    )

    tasa_empleo = (
        poblacion_ocupada / poblacion_total * 100
        if poblacion_total > 0
        else None
    )

    tasa_desocupacion = (
        poblacion_desocupada / poblacion_activa * 100
        if poblacion_activa > 0
        else None
    )

    return {
        "Anio": anio,
        "Codigo_aglomerado": codigo_aglomerado,
        "Aglomerado": nombre_aglomerado,
        "Poblacion_total_ponderada": poblacion_total,
        "Poblacion_activa_ponderada": poblacion_activa,
        "Poblacion_ocupada_ponderada": poblacion_ocupada,
        "Poblacion_desocupada_ponderada": poblacion_desocupada,
        "Tasa_actividad": tasa_actividad,
        "Tasa_empleo": tasa_empleo,
        "Tasa_desocupacion": tasa_desocupacion
    }


def generar_grafico(df_resultados, columna, titulo, etiqueta_y, archivo_salida):
    """
    Genera un gráfico de líneas comparando ambos aglomerados.
    """

    plt.figure(figsize=(9, 5))

    for aglomerado in AGLOMERADOS.values():
        datos = df_resultados[
            df_resultados["Aglomerado"] == aglomerado
        ]

        plt.plot(
            datos["Anio"],
            datos[columna],
            marker="o",
            label=aglomerado
        )

    plt.title(titulo)
    plt.xlabel("Año")
    plt.ylabel(etiqueta_y)
    plt.xticks(ANIOS)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        archivo_salida,
        dpi=300
    )

    plt.close()


# ============================================================
# PROCESAMIENTO
# ============================================================

print("=" * 75)
print("SERIE HISTÓRICA DEL MERCADO LABORAL - EPH TERCER TRIMESTRE")
print("=" * 75)

validar_archivos()

resultados = []

for anio in ANIOS:
    archivo = nombre_archivo(anio)

    print(f"\nProcesando {archivo}...")

    df = cargar_base(anio)

    for codigo, nombre in AGLOMERADOS.items():
        indicadores = calcular_indicadores(
            df=df,
            anio=anio,
            codigo_aglomerado=codigo,
            nombre_aglomerado=nombre
        )

        resultados.append(indicadores)

df_resultados = pd.DataFrame(resultados)


# ============================================================
# REDONDEAR PARA PRESENTACIÓN
# ============================================================

columnas_tasas = [
    "Tasa_actividad",
    "Tasa_empleo",
    "Tasa_desocupacion"
]

for columna in columnas_tasas:
    df_resultados[columna] = df_resultados[columna].round(2)


# ============================================================
# MOSTRAR RESULTADOS
# ============================================================

tabla_presentacion = df_resultados[
    [
        "Anio",
        "Aglomerado",
        "Tasa_actividad",
        "Tasa_empleo",
        "Tasa_desocupacion"
    ]
].copy()

tabla_presentacion = tabla_presentacion.rename(
    columns={
        "Anio": "Año",
        "Tasa_actividad": "Actividad (%)",
        "Tasa_empleo": "Empleo (%)",
        "Tasa_desocupacion": "Desocupación (%)"
    }
)

print("\n" + "=" * 75)
print("RESULTADOS")
print("=" * 75)

print(
    tabla_presentacion.to_string(
        index=False
    )
)


# ============================================================
# EXPORTAR TABLAS
# ============================================================

df_resultados.to_csv(
    "serie_historica_mercado_laboral_completa.csv",
    index=False,
    encoding="utf-8-sig"
)

tabla_presentacion.to_csv(
    "tabla_mercado_laboral_para_informe.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# GENERAR GRÁFICOS
# ============================================================

generar_grafico(
    df_resultados=df_resultados,
    columna="Tasa_actividad",
    titulo="Evolución de la tasa de actividad (2016-2025)",
    etiqueta_y="Tasa de actividad (%)",
    archivo_salida="grafico_tasa_actividad.png"
)

generar_grafico(
    df_resultados=df_resultados,
    columna="Tasa_empleo",
    titulo="Evolución de la tasa de empleo (2016-2025)",
    etiqueta_y="Tasa de empleo (%)",
    archivo_salida="grafico_tasa_empleo.png"
)

generar_grafico(
    df_resultados=df_resultados,
    columna="Tasa_desocupacion",
    titulo="Evolución de la tasa de desocupación (2016-2025)",
    etiqueta_y="Tasa de desocupación (%)",
    archivo_salida="grafico_tasa_desocupacion.png"
)


# ============================================================
# FINALIZACIÓN
# ============================================================

print("\n" + "=" * 75)
print("ARCHIVOS GENERADOS")
print("=" * 75)

print("- serie_historica_mercado_laboral_completa.csv")
print("- tabla_mercado_laboral_para_informe.csv")
print("- grafico_tasa_actividad.png")
print("- grafico_tasa_empleo.png")
print("- grafico_tasa_desocupacion.png")

print("=" * 75)