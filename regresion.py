import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURACIÓN
# ============================================================

ARCHIVO = "usu_individual_T325.txt"

# Códigos oficiales:
# 33 = Partidos del GBA
# 04 = Gran Rosario
AGLOMERADOS_SELECCIONADOS = [33, 4]

NOMBRES_AGLOMERADOS = {
    33: "Partidos del GBA",
    4: "Gran Rosario"
}

COLUMNAS_NECESARIAS = [
    "AGLOMERADO",
    "ESTADO",
    "CH04",
    "CH06",
    "NIVEL_ED",
    "CAT_OCUP",
    "PP3E_TOT",
    "PP04B_COD",
    "PP04D_COD",
    "P21",
    "PONDERA",
    "PONDIIO"
]


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def promedio_ponderado(valores, ponderadores):
    """
    Calcula la media ponderada descartando registros inválidos.
    """
    valores = pd.to_numeric(valores, errors="coerce")
    ponderadores = pd.to_numeric(ponderadores, errors="coerce")

    mascara = (
        valores.notna()
        & ponderadores.notna()
        & (ponderadores > 0)
    )

    if mascara.sum() == 0:
        return np.nan

    return np.average(
        valores[mascara],
        weights=ponderadores[mascara]
    )


def limpiar_categorica(serie):
    """
    Convierte una variable categórica a texto y reemplaza
    valores faltantes por una categoría explícita.
    """
    serie = serie.astype("string").str.strip()

    serie = serie.replace({
        "": pd.NA,
        "nan": pd.NA,
        "None": pd.NA,
        "<NA>": pd.NA
    })

    return serie.fillna("SIN_DATO")

def agrupar_codigo(serie, cantidad_digitos):
    """
    Agrupa códigos ocupacionales o de actividad económica.
    Los valores faltantes y códigos de no respuesta se guardan como SIN_DATO.
    """
    serie = pd.to_numeric(serie, errors="coerce")

    def transformar(valor):
        if pd.isna(valor):
            return "SIN_DATO"

        valor_entero = int(valor)
        texto = str(valor_entero)

        if set(texto) == {"9"}:
            return "SIN_DATO"

        texto = texto.zfill(cantidad_digitos)

        return texto[:cantidad_digitos]

    return serie.apply(transformar)

def crear_preprocesador(categoricas, numericas):
    """
    Crea el transformador para preparar variables categóricas
    y numéricas antes de ajustar una regresión.
    """
    return ColumnTransformer(
        transformers=[
            (
                "categoricas",
                Pipeline(
                    steps=[
                        (
                            "onehot",
                            OneHotEncoder(
                                drop="first",
                                handle_unknown="ignore"
                            )
                        )
                    ]
                ),
                categoricas
            ),
            (
                "numericas",
                Pipeline(
                    steps=[
                        (
                            "imputacion",
                            SimpleImputer(strategy="median")
                        )
                    ]
                ),
                numericas
            )
        ]
    )


def evaluar_modelo(
    nombre,
    modelo,
    X_train,
    X_test,
    y_train_original,
    y_test_original,
    pesos_train,
    pesos_test,
    usar_logaritmo=False
):
    """
    Ajusta y evalúa un modelo.
    Devuelve el modelo entrenado y las predicciones originales.
    """

    if usar_logaritmo:
        y_train_modelo = np.log1p(y_train_original)
    else:
        y_train_modelo = y_train_original

    modelo.fit(
        X_train,
        y_train_modelo,
        regresion__sample_weight=pesos_train
    )

    predicciones = modelo.predict(X_test)

    if usar_logaritmo:
        predicciones = np.expm1(predicciones)

    predicciones = np.clip(
        predicciones,
        a_min=0,
        a_max=None
    )

    r2 = r2_score(
        y_test_original,
        predicciones,
        sample_weight=pesos_test
    )

    mae = mean_absolute_error(
        y_test_original,
        predicciones,
        sample_weight=pesos_test
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test_original,
            predicciones,
            sample_weight=pesos_test
        )
    )

    print("\n" + "=" * 70)
    print(nombre)
    print("=" * 70)
    print(f"Casos de entrenamiento: {len(X_train)}")
    print(f"Casos de prueba: {len(X_test)}")
    print(f"R² ponderado: {r2:.4f}")
    print(f"MAE ponderado: ${mae:,.2f}")
    print(f"RMSE ponderado: ${rmse:,.2f}")

    return modelo, predicciones, r2, mae, rmse


# ============================================================
# VALIDAR ARCHIVO
# ============================================================

if not os.path.exists(ARCHIVO):
    print(f"ERROR: No se encontró el archivo '{ARCHIVO}'.")
    print("Guardalo en la misma carpeta que este script.")
    sys.exit(1)


# ============================================================
# CARGAR BASE
# ============================================================

print("=" * 70)
print("MODELO MEJORADO PARA IMPUTACIÓN DE INGRESOS - EPH 2025")
print("=" * 70)

df = pd.read_csv(
    ARCHIVO,
    sep=";",
    decimal=",",
    low_memory=False
)

df.columns = df.columns.str.strip()

print(f"Archivo cargado correctamente: {ARCHIVO}")
print(f"Cantidad de filas originales: {len(df)}")


# ============================================================
# VALIDAR COLUMNAS
# ============================================================

columnas_faltantes = [
    columna
    for columna in COLUMNAS_NECESARIAS
    if columna not in df.columns
]

if columnas_faltantes:
    print("\nERROR: Faltan columnas necesarias:")
    print(columnas_faltantes)
    sys.exit(1)


# ============================================================
# LIMPIEZA DE VARIABLES
# ============================================================

variables_numericas_originales = [
    "AGLOMERADO",
    "ESTADO",
    "CH06",
    "PP3E_TOT",
    "P21",
    "PONDERA",
    "PONDIIO"
]

variables_categoricas_originales = [
    "CH04",
    "NIVEL_ED",
    "CAT_OCUP"
]

for columna in variables_categoricas_originales:
    df[columna] = limpiar_categorica(df[columna])

df["PP04B_GRUPO"] = agrupar_codigo(
    df["PP04B_COD"],
    cantidad_digitos=2
)

df["PP04D_GRUPO"] = agrupar_codigo(
    df["PP04D_COD"],
    cantidad_digitos=1
)


# ============================================================
# FILTRAR OCUPADOS DE LOS DOS AGLOMERADOS
# ============================================================

df_filtrado = df[
    (df["AGLOMERADO"].isin(AGLOMERADOS_SELECCIONADOS))
    & (df["ESTADO"] == 1)
].copy()

# Se crea una variable cuadrática para admitir una relación no lineal
# entre edad e ingresos.
df_filtrado["CH06_CUADRADO"] = df_filtrado["CH06"] ** 2

# Aglomerado se trata como categoría dentro del modelo.
df_filtrado["AGLOMERADO_CAT"] = (
    df_filtrado["AGLOMERADO"]
    .astype("Int64")
    .astype("string")
)

# Excluir edades inválidas
df_filtrado = df_filtrado[
    (df_filtrado["CH06"].notna())
    & (df_filtrado["CH06"] >= 10)
    & (df_filtrado["CH06"] <= 100)
].copy()


# ============================================================
# SEPARAR RESPUESTAS VÁLIDAS Y NO RESPUESTA
# ============================================================

df_con_ingreso = df_filtrado[
    df_filtrado["P21"] > 0
].copy()

df_sin_ingreso = df_filtrado[
    df_filtrado["P21"] == -9
].copy()

cantidad_respondentes = len(df_con_ingreso)
cantidad_no_respuesta = len(df_sin_ingreso)
cantidad_total = cantidad_respondentes + cantidad_no_respuesta

porcentaje_no_respuesta = (
    cantidad_no_respuesta / cantidad_total * 100
    if cantidad_total > 0
    else 0
)

print("\n" + "=" * 70)
print("RESUMEN DE LOS DATOS")
print("=" * 70)
print(f"Ocupados considerados: {cantidad_total}")
print(f"Respondentes con ingreso mayor a 0: {cantidad_respondentes}")
print(f"No respuesta con P21 = -9: {cantidad_no_respuesta}")
print(f"Porcentaje de no respuesta: {porcentaje_no_respuesta:.2f}%")

if df_con_ingreso.empty:
    print("\nERROR: No existen datos suficientes para entrenar el modelo.")
    sys.exit(1)


# ============================================================
# VARIABLES DE LOS MODELOS
# ============================================================

# Modelo inicial para comparar con el anterior
VARIABLES_BASE = [
    "CH04",
    "CH06",
    "NIVEL_ED"
]

CATEGORICAS_BASE = [
    "CH04",
    "NIVEL_ED"
]

NUMERICAS_BASE = [
    "CH06"
]

# Modelo mejorado
VARIABLES_MEJORADAS = [
    "AGLOMERADO_CAT",
    "CH04",
    "CH06",
    "CH06_CUADRADO",
    "NIVEL_ED",
    "CAT_OCUP",
    "PP3E_TOT",
    "PP04B_GRUPO",
    "PP04D_GRUPO"
]

CATEGORICAS_MEJORADAS = [
    "AGLOMERADO_CAT",
    "CH04",
    "NIVEL_ED",
    "CAT_OCUP",
    "PP04B_GRUPO",
    "PP04D_GRUPO"
]

NUMERICAS_MEJORADAS = [
    "CH06",
    "CH06_CUADRADO",
    "PP3E_TOT"
]


# ============================================================
# DIVIDIR ENTRENAMIENTO Y PRUEBA
# ============================================================

indices_train, indices_test = train_test_split(
    df_con_ingreso.index,
    test_size=0.20,
    random_state=42
)

train = df_con_ingreso.loc[indices_train].copy()
test = df_con_ingreso.loc[indices_test].copy()

y_train = train["P21"]
y_test = test["P21"]

# PONDIIO es el ponderador específico del ingreso de la ocupación principal.
pesos_train = train["PONDIIO"].fillna(train["PONDERA"])
pesos_test = test["PONDIIO"].fillna(test["PONDERA"])

pesos_train = pesos_train.where(
    pesos_train > 0,
    train["PONDERA"]
)

pesos_test = pesos_test.where(
    pesos_test > 0,
    test["PONDERA"]
)


# ============================================================
# MODELO BASE
# ============================================================

preprocesador_base = crear_preprocesador(
    CATEGORICAS_BASE,
    NUMERICAS_BASE
)

modelo_base = Pipeline(
    steps=[
        ("preprocesamiento", preprocesador_base),
        ("regresion", LinearRegression())
    ]
)

modelo_base, pred_base, r2_base, mae_base, rmse_base = evaluar_modelo(
    nombre="MODELO BASE: SEXO, EDAD Y NIVEL EDUCATIVO",
    modelo=modelo_base,
    X_train=train[VARIABLES_BASE],
    X_test=test[VARIABLES_BASE],
    y_train_original=y_train,
    y_test_original=y_test,
    pesos_train=pesos_train,
    pesos_test=pesos_test,
    usar_logaritmo=False
)


# ============================================================
# MODELO MEJORADO CON LOGARITMO
# ============================================================

preprocesador_mejorado = crear_preprocesador(
    CATEGORICAS_MEJORADAS,
    NUMERICAS_MEJORADAS
)

modelo_mejorado = Pipeline(
    steps=[
        ("preprocesamiento", preprocesador_mejorado),
        ("regresion", LinearRegression())
    ]
)

modelo_mejorado, pred_mejoradas, r2_mejorado, mae_mejorado, rmse_mejorado = evaluar_modelo(
    nombre="MODELO MEJORADO: VARIABLES LABORALES Y LOGARITMO DEL INGRESO",
    modelo=modelo_mejorado,
    X_train=train[VARIABLES_MEJORADAS],
    X_test=test[VARIABLES_MEJORADAS],
    y_train_original=y_train,
    y_test_original=y_test,
    pesos_train=pesos_train,
    pesos_test=pesos_test,
    usar_logaritmo=True
)


# ============================================================
# COMPARACIÓN DE LOS MODELOS
# ============================================================

print("\n" + "=" * 70)
print("COMPARACIÓN DE LOS MODELOS")
print("=" * 70)

comparacion = pd.DataFrame({
    "Modelo": [
        "Base",
        "Mejorado"
    ],
    "R2_ponderado": [
        r2_base,
        r2_mejorado
    ],
    "MAE_ponderado": [
        mae_base,
        mae_mejorado
    ],
    "RMSE_ponderado": [
        rmse_base,
        rmse_mejorado
    ]
})

print(comparacion.to_string(index=False))

comparacion.to_csv(
    "comparacion_modelos.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# COEFICIENTES DEL MODELO MEJORADO
# ============================================================

print("\n" + "=" * 70)
print("COEFICIENTES DEL MODELO MEJORADO")
print("=" * 70)

nombres_variables = (
    modelo_mejorado
    .named_steps["preprocesamiento"]
    .get_feature_names_out()
)

coeficientes = (
    modelo_mejorado
    .named_steps["regresion"]
    .coef_
)

intercepto = (
    modelo_mejorado
    .named_steps["regresion"]
    .intercept_
)

tabla_coeficientes = pd.DataFrame({
    "Variable": nombres_variables,
    "Coeficiente": coeficientes
})

tabla_coeficientes["Variable"] = (
    tabla_coeficientes["Variable"]
    .str.replace("categoricas__", "", regex=False)
    .str.replace("numericas__", "", regex=False)
)

tabla_coeficientes = tabla_coeficientes.sort_values(
    by="Coeficiente",
    ascending=False
)

print(tabla_coeficientes.to_string(index=False))
print(f"\nIntercepto: {intercepto:,.6f}")

tabla_coeficientes.to_csv(
    "coeficientes_modelo_mejorado.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# GRÁFICOS DE DIAGNÓSTICO
# ============================================================
# ============================================================
# GRÁFICOS DE DIAGNÓSTICO
# ============================================================

# El modelo mejorado fue ajustado sobre log(P21 + 1).
# Por lo tanto, los residuos para evaluar los supuestos
# deben analizarse en escala logarítmica.

predicciones_log_test = modelo_mejorado.predict(
    test[VARIABLES_MEJORADAS]
)

y_test_log = np.log1p(y_test)

residuos_log = y_test_log - predicciones_log_test

# ------------------------------------------------------------
# Gráfico de residuos en escala logarítmica
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    predicciones_log_test,
    residuos_log,
    alpha=0.5
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Logaritmo del ingreso predicho")
plt.ylabel("Residuo en escala logarítmica")
plt.title("Residuos frente a valores predichos - escala logarítmica")

plt.tight_layout()

plt.savefig(
    "grafico_residuos_log.png",
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# Gráfico Q-Q en escala logarítmica
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

stats.probplot(
    residuos_log,
    dist="norm",
    plot=plt
)

plt.title("Gráfico Q-Q de residuos - escala logarítmica")

plt.tight_layout()

plt.savefig(
    "grafico_qqplot_log.png",
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# Gráfico adicional de errores expresados en pesos
# ------------------------------------------------------------

residuos_pesos = y_test - pred_mejoradas

plt.figure(figsize=(8, 5))

plt.scatter(
    pred_mejoradas,
    residuos_pesos,
    alpha=0.5
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Ingreso predicho")
plt.ylabel("Residuo expresado en pesos")
plt.title("Errores de predicción expresados en pesos")

plt.tight_layout()

plt.savefig(
    "grafico_errores_pesos.png",
    dpi=300
)

plt.close()

# ============================================================
# IMPUTACIÓN DE INGRESOS
# ============================================================

df_con_ingreso["P21_IMPUTADO"] = df_con_ingreso["P21"]
df_con_ingreso["ES_IMPUTADO"] = False

if len(df_sin_ingreso) > 0:
    predicciones_imputacion_log = modelo_mejorado.predict(
        df_sin_ingreso[VARIABLES_MEJORADAS]
    )

    predicciones_imputacion = np.expm1(
        predicciones_imputacion_log
    )

    predicciones_imputacion = np.clip(
        predicciones_imputacion,
        a_min=0,
        a_max=None
    )

    df_sin_ingreso["P21_IMPUTADO"] = predicciones_imputacion
    df_sin_ingreso["ES_IMPUTADO"] = True

else:
    print("\nNo se encontraron casos con P21 = -9.")

df_consolidado = pd.concat(
    [
        df_con_ingreso,
        df_sin_ingreso
    ],
    ignore_index=True
)


# ============================================================
# IMPACTO DE LA IMPUTACIÓN
# ============================================================

print("\n" + "=" * 70)
print("IMPACTO DE LA IMPUTACIÓN")
print("=" * 70)

resumen_impacto = []

for codigo in AGLOMERADOS_SELECCIONADOS:
    nombre = NOMBRES_AGLOMERADOS[codigo]

    respondentes = df_con_ingreso[
        df_con_ingreso["AGLOMERADO"] == codigo
    ]

    consolidado = df_consolidado[
        df_consolidado["AGLOMERADO"] == codigo
    ]

    promedio_original = promedio_ponderado(
        respondentes["P21"],
        respondentes["PONDIIO"]
    )

    promedio_imputado = promedio_ponderado(
        consolidado["P21_IMPUTADO"],
        consolidado["PONDERA"]
    )

    casos_imputados = consolidado[
        consolidado["ES_IMPUTADO"]
    ].shape[0]

    variacion_porcentual = (
        (promedio_imputado / promedio_original - 1) * 100
        if promedio_original > 0
        else np.nan
    )

    resumen_impacto.append({
        "Aglomerado": nombre,
        "Casos_imputados": casos_imputados,
        "Promedio_original_PONDIIO": promedio_original,
        "Promedio_posterior_PONDERA": promedio_imputado,
        "Variacion_porcentual": variacion_porcentual
    })

    print(f"\n{nombre}")
    print(f"Casos imputados: {casos_imputados}")
    print(
        "Promedio original ponderado con PONDIIO: "
        f"${promedio_original:,.2f}"
    )
    print(
        "Promedio posterior ponderado con PONDERA: "
        f"${promedio_imputado:,.2f}"
    )
    print(
        "Variación posterior a la imputación: "
        f"{variacion_porcentual:.2f}%"
    )

pd.DataFrame(resumen_impacto).to_csv(
    "impacto_imputacion.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# EXPORTAR RESULTADOS
# ============================================================

df_consolidado.to_csv(
    "resultado_imputacion_2025_mejorado.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 70)
print("ARCHIVOS GENERADOS")
print("=" * 70)
print("- resultado_imputacion_2025_mejorado.csv")
print("- comparacion_modelos.csv")
print("- coeficientes_modelo_mejorado.csv")
print("- impacto_imputacion.csv")
print("- grafico_residuos_log.png")
print("- grafico_qqplot_log.png")
print("- grafico_errores_pesos.png")
print("=" * 70)