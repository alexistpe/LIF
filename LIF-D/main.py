# main.py

import os

import numpy as np

from manejador_datos import (
    cargar_datos_meteostat,
    dividir_datos_por_fechas,
    preprocesar_caracteristicas,
    transformar_datos,
)
from utilidades_entrenamiento import busqueda_hiperparametros
from validacion import validar_modelo

# Carpeta base del modelo LIF-D (raíz de los scripts).
CARPETA_BASE = os.path.dirname(os.path.abspath(__file__))


def correr_modelo_lif_monocapa():
    """
    Función principal del modelo LIF de una sola capa (Modelo LIF D): carga,
    divide por fechas, preprocesa, entrena los pesos, busca los parámetros de
    la alerta y valida. Las variables meteorológicas normalizadas y el contexto
    entran ponderadas directamente a la única neurona de alerta.
    """
    print("--- Inicio del Proceso del Modelo LIF Monocapa (Modelo LIF D) ---")

    # --- Configuración General ---
    # Dataset compartido en la carpeta 'dataset' de SCRIPTS.
    RUTA_DATASET = os.path.join(CARPETA_BASE, "..", "dataset", "EDDF.csv")

    # Carpeta de resultados del modelo LIF-D (identificador de script LIFD).
    PREFIJO_SCRIPT = "LIFD"
    RUTA_BASE_SALIDA = os.path.join(CARPETA_BASE, "datos", "")

    # División del dataset por fechas (convención: inicio <= t < fin).
    FECHA_INICIO_AJUSTE = '2018-01-01'
    FECHA_FIN_AJUSTE = '2023-01-01'
    FECHA_INICIO_CALIBRACION = '2023-01-01'
    FECHA_FIN_CALIBRACION = '2024-01-01'
    FECHA_INICIO_VALIDACION = '2024-01-01'
    FECHA_FIN_VALIDACION = '2025-01-01'

    # Cantidad de variables de entrada (meteorológicas del ProcesadorDatos) y
    # de variables de contexto (sin/cos de día y hora).
    num_variables = 7  # rhum, temp, pres, wspd, u, v, prcp
    num_vars_contexto = 4        # sin(doy), cos(doy), sin(hora), cos(hora)

    # Memoria de la alerta (tau_A) en horas: 0.5 a 12 con paso 0.5.
    # El rango incluye el valor "12", debido a esto se extiende el límite ligeramente.
    RANGO_TAU_ALERTA_HORAS = np.arange(0.5, 12.001, 0.5)

    # Umbral de la alerta por percentil del potencial de alerta en Calibración.
    RANGO_PERCENTIL_ALERTA = np.arange(40, 96, 5)

    # Regresión logística: épocas por convergencia y tasa de aprendizaje fija.
    # Entrenamiento de pesos con búsqueda de línea de Armijo (sin tasa de
    # aprendizaje manual).
    EPOCAS_MAXIMAS = 1000
    TOLERANCIA = 1e-12
    ALPHA_ARMIJO = 0.1
    BETA_ARMIJO = 0.5

    # --- 1. Carga, División y Preprocesamiento de Datos ---
    print("\n--- Cargando y Preprocesando Datos ---")
    df = cargar_datos_meteostat(RUTA_DATASET)

    df_ajuste, df_calibracion, df_validacion = dividir_datos_por_fechas(
        df,
        FECHA_INICIO_AJUSTE, FECHA_FIN_AJUSTE,
        FECHA_INICIO_CALIBRACION, FECHA_FIN_CALIBRACION,
        FECHA_INICIO_VALIDACION, FECHA_FIN_VALIDACION,
    )

    # El ProcesadorDatos se ajusta sobre el Ajuste y se reutiliza para
    # transformar Calibración y Validación.
    X_ajuste, y_ajuste, procesador = preprocesar_caracteristicas(df_ajuste)
    X_calibracion, y_calibracion = transformar_datos(procesador, df_calibracion)
    X_validacion, y_validacion = transformar_datos(procesador, df_validacion)

    print(f"Características meteorológicas: {procesador.obtener_nombres_caracteristicas()}")
    print(f"Datos divididos por fechas: Ajuste={X_ajuste.shape[0]}, "
          f"Calibración={X_calibracion.shape[0]}, Validación={X_validacion.shape[0]}")

    # --- 2. Ajuste y Calibración (Entrenamiento de Pesos y Búsqueda de Parámetros de la Alerta) ---
    os.makedirs(RUTA_BASE_SALIDA, exist_ok=True)

    print("\n--- Iniciando Ajuste y Calibración del Modelo LIF ---")
    resultados_busqueda = busqueda_hiperparametros(
        X_ajuste, y_ajuste, X_calibracion, y_calibracion,
        RANGO_TAU_ALERTA_HORAS,
        RANGO_PERCENTIL_ALERTA,
        procesador.obtener_nombres_caracteristicas(),
        num_variables=num_variables,
        num_vars_contexto=num_vars_contexto,
        epocas_maximas=EPOCAS_MAXIMAS,
        tolerancia=TOLERANCIA,
        alpha_armijo=ALPHA_ARMIJO,
        beta_armijo=BETA_ARMIJO,
        prefijo_script=PREFIJO_SCRIPT,
        ruta_base_salida=RUTA_BASE_SALIDA,
    )

    mejor_config = resultados_busqueda['mejor_config']
    mejor_modelo_lif = resultados_busqueda['mejor_modelo']

    print(f"\n--- Mejor Configuración Encontrada en Calibración (CSI: {resultados_busqueda['mejor_csi']:.4f}) ---")
    print(f"  Tau alerta: {mejor_config['tau_alerta_horas']:.1f} h, "
          f"Percentil umbral alerta: {mejor_config['percentil_umbral_alerta']}")
    print(f"  Umbral alerta: {mejor_config['umbral_alerta']:.4f}")
    print(f"  Pesos entrenados (bias + variables + contexto): {np.round(mejor_config['pesos_modelo'], 4)}")

    # --- 3. Validación Final ---
    metricas_validacion = validar_modelo(
        mejor_modelo_lif,
        X_ajuste,
        X_calibracion,
        X_validacion, y_validacion,
        mejor_config['tau_alerta_horas'],
        mejor_config['umbral_alerta'],
        prefijo_script=PREFIJO_SCRIPT,
        ruta_base_salida=RUTA_BASE_SALIDA,
    )

    print("\n--- Proceso del Modelo LIF Monocapa Finalizado ---")
    print("Métricas Finales en Validación:", metricas_validacion)


if __name__ == "__main__":
    correr_modelo_lif_monocapa()
