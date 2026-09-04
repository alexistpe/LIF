# modelos_referencia.py

import os

import numpy as np
import pandas as pd

from manejador_datos import (
    cargar_datos_meteostat,
    dividir_datos_por_fechas,
    preprocesar_caracteristicas,
    transformar_datos,
    UMBRAL_LLUVIA_MM_H,
)
from metricas import calcular_csi, calcular_far, calcular_pod

# Nombres de las variables de contexto (ciclo del día y de la hora). Se usan
# para identificar los pesos correspondientes al contexto en los resultados.
NOMBRES_CONTEXTO = ["sin(doy)", "cos(doy)", "sin(hora)", "cos(hora)"]

# Carpeta base de los modelos de referencia.
CARPETA_BASE = os.path.dirname(os.path.abspath(__file__))


def sigmoide(z):
    """Función de activación sigmoide."""
    return 1 / (1 + np.exp(-z))


def _calcular_perdida(X, y, pesos):
    """
    Calcula la entropía cruzada binaria promedio para una regresión logística.

    Argumentos:
        X (np.ndarray): matriz de características con el término de bias.
        y (np.ndarray): etiquetas reales.
        pesos (np.ndarray): pesos del modelo.

    Retorna:
        float: pérdida BCE promedio.
    """
    probabilidades = sigmoide(np.dot(X, pesos))
    return -np.mean(
        y * np.log(probabilidades + 1e-12)
        + (1 - y) * np.log(1 - probabilidades + 1e-12)
    )


def entrenar_regresion_logistica(X, y, epocas=1000, tolerancia=1e-12,
                                 alpha_armijo=0.1, beta_armijo=0.5):
    """
    Entrena los pesos de una regresión logística sobre las variables
    normalizadas (con término de bias) mediante descenso de gradiente sobre la
    entropía cruzada binaria.

    El tamaño de paso se determina con búsqueda de línea de Armijo
    (backtracking): por cada iteración se prueba un paso y se reduce hasta
    garantizar un descenso suficiente de la pérdida. Esto asegura un descenso
    monótono hacia el mínimo de la BCE (convexa) sin oscilar ni divergir, y
    elimina la elección manual de una tasa de aprendizaje. El entrenamiento se
    detiene cuando la norma del gradiente es menor que la tolerancia o al
    alcanzar el máximo de iteraciones.

    Argumentos:
        X (np.ndarray): características normalizadas (sin bias).
        y (np.ndarray): etiquetas reales.
        epocas (int): número máximo de iteraciones.
        tolerancia (float): norma del gradiente para considerar convergencia.
        alpha_armijo (float): fracción de descenso exigida por la condición de
            Armijo (típicamente 0.1-0.5).
        beta_armijo (float): factor de reducción del paso en el backtracking
            (típicamente 0.5).

    Retorna:
        np.ndarray: pesos (bias + variables).
    """
    X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
    pesos = np.zeros(X_bias.shape[1])
    num_muestras = X_bias.shape[0]

    for _ in range(epocas):
        predicciones_proba = sigmoide(np.dot(X_bias, pesos))
        error = predicciones_proba - y
        gradiente = np.dot(X_bias.T, error) / num_muestras

        # Convergencia: el gradiente nulo identifica el mínimo de la BCE.
        if np.linalg.norm(gradiente) < tolerancia:
            break

        # Búsqueda de línea de Armijo: se encoge el paso hasta que la pérdida
        # del candidato baje al menos alpha·paso·||gradiente||^2.
        paso = 1.0
        perdida_actual = _calcular_perdida(X_bias, y, pesos)
        while True:
            candidato = pesos - paso * gradiente
            if _calcular_perdida(X_bias, y, candidato) <= (
                perdida_actual - alpha_armijo * paso * np.dot(gradiente, gradiente)
            ):
                break
            paso *= beta_armijo
        pesos = candidato
    return pesos


def elegir_umbral_probabilidad(probabilidades, reales, percentiles):
    """
    Elige el umbral de probabilidad que maximiza el CSI sobre un conjunto de
    referencia (Calibración), recorriendo percentiles de la probabilidad.

    Argumentos:
        probabilidades (np.ndarray): probabilidades predichas de lluvia.
        reales (np.ndarray): etiquetas reales.
        percentiles (list): percentiles candidatos del umbral.

    Retorna:
        float: umbral de probabilidad elegido.
    """
    mejor_csi = -1.0
    mejor_umbral = 0.5
    for percentil in percentiles:
        umbral = float(np.percentile(probabilidades, percentil))
        predicciones = (probabilidades >= umbral).astype(int)
        csi = calcular_csi(predicciones, reales)
        if csi > mejor_csi:
            mejor_csi = csi
            mejor_umbral = umbral
    return mejor_umbral


def evaluar_regresion_logistica(X_ajuste, y_ajuste, X_calibracion, y_calibracion,
                                X_validacion, y_validacion, percentiles,
                                nombres_pesos):
    """
    Entrena una regresión logística sobre el Ajuste, calibra el umbral de
    probabilidad (percentil con máximo CSI en Calibración) y evalúa en
    Validación.

    Argumentos:
        X_ajuste, y_ajuste: características y etiquetas de Ajuste.
        X_calibracion, y_calibracion: características y etiquetas de Calibración.
        X_validacion, y_validacion: características y etiquetas de Validación.
        percentiles (list): percentiles candidatos del umbral de probabilidad.
        nombres_pesos (list): nombres de las variables asociadas a cada peso
            (bias + variables), en el orden de las columnas.

    Retorna:
        dict: csi, far, pod, umbral_calibrado, pesos y nombres_pesos.
    """
    pesos = entrenar_regresion_logistica(X_ajuste, y_ajuste)

    X_calibracion_bias = np.hstack([np.ones((X_calibracion.shape[0], 1)), X_calibracion])
    probabilidad_calibracion = sigmoide(np.dot(X_calibracion_bias, pesos))

    X_validacion_bias = np.hstack([np.ones((X_validacion.shape[0], 1)), X_validacion])
    probabilidad_lluvia = sigmoide(np.dot(X_validacion_bias, pesos))

    umbral_calibrado = elegir_umbral_probabilidad(
        probabilidad_calibracion, y_calibracion, percentiles
    )
    predicciones = (probabilidad_lluvia >= umbral_calibrado).astype(int)

    return {
        "csi": calcular_csi(predicciones, y_validacion),
        "far": calcular_far(predicciones, y_validacion),
        "pod": calcular_pod(predicciones, y_validacion),
        "umbral_calibrado": umbral_calibrado,
        "nombres_pesos": nombres_pesos,
        "pesos": pesos.tolist(),
    }


def evaluar_referencias():
    """
    Evalúa los modelos de referencia sobre el conjunto de Validación
    (2024-01-01 -> 2025-01-01):

      - Persistencia: predice lluvia en t+1 si la precipitación en t supera el
        umbral (0.2 mm/h).
      - Sin destreza: predice por aleatoriedad, con la probabilidad de lluvia
        igual a la frecuencia del evento en el Ajuste (Bernoulli).
      - Regresión logística: combina linealmente las variables normalizadas y
        aplica la sigmoide; el umbral de probabilidad se calibra por percentil
        (máximo CSI en Calibración). Se evalúan dos variantes: con precipitación
        y sin precipitación como entrada.

    Guarda las métricas en referencias/datos/REFERENCIAS-VALIDACION.json.
    """
    print("--- Modelos de Referencia (Persistencia, Sin Destreza, Regresión Logística) ---")

    RUTA_DATASET = os.path.join(CARPETA_BASE, "..", "dataset", "EDDF.csv")

    FECHA_INICIO_AJUSTE = '2018-01-01'
    FECHA_FIN_AJUSTE = '2023-01-01'
    FECHA_INICIO_CALIBRACION = '2023-01-01'
    FECHA_FIN_CALIBRACION = '2024-01-01'
    FECHA_INICIO_VALIDACION = '2024-01-01'
    FECHA_FIN_VALIDACION = '2025-01-01'

    print("\n--- Cargando y Preprocesando Datos ---")
    df = cargar_datos_meteostat(RUTA_DATASET)

    df_ajuste, df_calibracion, df_validacion = dividir_datos_por_fechas(
        df,
        FECHA_INICIO_AJUSTE, FECHA_FIN_AJUSTE,
        FECHA_INICIO_CALIBRACION, FECHA_FIN_CALIBRACION,
        FECHA_INICIO_VALIDACION, FECHA_FIN_VALIDACION,
    )

    X_ajuste, y_ajuste, procesador = preprocesar_caracteristicas(df_ajuste)
    X_calibracion, y_calibracion = transformar_datos(procesador, df_calibracion)
    X_validacion, y_validacion = transformar_datos(procesador, df_validacion)

    # Precipitación observada de la Validación (para la persistencia), alineada
    # con y_validacion (se descarta la última observación no confirmable).
    precipitacion_validacion = df_validacion["prcp"].to_numpy()
    precipitacion_alineada = precipitacion_validacion[:len(y_validacion)]

    resultados = {}

    # --- 1. Persistencia ---
    print("\n--- Persistencia ---")
    predicciones_persistencia = (precipitacion_alineada >= UMBRAL_LLUVIA_MM_H).astype(int)
    resultados["persistencia"] = {
        "csi": calcular_csi(predicciones_persistencia, y_validacion),
        "far": calcular_far(predicciones_persistencia, y_validacion),
        "pod": calcular_pod(predicciones_persistencia, y_validacion),
        "umbral": UMBRAL_LLUVIA_MM_H,
    }

    # --- 2. Sin destreza (aleatoriedad Bernoulli) ---
    print("\n--- Sin Destreza (Bernoulli) ---")
    base_rate = float(y_ajuste.mean())
    rng = np.random.default_rng(seed=42)
    predicciones_aleatorias = rng.binomial(1, base_rate, len(y_validacion))
    resultados["sin_destreza"] = {
        "csi": calcular_csi(predicciones_aleatorias, y_validacion),
        "far": calcular_far(predicciones_aleatorias, y_validacion),
        "pod": calcular_pod(predicciones_aleatorias, y_validacion),
        "base_rate_ajuste": base_rate,
    }

    # --- 3. Regresión logística (sin memoria), con umbral calibrado por percentil ---
    percentiles_umbral = np.arange(5, 96, 5)
    nombres_variables = procesador.obtener_nombres_caracteristicas()

    # Variante con precipitación (todas las variables normalizadas).
    print("\n--- Regresión Logística con Precipitación (sin memoria) ---")
    nombres_pesos_con_prcp = ["bias"] + list(nombres_variables) + NOMBRES_CONTEXTO
    resultados["regresion_logistica"] = evaluar_regresion_logistica(
        X_ajuste, y_ajuste, X_calibracion, y_calibracion,
        X_validacion, y_validacion, percentiles_umbral,
        nombres_pesos_con_prcp,
    )

    # Variante sin precipitación: se elimina la característica prcp_log.
    print("\n--- Regresión Logística sin Precipitación (sin memoria) ---")
    indice_prcp = nombres_variables.index("prcp_log")
    X_ajuste_sin_prcp = np.delete(X_ajuste, indice_prcp, axis=1)
    X_calibracion_sin_prcp = np.delete(X_calibracion, indice_prcp, axis=1)
    X_validacion_sin_prcp = np.delete(X_validacion, indice_prcp, axis=1)
    nombres_pesos_sin_prcp = (
        ["bias"]
        + [nombre for nombre in nombres_variables if nombre != "prcp_log"]
        + NOMBRES_CONTEXTO
    )
    resultados["regresion_logistica_sin_precipitacion"] = evaluar_regresion_logistica(
        X_ajuste_sin_prcp, y_ajuste, X_calibracion_sin_prcp, y_calibracion,
        X_validacion_sin_prcp, y_validacion, percentiles_umbral,
        nombres_pesos_sin_prcp,
    )

    # --- Salida ---
    print("\n--- Resultados en Validación (2024-01-01 -> 2025-01-01) ---")
    for nombre in resultados:
        r = resultados[nombre]
        print(f"  {nombre}: CSI={r['csi']:.4f}, FAR={r['far']:.4f}, POD={r['pod']:.4f}")

    os.makedirs(os.path.join(CARPETA_BASE, "datos"), exist_ok=True)
    ruta_salida = os.path.join(CARPETA_BASE, "datos", "REFERENCIAS-VALIDACION.json")
    pd.DataFrame([{**{'modelo': k}, **v} for k, v in resultados.items()]).to_json(
        ruta_salida, orient='records', indent=4
    )
    print(f"Resultados guardados en: {ruta_salida}")


if __name__ == "__main__":
    evaluar_referencias()
