# validacion.py

import os

import numpy as np
import pandas as pd

from metricas import calcular_csi, calcular_far, calcular_pod
from utilidades_entrenamiento import aplicar_alerta


def validar_modelo(modelo, X_ajuste, X_calibracion, X_validacion, y_validacion,
                   tau_alerta_horas, umbral_alerta, prefijo_script="LIFD",
                   ruta_base_salida=None):
    """
    Valida el modelo LIF en el conjunto de Validación.

    La simulación es continua: la memoria escalar de la alerta (EMA sobre la
    evidencia) parte del estado de la sección anterior (potencial al final del
    Ajuste para Calibración, y al final de Calibración para Validación). De
    este modo se utilizan TODAS las horas de Validación para la predicción (no
    se descartan horas de estabilización). Se calcula la evidencia (potencial
    de alerta V_A), se aplica el umbral umbral_alerta y se reportan las
    métricas CSI, FAR y POD.

    Argumentos:
        modelo (ModeloLIFMonocapa): modelo LIF con los pesos entrenados.
        X_ajuste (np.ndarray): características del conjunto de Ajuste.
        X_calibracion (np.ndarray): características del conjunto de Calibración.
        X_validacion (np.ndarray): características del conjunto de Validación.
        y_validacion (np.ndarray): etiquetas reales del conjunto de Validación.
        tau_alerta_horas (float): memoria de la alerta en horas.
        umbral_alerta (float): umbral de decisión de la alerta.
        prefijo_script (str): identificador del script (LIFA, LIFB, LIFC, ...).
        ruta_base_salida (str|None): carpeta donde se guardan los resultados;
            si es None, se usa "./datos/".

    Retorna:
        dict: métricas CSI, FAR y POD del modelo en Validación.
    """
    if ruta_base_salida is None:
        ruta_salida = "./datos/"
    else:
        ruta_salida = ruta_base_salida

    print("\n--- Iniciando Validación del Modelo ---")

    # Simulación continua: Ajuste -> Calibración -> Validación, con la memoria
    # de la alerta (EMA escalar) y el estado heredado de cada sección.
    modelo.calcular_evidencia_secuencia(X_ajuste, tau_alerta_horas=tau_alerta_horas)
    estado_alerta = modelo.estado_alerta

    modelo.calcular_evidencia_secuencia(
        X_calibracion, tau_alerta_horas=tau_alerta_horas,
        estado_alerta=estado_alerta,
    )
    estado_alerta = modelo.estado_alerta

    evidencia_validacion = modelo.calcular_evidencia_secuencia(
        X_validacion, tau_alerta_horas=tau_alerta_horas,
        estado_alerta=estado_alerta,
    )
    predicciones, _ = aplicar_alerta(evidencia_validacion, umbral_alerta)
    reales_validacion = y_validacion

    csi = calcular_csi(predicciones, reales_validacion)
    far = calcular_far(predicciones, reales_validacion)
    pod = calcular_pod(predicciones, reales_validacion)

    print(f"Resultados de Validación:")
    print(f"  CSI: {csi:.4f}")
    print(f"  FAR: {far:.4f}")
    print(f"  POD: {pod:.4f}")

    nombre_archivo_validacion = os.path.join(
        ruta_salida, f"{prefijo_script}-VALIDACION-MEJOR.json"
    )
    resultados_validacion = {
        'csi': csi,
        'far': far,
        'pod': pod,
        'tau_alerta_horas': tau_alerta_horas,
        'umbral_alerta': umbral_alerta,
        'predicciones_validacion': predicciones.tolist(),
        'reales_validacion': reales_validacion.tolist(),
    }
    pd.DataFrame([resultados_validacion]).to_json(
        nombre_archivo_validacion, orient='records', indent=4
    )
    print(f"Resultados de validación guardados en: {nombre_archivo_validacion}")

    return {'csi': csi, 'far': far, 'pod': pod}
