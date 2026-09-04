# utilidades_entrenamiento.py

import os

import numpy as np
import pandas as pd

from modelo_lif import ModeloLIFMonocapa
from metricas import calcular_csi, calcular_far, calcular_pod

# Nombres de las variables de contexto (ciclo del día y de la hora). Se usan
# para identificar los pesos correspondientes al contexto en los resultados.
NOMBRES_CONTEXTO = ["sin(doy)", "cos(doy)", "sin(hora)", "cos(hora)"]


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


def entrenar_regresion_logistica_monocapa(modelo, X_entrenamiento, y_entrenamiento,
                                          epocas=1000, tolerancia=1e-12,
                                          alpha_armijo=0.1, beta_armijo=0.5):
    """
    Entrena los pesos del modelo LIF monocapa con regresión logística.

    Las características del readout son las variables normalizadas directas
    más el contexto, con un término de bias: X_lr = [1, variables, contexto].
    Como no existe memoria de sensores, todas las horas del Ajuste participan
    en el ajuste de los pesos.

    El tamaño de paso del descenso de gradiente se determina con búsqueda de
    línea de Armijo (backtracking): por cada iteración se prueba un paso y se
    reduce hasta garantizar un descenso suficiente de la pérdida. Esto asegura
    un descenso monótono hacia el mínimo de la BCE (convexa) sin oscilar ni
    divergir, y elimina la elección manual de una tasa de aprendizaje. El
    entrenamiento se detiene cuando la norma del gradiente es menor que la
    tolerancia o al alcanzar el máximo de iteraciones.

    Argumentos:
        modelo (ModeloLIFMonocapa): modelo LIF monocapa a entrenar.
        X_entrenamiento (np.ndarray): características de entrenamiento
            (meteorológicas normalizadas + contexto).
        y_entrenamiento (np.ndarray): etiquetas reales de entrenamiento.
        epocas (int): número máximo de iteraciones.
        tolerancia (float): norma del gradiente para considerar convergencia.
        alpha_armijo (float): fracción de descenso exigida por la condición de
            Armijo (típicamente 0.1-0.5).
        beta_armijo (float): factor de reducción del paso en el backtracking
            (típicamente 0.5).
    """
    X_lr = np.hstack([np.ones((X_entrenamiento.shape[0], 1)), X_entrenamiento])
    y_lr = y_entrenamiento
    num_muestras = X_lr.shape[0]

    for epoca in range(epocas):
        modelo_lineal = np.dot(X_lr, modelo.pesos)
        predicciones_proba = sigmoide(modelo_lineal)
        error = predicciones_proba - y_lr

        gradiente = np.dot(X_lr.T, error) / num_muestras

        # Convergencia: el gradiente nulo identifica el mínimo de la BCE.
        if np.linalg.norm(gradiente) < tolerancia:
            break

        # Búsqueda de línea de Armijo: se encoge el paso hasta que la pérdida
        # del candidato baje al menos alpha·paso·||gradiente||^2.
        paso = 1.0
        perdida_actual = _calcular_perdida(X_lr, y_lr, modelo.pesos)
        while True:
            candidato = modelo.pesos - paso * gradiente
            if _calcular_perdida(X_lr, y_lr, candidato) <= (
                perdida_actual - alpha_armijo * paso * np.dot(gradiente, gradiente)
            ):
                break
            paso *= beta_armijo

        modelo.pesos = candidato


def aplicar_alerta(evidencia, umbral_alerta):
    """
    Aplica la neurona de alerta: decide con el umbral sobre el potencial de
    alerta V_A, que ya integra la memoria de la alerta (EMA escalar) en su
    cálculo. Alerta = 1 si V_A >= umbral_alerta.

    Retorna:
        tuple: (predicciones, potencial).
    """
    predicciones = (evidencia >= umbral_alerta).astype(int)
    return predicciones, evidencia


def busqueda_hiperparametros(X_ajuste, y_ajuste, X_calibracion, y_calibracion,
                             rango_tau_alerta_horas, rango_percentil_alerta,
                             nombres_variables_meteorologicas,
                             num_variables, num_vars_contexto,
                             epocas_maximas=1000, tolerancia=1e-12,
                             alpha_armijo=0.1, beta_armijo=0.5,
                             prefijo_script="LIFD", ruta_base_salida=None):
    """
    Búsqueda de hiperparámetros del modelo LIF monocapa.

    Los pesos se entrenan una sola vez (regresión logística sobre las
    variables normalizadas) y no dependen de la alerta. Se buscan:
        - la memoria de la alerta tau_A (horas);
        - el percentil de umbral de la alerta (theta_A por percentil del
          potencial de alerta en Calibración).

    El mejor modelo se elige por máximo CSI en Calibración; POD y FAR se
    almacenan como estadísticas de cada instancia. Cada registro identifica la
    variable correspondiente a cada peso (nombres_pesos_modelo). La calibración
    genera dos archivos: el resumen total de todas las instancias
    ({prefijo}-CALIBRACION-RESUMEN_TOTAL.json) y el registro del mejor modelo
    ({prefijo}-CALIBRACION-MEJOR_MODELO.json). No se genera un archivo por
    instancia.

    Argumentos:
        X_ajuste, y_ajuste (np.ndarray): datos de Ajuste.
        X_calibracion, y_calibracion (np.ndarray): datos de Calibración.
        rango_tau_alerta_horas (list): memorias de la alerta (horas) a probar.
        rango_percentil_alerta (list): percentiles candidatos para theta_A.
        nombres_variables_meteorologicas (list): nombres de las variables
            meteorológicas, en el orden de la salida del ProcesadorDatos.
        num_variables (int): cantidad de variables de entrada (meteorológicas).
        num_vars_contexto (int): cantidad de variables de contexto.
        epocas_maximas (int): máximo de iteraciones de la regresión logística.
        tolerancia (float): norma del gradiente para considerar convergencia.
        alpha_armijo (float): fracción de descenso exigida por Armijo.
        beta_armijo (float): factor de reducción del paso en el backtracking.
        prefijo_script (str): identificador del script (LIFA, LIFB, LIFC, ...).
        ruta_base_salida (str|None): carpeta donde se guardan los resultados;
            si es None, se usa "./datos/".

    Retorna:
        dict: mejor CSI, mejor configuración, mejor modelo y todos los resultados.
    """
    if ruta_base_salida is None:
        ruta_salida = "./datos/"
    else:
        ruta_salida = ruta_base_salida

    mejor_csi = -1.0
    mejor_config = None
    mejor_modelo = None
    mejor_resultado = None
    resultados_calibracion = []
    instancia_id = 0

    # Nombres asociados a cada peso del readout.
    nombres_pesos_modelo = (
        ["bias"] + list(nombres_variables_meteorologicas) + NOMBRES_CONTEXTO
    )

    # Entrenamiento de los pesos (no depende de tau_A ni de theta_A).
    modelo_actual = ModeloLIFMonocapa(num_variables, num_vars_contexto)
    entrenar_regresion_logistica_monocapa(
        modelo_actual, X_ajuste, y_ajuste,
        epocas=epocas_maximas, tolerancia=tolerancia,
        alpha_armijo=alpha_armijo, beta_armijo=beta_armijo,
    )

    for tau_alerta in rango_tau_alerta_horas:
        # Evidencia de Ajuste con la memoria escalar de la alerta (tau_A): la
        # simulación deja el estado final escalar de la alerta (V_A), que se
        # usa como estado inicial de Calibración. Así se utilizan todas las
        # horas de Calibración.
        evidencia_ajuste = modelo_actual.calcular_evidencia_secuencia(
            X_ajuste, tau_alerta_horas=tau_alerta
        )
        estado_alerta = modelo_actual.estado_alerta

        potencial_efectivo = modelo_actual.calcular_evidencia_secuencia(
            X_calibracion, tau_alerta_horas=tau_alerta,
            estado_alerta=estado_alerta,
        )
        reales_efectivos = y_calibracion

        for percentil_alerta in rango_percentil_alerta:
            instancia_id += 1
            umbral_alerta = float(np.percentile(potencial_efectivo,
                                                percentil_alerta))
            predicciones = (potencial_efectivo >= umbral_alerta).astype(int)

            csi_actual = calcular_csi(predicciones, reales_efectivos)
            far_actual = calcular_far(predicciones, reales_efectivos)
            pod_actual = calcular_pod(predicciones, reales_efectivos)

            resultados_instancia = {
                'instancia_id': instancia_id,
                'tau_alerta_horas': tau_alerta,
                'percentil_umbral_alerta': percentil_alerta,
                'umbral_alerta': umbral_alerta,
                'csi_calibracion': csi_actual,
                'far_calibracion': far_actual,
                'pod_calibracion': pod_actual,
                'nombres_pesos_modelo': nombres_pesos_modelo,
                'pesos_modelo': modelo_actual.pesos.tolist(),
                'predicciones_calibracion': predicciones.tolist(),
                'reales_calibracion': reales_efectivos.tolist(),
            }
            resultados_calibracion.append(resultados_instancia)

            if csi_actual > mejor_csi:
                mejor_csi = csi_actual
                mejor_config = {
                    'tau_alerta_horas': tau_alerta,
                    'percentil_umbral_alerta': percentil_alerta,
                    'umbral_alerta': umbral_alerta,
                    'pesos_modelo': modelo_actual.pesos.copy(),
                }
                mejor_modelo = modelo_actual
                mejor_resultado = resultados_instancia

    ruta_resumen = os.path.join(
        ruta_salida, f"{prefijo_script}-CALIBRACION-RESUMEN_TOTAL.json"
    )
    pd.DataFrame(resultados_calibracion).to_json(
        ruta_resumen, orient='records', indent=4
    )

    ruta_mejor_modelo = os.path.join(
        ruta_salida, f"{prefijo_script}-CALIBRACION-MEJOR_MODELO.json"
    )
    pd.DataFrame([mejor_resultado]).to_json(
        ruta_mejor_modelo, orient='records', indent=4
    )

    return {
        'mejor_csi': mejor_csi,
        'mejor_config': mejor_config,
        'mejor_modelo': mejor_modelo,
        'mejor_resultado': mejor_resultado,
        'todos_resultados': resultados_calibracion,
    }
