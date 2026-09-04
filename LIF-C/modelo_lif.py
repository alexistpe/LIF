# modelo_lif.py

import numpy as np


class ModeloLIFMonocapa:
    """
    Modelo Leaky Integrate-and-Fire de una sola capa sin precipitación
    (Modelo LIF C).

    No existen neuronas sensor: las variables meteorológicas normalizadas y el
    contexto entran ponderadas directamente por los pesos del readout, que
    producen la evidencia escalar I[t] = w·[1, X[t], contexto[t]]. La memoria
    de la alerta se aplica como EMA escalar sobre esa evidencia, después del
    readout:
        V_A[t] = alpha_A·V_A[t-1] + (1 - alpha_A)·I[t]
    con alpha_A = exp(-1/tau_alerta_horas). La decisión binaria (V_A >=
    theta_A) se aplica externamente.

    No existe memoria de sensores; la única memoria temporal es la de la
    alerta (EMA escalar). El estado inicial de la alerta para Calibración y
    Validación se obtiene de la simulación del Ajuste, de modo que todas las
    horas de esos conjuntos se utilizan para la predicción (no se descartan
    horas de estabilización).
    """

    def __init__(self, num_variables, num_vars_contexto):
        """
        Argumentos:
            num_variables (int): cantidad de variables de entrada (meteorológicas).
            num_vars_contexto (int): cantidad de variables de contexto.
        """
        # Pesos que conectan [bias, variables, contexto] a la evidencia escalar
        # I[t]. Inicializados en 0 y entrenados con regresión logística.
        self.pesos = np.zeros(num_variables + num_vars_contexto + 1)

        self.num_variables = num_variables
        self.num_vars_contexto = num_vars_contexto

    def calcular_evidencia_secuencia(self, X_secuencia, tau_alerta_horas=None,
                                     estado_alerta=None):
        """
        Calcula el potencial de alerta (V_A) sobre una secuencia completa.

        El readout produce la evidencia escalar I[t] = w·[1, X[t], contexto[t]]
        y la memoria de la alerta se aplica como EMA escalar sobre esa
        evidencia:
            V_A[t] = alpha_A·V_A[t-1] + (1 - alpha_A)·I[t]
        con alpha_A = exp(-1/tau_alerta_horas). El potencial de alerta V_A se
        compara con el umbral theta_A en la aplicación externa.

        El estado escalar de la alerta (V_A) se inicializa con `estado_alerta`
        (por defecto 0) y el estado final queda en `self.estado_alerta`.

        Argumentos:
            X_secuencia (np.ndarray): secuencia [meteorológicas..., contexto...].
            tau_alerta_horas (float): memoria de la alerta (EMA escalar) en horas.
            estado_alerta (float|None): estado inicial escalar de la alerta (V_A).

        Retorna:
            np.ndarray (n,): potencial de alerta V_A por hora.
        """
        if tau_alerta_horas is None:
            raise ValueError("tau_alerta_horas es obligatorio")
        alpha_alerta = np.exp(-1.0 / tau_alerta_horas)
        v_alerta = 0.0
        if estado_alerta is not None:
            v_alerta = estado_alerta

        potenciales_alerta = []
        for i in range(X_secuencia.shape[0]):
            variables = X_secuencia[i, :self.num_variables]
            entradas_contexto = X_secuencia[i, self.num_variables:]
            entradas_combinadas = np.concatenate(
                ([1], variables, entradas_contexto)
            )
            evidencia_instantanea = np.dot(entradas_combinadas, self.pesos)

            # Memoria de la alerta: EMA escalar sobre la evidencia I[t].
            v_alerta = (
                alpha_alerta * v_alerta
                + (1.0 - alpha_alerta) * evidencia_instantanea
            )
            potenciales_alerta.append(v_alerta)

        self.estado_alerta = v_alerta
        return np.array(potenciales_alerta)
