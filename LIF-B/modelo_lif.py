# modelo_lif.py

import numpy as np

from neurona_lif import NeuronaLIF


class ModeloLIFDobleCapa:
    """
    Modelo Leaky Integrate-and-Fire de doble capa (Modelo LIF B).

    Primera capa: neuronas sensor, una por variable meteorológica. Cada sensor
    tiene su propio umbral (percentil de su activación en el Ajuste) y su
    propia constante de tiempo (decaimiento fijado por variable). El sensor
    integra su variable con fuga
    (V[t] = alpha·V[t-1] + (1 - alpha)·x[t]) y emite un spike binario (0/1) si V
    supera su umbral, con reinicio del potencial.

    Segunda capa (readout): los pesos aprendidos por regresión logística
    combinan [bias, spikes, contexto] en la evidencia escalar I[t]. La memoria
    de la alerta se aplica como EMA escalar sobre esa evidencia, después del
    readout:
        I[t] = w·[1, spikes[t], contexto[t]]
        V_A[t] = alpha_A·V_A[t-1] + (1 - alpha_A)·I[t]
    con alpha_A = exp(-1/tau_alerta_horas). La decisión binaria (V_A >=
    theta_A) se aplica externamente, de modo que la misma evidencia y regla de
    decisión se utilizan en Calibración y en Validación.

    El estado inicial de Calibración y Validación (potenciales de los sensores
    y de la alerta) se obtiene de la simulación del Ajuste, de modo que se
    utilizan todas las horas de esos conjuntos para la predicción (no se
    descartan horas de estabilización).
    """

    def __init__(self, num_neuronas_sensor, num_vars_contexto, params_neurona_sensor):
        """
        Argumentos:
            num_neuronas_sensor (int): cantidad de neuronas sensor.
            num_vars_contexto (int): cantidad de variables de contexto.
            params_neurona_sensor (list): una entrada por sensor, cada una un
                dict con 'umbral' y 'tasa_decaimiento'.
        """
        self.neuronas_sensor = [
            NeuronaLIF(umbral=parametros["umbral"],
                       tasa_decaimiento=parametros["tasa_decaimiento"])
            for parametros in params_neurona_sensor
        ]

        # Pesos que conectan [bias, spikes, contexto] a la evidencia escalar
        # I[t]. Inicializados en 0 y entrenados con regresión logística.
        self.pesos = np.zeros(num_neuronas_sensor + num_vars_contexto + 1)

        self.num_neuronas_sensor = num_neuronas_sensor
        self.num_vars_contexto = num_vars_contexto

    def calcular_horas_burn_in(self, factor=5.0, tau_alerta_horas=None):
        """
        Calcula la cantidad de horas de estabilización (burn-in) de la memoria.

        La membrana responde como un filtro exponencial de constante de tiempo
        tau = -1/ln(decaimiento). Tras k horas, la influencia del estado inicial
        que queda es decaimiento^k = e^(-k/tau); con k = factor·tau la memoria
        alcanza un estado estacionario (factor = 5 ≈ 99 % de la respuesta). Se
        toma la mayor constante de tiempo entre los sensores y, si se indica,
        la de la alerta.

        Argumentos:
            factor (float): múltiplo de la constante de tiempo máxima.
            tau_alerta_horas (float|None): memoria de la alerta en horas, si se
                debe incluir en el cálculo.

        Retorna:
            int: horas de estabilización necesarias.
        """
        decaimientos = [neurona.tasa_decaimiento for neurona in self.neuronas_sensor]
        if tau_alerta_horas is not None:
            decaimientos.append(np.exp(-1.0 / tau_alerta_horas))
        tau_maxima = max(-1.0 / np.log(decaimiento) for decaimiento in decaimientos)
        return int(np.ceil(factor * tau_maxima))

    def reestablecer_estados(self):
        """Reinicia a 0 el potencial de membrana de todas las neuronas."""
        for neurona in self.neuronas_sensor:
            neurona.potencial_membrana = 0.0

    def obtener_estados_sensores(self):
        """Devuelve los potenciales de membrana actuales de los sensores."""
        return [neurona.potencial_membrana for neurona in self.neuronas_sensor]

    def calcular_activacion_subumbral(self, X_secuencia):
        """
        Calcula la activación subumbral de cada sensor sobre una secuencia.

        Es la evolución de la membrana sin umbral ni reinicio (EMA pura):
        A_i[t] = alpha_i·A_i[t-1] + (1 - alpha_i)·x_i[t]. Se utiliza para
        obtener los umbrales por percentil: theta_i = percentil_p(A_i) sobre el
        Ajuste. El factor (1 - alpha_i) acota la activación en la escala de la
        entrada.

        Argumentos:
            X_secuencia (np.ndarray): secuencia [meteorológicas..., contexto...].

        Retorna:
            np.ndarray (n, num_neuronas_sensor): activación por hora y sensor.
        """
        activacion = np.zeros((X_secuencia.shape[0], self.num_neuronas_sensor))
        for t in range(X_secuencia.shape[0]):
            for i, neurona in enumerate(self.neuronas_sensor):
                activacion[t, i] = (
                    neurona.tasa_decaimiento * activacion[t - 1, i]
                    + (1.0 - neurona.tasa_decaimiento) * X_secuencia[t, i]
                )
        return activacion

    def calcular_evidencia_secuencia(self, X_secuencia, estados_iniciales=None,
                                     tau_alerta_horas=None, estado_alerta=None):
        """
        Calcula el potencial de alerta (V_A) sobre una secuencia completa.

        Simula los sensores con memoria y aplica el readout sobre los spikes
        crudos y el contexto, produciendo la evidencia escalar I[t]. Luego
        aplica la memoria de la alerta como EMA escalar sobre esa evidencia:
            I[t] = w·[1, spikes[t], contexto[t]]
            V_A[t] = alpha_A·V_A[t-1] + (1 - alpha_A)·I[t]
        con alpha_A = exp(-1/tau_alerta_horas). El potencial de alerta V_A se
        compara con el umbral theta_A en la aplicación externa.

        Si se indican `estados_iniciales` (potenciales de los sensores
        obtenidos de la simulación del Ajuste), la simulación continúa desde
        esos estados; en caso contrario parte del estado en reposo. El estado
        escalar de la alerta (V_A) se inicializa con `estado_alerta` (por
        defecto 0) y el estado final queda en `self.estado_alerta`.

        Argumentos:
            X_secuencia (np.ndarray): secuencia [meteorológicas..., contexto...].
            estados_iniciales (list|None): potenciales iniciales de los
                sensores (una entrada por sensor).
            tau_alerta_horas (float): memoria de la alerta (EMA escalar) en horas.
            estado_alerta (float|None): estado inicial escalar de la alerta (V_A).

        Retorna:
            np.ndarray (n,): potencial de alerta V_A por hora.
        """
        if estados_iniciales is None:
            self.reestablecer_estados()
        else:
            for neurona, estado in zip(self.neuronas_sensor, estados_iniciales):
                neurona.potencial_membrana = estado

        if tau_alerta_horas is None:
            raise ValueError("tau_alerta_horas es obligatorio")
        alpha_alerta = np.exp(-1.0 / tau_alerta_horas)
        v_alerta = 0.0
        if estado_alerta is not None:
            v_alerta = estado_alerta

        potenciales_alerta = []
        for i in range(X_secuencia.shape[0]):
            entradas_meteorologicas = X_secuencia[i, :self.num_neuronas_sensor]

            spikes_actuales = []
            for k, neurona in enumerate(self.neuronas_sensor):
                neurona.calcular_potencial(entradas_meteorologicas[k])
                spikes_actuales.append(neurona.emitir_spike())

            entradas_contexto = X_secuencia[i, self.num_neuronas_sensor:]
            entradas_combinadas = np.concatenate(
                ([1], np.array(spikes_actuales), entradas_contexto)
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
