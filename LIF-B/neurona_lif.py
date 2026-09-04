# neurona_lif.py

import numpy as np


class NeuronaLIF:
    """
    Neurona Leaky Integrate-and-Fire (LIF) de un solo compartimiento, en forma
    discreta y determinista (inyección directa de corriente).

    Ecuación por paso (Δt = 1 h):
        V[t] = V[t-1] * tasa_decaimiento
               + (1 - tasa_decaimiento) * corriente_entrada
    y si V[t] >= umbral -> spike = 1 y V se reinicia a 0 (reset total).
    Sin período refractario.
    """

    def __init__(self, umbral, tasa_decaimiento):
        """
        Argumentos:
            umbral (float): umbral de disparo de la neurona.
            tasa_decaimiento (float): fracción del potencial retenida por paso
                (entre 0 y 1; un valor más cercano a 1 = decaimiento más lento).
        """
        self.umbral = umbral
        self.tasa_decaimiento = tasa_decaimiento
        self.potencial_membrana = 0.0

    def calcular_potencial(self, corriente_entrada):
        """
        Calcula el nuevo potencial de membrana:
        V[t] = V[t-1] * tasa_decaimiento
               + (1 - tasa_decaimiento) * corriente_entrada.
        El factor (1 - tasa_decaimiento) normaliza la ganancia de la entrada:
        ante una entrada constante, el potencial converge al nivel de la
        entrada, quedando acotado en la escala de esta.

        Argumentos:
            corriente_entrada (float): corriente de entrada en este paso.

        Retorna:
            float: el nuevo potencial de membrana.
        """
        self.potencial_membrana = (
            self.potencial_membrana * self.tasa_decaimiento
            + (1.0 - self.tasa_decaimiento) * corriente_entrada
        )
        return self.potencial_membrana

    def emitir_spike(self):
        """
        Determina si la neurona emite un spike y actualiza su estado.

        Retorna:
            int: 1 si V >= umbral (y reinicia V a 0), 0 en caso contrario.
        """
        if self.potencial_membrana >= self.umbral:
            self.potencial_membrana = 0.0
            return 1
        return 0
