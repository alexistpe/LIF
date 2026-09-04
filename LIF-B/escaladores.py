# escaladores.py

import numpy as np


class EscaladorZ:
    """
    Escalador de puntuación z (z-score).

    Estandariza una característica con su media y su desvío estándar
    (poblacional): z = (x - media) / desvio.
    """

    def __init__(self):
        self.media = None
        self.desvio = None

    def ajustar(self, datos):
        """Calcula la media y el desvío estándar de la característica."""
        self.media = datos.mean(axis=0)
        self.desvio = datos.std(axis=0)
        return self

    def transformar(self, datos):
        """Aplica la estandarización con los parámetros ajustados."""
        return (datos - self.media) / self.desvio

    def ajustar_y_transformar(self, datos):
        """Ajusta y estandariza en una sola operación."""
        self.ajustar(datos)
        return self.transformar(datos)


class EscaladorRobusto:
    """
    Escalador robusto con mediana y rango intercuartílico (IQR).

    Escala una característica restando su mediana y dividiendo por la
    diferencia entre los percentiles 75 y 25: (x - mediana) / (p75 - p25).
    El IQR se toma de los percentiles 25 y 75 por defecto.
    """

    def __init__(self):
        self.mediana = None
        self.rango_iqr = None

    def ajustar(self, datos):
        """Calcula la mediana y el rango intercuartílico."""
        self.mediana = np.median(datos, axis=0)
        percentiles = np.percentile(datos, [25.0, 75.0], axis=0)
        self.rango_iqr = percentiles[1] - percentiles[0]
        self.rango_iqr = np.where(self.rango_iqr == 0, 1.0, self.rango_iqr)
        return self

    def transformar(self, datos):
        """Aplica el escalado robusto con los parámetros ajustados."""
        return (datos - self.mediana) / self.rango_iqr

    def ajustar_y_transformar(self, datos):
        """Ajusta y escala en una sola operación."""
        self.ajustar(datos)
        return self.transformar(datos)


class EscaladorMinMax:
    """
    Escalador al intervalo [0, 1] con el mínimo y el máximo de la característica.

    Escala una característica restando su mínimo y dividiendo por su rango:
    (x - minimo) / (maximo - minimo). Si el rango es nulo, la característica
    se deja sin escalar (escala unitaria y desplazamiento nulo).
    """

    def __init__(self):
        self.minimo_datos = None
        self.maximo_datos = None
        self.escala = None
        self.desplazamiento = None

    def ajustar(self, datos):
        """Calcula el mínimo, el máximo, la escala y el desplazamiento."""
        self.minimo_datos = datos.min(axis=0)
        self.maximo_datos = datos.max(axis=0)
        rango = self.maximo_datos - self.minimo_datos
        self.escala = np.where(rango == 0, 1.0, 1.0 / rango)
        self.desplazamiento = -self.minimo_datos * self.escala
        return self

    def transformar(self, datos):
        """Aplica el escalado al intervalo [0, 1]."""
        return datos * self.escala + self.desplazamiento

    def ajustar_y_transformar(self, datos):
        """Ajusta y escala en una sola operación."""
        self.ajustar(datos)
        return self.transformar(datos)
