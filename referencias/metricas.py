# metricas.py

import numpy as np

def calcular_tabla_contingencia(predicciones, reales):
    """
    Calcula la tabla de contingencia (VP, FP, FN, VN).
    
    Argumentos:
        predicciones (np.ndarray): Predicciones binarias del modelo.
        reales (np.ndarray): Valores reales binarios.
        
    Retorna:
        tuple: (VP, FP, FN, VN)
    """
    VP = np.sum((predicciones == 1) & (reales == 1)) # Verdaderos Positivos
    FP = np.sum((predicciones == 1) & (reales == 0)) # Falsos Positivos
    FN = np.sum((predicciones == 0) & (reales == 1)) # Falsos Negativos
    VN = np.sum((predicciones == 0) & (reales == 0)) # Verdaderos Negativos
    
    return VP, FP, FN, VN

def calcular_csi(predicciones, reales):
    """
    Calcula el Critical Success Index (CSI).
    CSI = VP / (VP + FN + FP)
    
    Argumentos:
        predicciones (np.ndarray): Predicciones binarias del modelo.
        reales (np.ndarray): Valores reales binarios.
        
    Retorna:
        float: Valor del CSI. Retorna 0 si el denominador es 0.
    """
    VP, FP, FN, _ = calcular_tabla_contingencia(predicciones, reales)
    denominador = VP + FN + FP
    if denominador == 0:
        return 0.0
    return VP / denominador

def calcular_far(predicciones, reales):
    """
    Calcula el False Alarm Ratio (FAR).
    FAR = FP / (VP + FP)
    
    Argumentos:
        predicciones (np.ndarray): Predicciones binarias del modelo.
        reales (np.ndarray): Valores reales binarios.
        
    Retorna:
        float: Valor del FAR. Retorna 0 si el denominador es 0.
    """
    VP, FP, _, _ = calcular_tabla_contingencia(predicciones, reales)
    denominador = VP + FP
    if denominador == 0:
        return 0.0
    return FP / denominador

def calcular_pod(predicciones, reales):
    """
    Calcula la Probability of Detection (POD).
    POD = VP / (VP + FN)
    
    Argumentos:
        predicciones (np.ndarray): Predicciones binarias del modelo.
        reales (np.ndarray): Valores reales binarios.
        
    Retorna:
        float: Valor del POD. Retorna 0 si el denominador es 0.
    """
    VP, _, FN, _ = calcular_tabla_contingencia(predicciones, reales)
    denominador = VP + FN
    if denominador == 0:
        return 0.0
    return VP / denominador
