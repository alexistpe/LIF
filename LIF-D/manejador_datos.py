# manejador_datos.py

import numpy as np
import pandas as pd

from escaladores import EscaladorMinMax, EscaladorRobusto, EscaladorZ

# ---------------------------------------------------------------------------
# Parámetros del pipeline de datos
# ---------------------------------------------------------------------------
COLUMNA_TIEMPO = "time"
COLUMNA_OBJETIVO = "next_prcp_occurrence"

# Umbral de intensidad de lluvia (mm/h). El evento de lluvia se define cuando
# la precipitación horaria alcanza este valor.
UMBRAL_LLUVIA_MM_H = 0.2

# Variables meteorológicas con transformación de anomalía mensual
# (anomalía mensual -> z-score -> min-max).
VARIABLES_ANOMALIA = ["rhum", "temp", "pres"]

# Precipitación: transformación log(1+x) -> min-max.
VARIABLE_PRECIPITACION = "prcp"

# Velocidad del viento: robusto -> min-max.
VARIABLE_VELOCIDAD_VIENTO = "wspd"

# Dirección del viento: descomposición en componentes cartesianas u, v
# (u = -WS*sin(θ), v = -WS*cos(θ)) -> robusto -> min-max.
COMPONENTES_VIENTO = ["wspd", "wdir"]


def cargar_datos_meteostat(ruta_archivo):
    """
    Carga el dataset de Meteostat desde un archivo CSV.

    El archivo debe contener una columna 'time' datetime (que identifica la
    fecha y hora de cada observación) junto con las variables meteorológicas
    rhum, temp, pres, wspd, wdir y prcp. El resultado se ordena
    cronológicamente.

    Argumentos:
        ruta_archivo (str): ruta al CSV del dataset.

    Retorna:
        pd.DataFrame: datos cargados y ordenados por tiempo.
    """
    df = pd.read_csv(ruta_archivo)
    df[COLUMNA_TIEMPO] = pd.to_datetime(df[COLUMNA_TIEMPO])
    return df.sort_values(COLUMNA_TIEMPO).reset_index(drop=True)


def construir_contexto_temporal(dataframe):
    """
    Construye las características de contexto temporal cíclicas (sin/cos) del
    día del año y de la hora del día.

    La codificación circular evita la discontinuidad del calendario lineal
    (diciembre-enero y 23:00-00:00) y acota el contexto al intervalo [-1, 1].

    Retorna:
        np.ndarray de forma (n, 4): [sin(doy), cos(doy), sin(hora), cos(hora)].
    """
    tiempo = pd.to_datetime(dataframe[COLUMNA_TIEMPO])
    dia_del_anio = tiempo.dt.dayofyear.to_numpy()
    hora = tiempo.dt.hour.to_numpy()
    return np.column_stack(
        [
            np.sin(2 * np.pi * dia_del_anio / 365.25),
            np.cos(2 * np.pi * dia_del_anio / 365.25),
            np.sin(2 * np.pi * hora / 24),
            np.cos(2 * np.pi * hora / 24),
        ]
    )


def derivar_variable_objetivo(dataframe):
    """
    Define la variable objetivo: y[t] = 1 si la precipitación en t+1 es mayor
    o igual a UMBRAL_LLUVIA_MM_H (mm/h); 0 en caso contrario.

    La última observación de la sección no tiene una hora siguiente dentro de
    la sección, por lo que su etiqueta no es confirmable; dicha observación se
    descarta del conjunto de datos. El resto de las observaciones (incluida la
    penúltima, cuya etiqueta se confirma con la precipitación de la última)
    permanece válida para el entrenamiento y la evaluación.

    Argumentos:
        dataframe (pd.DataFrame): datos ordenados cronológicamente.

    Retorna:
        pd.DataFrame: DataFrame con la columna COLUMNA_OBJETIVO y sin la
        última observación (etiqueta no confirmable).
    """
    # 1. Conversión de la precipitación a tipo numérico para permitir la
    #    comparación contra el umbral.
    precipitacion = dataframe[VARIABLE_PRECIPITACION].astype(float)

    # 2. Evento de lluvia en cada hora: True si la precipitación horaria
    #    alcanza el umbral definido.
    lluvia_hora = precipitacion >= UMBRAL_LLUVIA_MM_H

    # 3. Desplazamiento temporal de una posición hacia atrás: la etiqueta de la
    #    hora t pasa a ser el evento ocurrido en la hora t+1.
    lluvia_hora_siguiente = lluvia_hora.shift(-1)

    # 4. La observación sin "hora siguiente" (la última de la sección) no tiene
    #    una etiqueta confirmable y se descarta del conjunto de datos.
    mascara_valida = lluvia_hora_siguiente.notna()
    dataframe = dataframe[mascara_valida].copy()

    # 5. Conversión del valor booleano a entero binario (1/0).
    dataframe[COLUMNA_OBJETIVO] = lluvia_hora_siguiente[mascara_valida].astype(int)
    return dataframe


class ProcesadorDatos:
    """
    Gestiona las transformaciones y el escalado de las variables meteorológicas
    para el modelo LIF. Cada variable se procesa para representar cuánta
    anomalía genera respecto de su climatología, normalizada a [0, 1], de modo
    que su señal relativa sea comparable y ponderable en el potencial de
    membrana.

    Pipelines por variable:
        rhum, temp, pres : anomalía mensual (mes, hora) -> z-score -> min-max
        wspd             : robusto -> min-max
        wdir             : u = -WS*sin(θ), v = -WS*cos(θ) -> robust -> min-max
        prcp             : log(1+x) -> min-max

    El procesador se ajusta SOLO sobre el período de Ajuste y se reutiliza
    para transformar Calibración y Validación.
    """

    def __init__(self):
        self.climatologias = {}
        self.escaladores_z = {}
        self.escaladores_robustos = {}
        self.escaladores_min_max = {}
        self.nombres_caracteristicas_salida = []

    def _asignar_mes_hora(self, dataframe):
        """Deriva las columnas 'month' y 'hour' a partir de la columna 'time'."""
        dataframe["month"] = dataframe[COLUMNA_TIEMPO].dt.month.astype(int)
        dataframe["hour"] = dataframe[COLUMNA_TIEMPO].dt.hour.astype(int)

    def _calcular_climatologia_mensual(self, dataframe, nombre_variable):
        """
        Climatología de una variable: valor medio por (mes, hora), calculada
        sobre el período de Ajuste.
        """
        self._asignar_mes_hora(dataframe)
        self.climatologias[nombre_variable] = (
            dataframe.groupby(["month", "hour"])[nombre_variable].mean()
        )

    def _aplicar_anomalia_mensual(self, dataframe, nombre_variable):
        """
        Anomalía mensual: valor observado menos la climatología de su
        (mes, hora).
        """
        self._asignar_mes_hora(dataframe)
        climatologia = (
            self.climatologias[nombre_variable]
            .rename(f"{nombre_variable}_clim")
            .reset_index()
        )
        fusionado = dataframe.merge(climatologia, on=["month", "hour"], how="left")
        anomalia = fusionado[nombre_variable] - fusionado[f"{nombre_variable}_clim"]
        return anomalia.to_numpy().reshape(-1, 1)

    def _transformacion_log_mas_uno(self, serie):
        """log(1 + x): comprime la cola pesada de la precipitación y conserva
        el significado del cero (log(1+0) = 0)."""
        return np.log1p(serie.to_numpy()).reshape(-1, 1)

    def _descomponer_viento(self, serie_ws, serie_wd):
        """
        Descompone velocidad (WS, m/s) y dirección (WD, grados) en componentes
        cartesianas u, v: u = -WS*sin(WD_rad), v = -WS*cos(WD_rad).
        """
        wd_rad = np.deg2rad(serie_wd.to_numpy())
        u = -serie_ws.to_numpy() * np.sin(wd_rad)
        v = -serie_ws.to_numpy() * np.cos(wd_rad)
        return u.reshape(-1, 1), v.reshape(-1, 1)

    def ajustar(self, dataframe_ajuste):
        """
        Ajusta las climatologías y los escaladores sobre el período de Ajuste
        y devuelve las características meteorológicas transformadas.

        Retorna:
            np.ndarray: columnas [rhum_anom, temp_anom, pres_anom, wspd_robust,
            u_viento, v_viento, prcp_log].
        """
        df = dataframe_ajuste.copy()
        self._asignar_mes_hora(df)
        caracteristicas = []
        nombres = []

        # --- Humedad relativa, temperatura, presión (anomalía + z-score + min-max) ---
        for var in VARIABLES_ANOMALIA:
            self._calcular_climatologia_mensual(df, var)
            anomalia = self._aplicar_anomalia_mensual(df, var)

            escalador_z = EscaladorZ()
            escalado_z = escalador_z.ajustar_y_transformar(anomalia)
            self.escaladores_z[var + "_anom"] = escalador_z

            escalador_mm = EscaladorMinMax()
            escalado_final = escalador_mm.ajustar_y_transformar(escalado_z)
            self.escaladores_min_max[var + "_anom"] = escalador_mm

            caracteristicas.append(escalado_final)
            nombres.append(var + "_anom")

        # --- Velocidad del viento (robust + min-max) ---
        wspd = df[VARIABLE_VELOCIDAD_VIENTO].to_numpy().reshape(-1, 1)
        escalador_robusto_w = EscaladorRobusto()
        escalado_robusto_w = escalador_robusto_w.ajustar_y_transformar(wspd)
        self.escaladores_robustos["wspd_robust"] = escalador_robusto_w

        escalador_mm_w = EscaladorMinMax()
        escalado_final_w = escalador_mm_w.ajustar_y_transformar(escalado_robusto_w)
        self.escaladores_min_max["wspd_robust"] = escalador_mm_w
        caracteristicas.append(escalado_final_w)
        nombres.append("wspd_robust")

        # --- Dirección del viento: componentes u/v (robust + min-max) ---
        ws, wd = COMPONENTES_VIENTO
        u_viento, v_viento = self._descomponer_viento(df[ws], df[wd])

        escalador_robusto_u = EscaladorRobusto()
        escalado_robusto_u = escalador_robusto_u.ajustar_y_transformar(u_viento)
        self.escaladores_robustos["u_viento"] = escalador_robusto_u

        escalador_mm_u = EscaladorMinMax()
        escalado_final_u = escalador_mm_u.ajustar_y_transformar(escalado_robusto_u)
        self.escaladores_min_max["u_viento"] = escalador_mm_u
        caracteristicas.append(escalado_final_u)
        nombres.append("u_viento")

        escalador_robusto_v = EscaladorRobusto()
        escalado_robusto_v = escalador_robusto_v.ajustar_y_transformar(v_viento)
        self.escaladores_robustos["v_viento"] = escalador_robusto_v

        escalador_mm_v = EscaladorMinMax()
        escalado_final_v = escalador_mm_v.ajustar_y_transformar(escalado_robusto_v)
        self.escaladores_min_max["v_viento"] = escalador_mm_v
        caracteristicas.append(escalado_final_v)
        nombres.append("v_viento")

        # --- Precipitación (log(1+x) + min-max) ---
        log_prcp = self._transformacion_log_mas_uno(df[VARIABLE_PRECIPITACION])
        escalador_mm = EscaladorMinMax()
        escalado_final = escalador_mm.ajustar_y_transformar(log_prcp)
        self.escaladores_min_max["prcp_log"] = escalador_mm
        caracteristicas.append(escalado_final)
        nombres.append("prcp_log")

        self.nombres_caracteristicas_salida = nombres
        return np.hstack(caracteristicas)

    def transformar(self, dataframe):
        """
        Aplica a un nuevo DataFrame las transformaciones aprendidas en el
        Ajuste (mismas climatologías y escaladores, sin re-ajustar).

        Retorna:
            np.ndarray: mismas columnas que ajustar(), en el mismo orden.
        """
        df = dataframe.copy()
        self._asignar_mes_hora(df)
        caracteristicas = []

        for var in VARIABLES_ANOMALIA:
            clave = var + "_anom"
            anomalia = self._aplicar_anomalia_mensual(df, var)
            escalado_z = self.escaladores_z[clave].transformar(anomalia)
            escalado_final = self.escaladores_min_max[clave].transformar(escalado_z)
            caracteristicas.append(escalado_final)

        wspd = df[VARIABLE_VELOCIDAD_VIENTO].to_numpy().reshape(-1, 1)
        escalado_robusto_w = self.escaladores_robustos["wspd_robust"].transformar(wspd)
        escalado_final_w = self.escaladores_min_max["wspd_robust"].transformar(escalado_robusto_w)
        caracteristicas.append(escalado_final_w)

        ws, wd = COMPONENTES_VIENTO
        u_viento, v_viento = self._descomponer_viento(df[ws], df[wd])

        escalado_robusto_u = self.escaladores_robustos["u_viento"].transformar(u_viento)
        escalado_final_u = self.escaladores_min_max["u_viento"].transformar(escalado_robusto_u)
        caracteristicas.append(escalado_final_u)

        escalado_robusto_v = self.escaladores_robustos["v_viento"].transformar(v_viento)
        escalado_final_v = self.escaladores_min_max["v_viento"].transformar(escalado_robusto_v)
        caracteristicas.append(escalado_final_v)

        log_prcp = self._transformacion_log_mas_uno(df[VARIABLE_PRECIPITACION])
        escalado_final = self.escaladores_min_max["prcp_log"].transformar(log_prcp)
        caracteristicas.append(escalado_final)

        return np.hstack(caracteristicas)

    def obtener_nombres_caracteristicas(self):
        """Nombres de las columnas meteorológicas que produce el procesador."""
        return self.nombres_caracteristicas_salida


def preprocesar_caracteristicas(dataframe_ajuste):
    """
    Ajusta el ProcesadorDatos sobre el período de Ajuste y construye la matriz
    de características X junto con la variable objetivo y.

    Columnas de salida:
        meteorológicas: [rhum_anom, temp_anom, pres_anom, wspd_robust,
                         u_viento, v_viento, prcp_log]
        contexto:       [sin(doy), cos(doy), sin(hora), cos(hora)]

    Argumentos:
        dataframe_ajuste (pd.DataFrame): subconjunto de Ajuste (por fechas).

    Retorna:
        tuple: (X_ajuste, y_ajuste, procesador).
    """
    dataframe_ajuste = derivar_variable_objetivo(dataframe_ajuste)

    procesador = ProcesadorDatos()
    X_meteorologicas = procesador.ajustar(dataframe_ajuste)
    X_contexto = construir_contexto_temporal(dataframe_ajuste)
    X = np.hstack([X_meteorologicas, X_contexto])
    y = dataframe_ajuste[COLUMNA_OBJETIVO].to_numpy()
    return X, y, procesador


def transformar_datos(procesador, dataframe):
    """
    Aplica un ProcesadorDatos ya ajustado a un DataFrame (Calibración o
    Validación) y devuelve (X, y).

    Retorna:
        tuple: (X, y).
    """
    dataframe = derivar_variable_objetivo(dataframe)

    X_meteorologicas = procesador.transformar(dataframe)
    X_contexto = construir_contexto_temporal(dataframe)
    X = np.hstack([X_meteorologicas, X_contexto]) 
    y = dataframe[COLUMNA_OBJETIVO].to_numpy()
    return X, y


def dividir_datos_por_fechas(dataframe,
                             fecha_inicio_ajuste, fecha_fin_ajuste,
                             fecha_inicio_calibracion, fecha_fin_calibracion,
                             fecha_inicio_validacion, fecha_fin_validacion):
    """
    Divide el dataset en Ajuste, Calibración y Validación por rangos de fechas.

    Convención: inicio <= t < fin. Los rangos deben generar tres conjuntos 
    cronológicamente disjuntos que cubran el registro.

    Argumentos:
        dataframe (pd.DataFrame): datos con columna 'time' datetime.
        fecha_inicio_ajuste, fecha_fin_ajuste (str/Timestamp): período de Ajuste.
        fecha_inicio_calibracion, fecha_fin_calibracion (str/Timestamp): Calibración.
        fecha_inicio_validacion, fecha_fin_validacion (str/Timestamp): Validación.

    Retorna:
        tuple: (df_ajuste, df_calibracion, df_validacion).
    """
    tiempo = pd.to_datetime(dataframe[COLUMNA_TIEMPO])

    df_ajuste = dataframe[
        (tiempo >= pd.Timestamp(fecha_inicio_ajuste)) & (tiempo < pd.Timestamp(fecha_fin_ajuste))
    ].copy()
    df_calibracion = dataframe[
        (tiempo >= pd.Timestamp(fecha_inicio_calibracion)) & (tiempo < pd.Timestamp(fecha_fin_calibracion))
    ].copy()
    df_validacion = dataframe[
        (tiempo >= pd.Timestamp(fecha_inicio_validacion)) & (tiempo < pd.Timestamp(fecha_fin_validacion))
    ].copy()

    print(f"Ajuste: {fecha_inicio_ajuste} -> {fecha_fin_ajuste} | {len(df_ajuste)} observaciones")
    print(f"Calibración: {fecha_inicio_calibracion} -> {fecha_fin_calibracion} | {len(df_calibracion)} observaciones")
    print(f"Validación: {fecha_inicio_validacion} -> {fecha_fin_validacion} | {len(df_validacion)} observaciones")

    return df_ajuste, df_calibracion, df_validacion
