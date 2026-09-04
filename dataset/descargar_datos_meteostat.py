# descargar_datos_meteostat.py

import os
from datetime import date

import meteostat
import pandas as pd
from meteostat.api.config import config

# ---------------------------------------------------------------------------
# Configuración de la descarga
# ---------------------------------------------------------------------------
# La librería bloquea por defecto las solicitudes horarias mayores a 3 años;
# el rango del registro supera ese límite, por lo que se habilita la solicitud.
config.block_large_requests = False

# Identificador de la estación EDDF (Frankfurt Flughafen, WMO 10637) en Meteostat.
ESTACION_ID = "10637"

# Rango temporal del registro (las horas se expresan en UTC).
FECHA_INICIO = date(2018, 1, 1)
FECHA_FIN = date(2025, 1, 1)

# Variables meteorológicas requeridas por el modelo.
PARAMETROS = [
    meteostat.Parameter.RHUM,
    meteostat.Parameter.TEMP,
    meteostat.Parameter.PRES,
    meteostat.Parameter.WSPD,
    meteostat.Parameter.WDIR,
    meteostat.Parameter.PRCP,
]

# Columnas finales del CSV, en el orden que consume el modelo.
COLUMNAS_SALIDA = ["time", "rhum", "temp", "pres", "wspd", "wdir", "prcp"]

# Ruta de salida: carpeta 'dataset' compartida entre los modelos.
RUTA_SALIDA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "dataset", "EDDF.csv",
)


def descargar_eddf():
    """
    Descarga el registro horario de la estación EDDF mediante la librería
    Meteostat y lo guarda como CSV en la carpeta 'dataset'.

    La librería devuelve un DataFrame (TimeSeries.fetch()); el CSV se genera
    con to_csv(). La columna 'time' (datetime, UTC) resulta del índice del
    DataFrame y el resto de columnas son las variables solicitadas.
    """
    serie = meteostat.hourly(
        ESTACION_ID,
        FECHA_INICIO,
        FECHA_FIN,
        timezone=None,
        parameters=PARAMETROS,
    )
    df = serie.fetch()
    df = df.reset_index()
    df = df[COLUMNAS_SALIDA]

    os.makedirs(os.path.dirname(RUTA_SALIDA), exist_ok=True)
    df.to_csv(RUTA_SALIDA, index=False)

    print(f"Registros descargados: {len(df)}")
    print(f"Rango temporal: {df['time'].min()} -> {df['time'].max()}")
    print(f"Guardado en: {RUTA_SALIDA}")


if __name__ == "__main__":
    descargar_eddf()
