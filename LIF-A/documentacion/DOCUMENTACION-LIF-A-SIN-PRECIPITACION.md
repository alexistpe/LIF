# Documentación del Modelo LIF-A (sin precipitación)

## Contexto
El modelo LIF-A es una adaptación del modelo LIF-B (documentado en `LIF-B/documentacion/`) en la que la precipitación deja de ser una neurona sensor (entrada de potencial). Se mantiene la misma metodología, el mismo dataset y el mismo pipeline general; únicamente se elimina la variable `prcp` como característica de entrada al modelo. La variable objetivo (¿llovió en la hora siguiente?, `next_prcp_occurrence`) se conserva y se sigue derivando de la precipitación del dataset.

El objetivo de esta versión es evaluar el modelo sin la entrada de precipitación y comparar sus resultados con la versión que sí la incluye (LIF-B).

## Estructura de carpetas (misma que LIF-B)
```
SCRIPTS/
├── dataset/                  (compartido entre modelos)
└── LIF-A/
    ├── manejador_datos.py, escaladores.py, neurona_lif.py, modelo_lif.py,
    ├── utilidades_entrenamiento.py, validacion.py, metricas.py,
    ├── main.py
    ├── datos/                 (resultados del modelo LIF-A, prefijo LIFA)
    └── documentacion/         (DOCUMENTACION-LIF-A-...)
```
La carpeta `LIF-A` replica la estructura de subcarpetas de `LIF-B`. El dataset se lee desde la carpeta `dataset` compartida de `SCRIPTS` (`../dataset/EDDF.csv`). `LIF-A` es una copia adaptada.

## Cambios respecto a la base LIF-B

| Archivo | Cambio | Detalle |
| --- | --- | --- |
| `manejador_datos.py` | Eliminada la precipitación como característica | El `ProcesadorDatos` produce 6 características meteorológicas en vez de 7: `rhum_anom, temp_anom, pres_anom, wspd_robust, u_viento, v_viento` (se elimina `prcp_log`). |
| `manejador_datos.py` | Comentario de `VARIABLE_PRECIPITACION` | Se aclara que `prcp` se usa únicamente para definir la variable objetivo, no como entrada de los sensores. |
| `main.py` | `num_vars_meteorologicas` | De `7` a 6. |
| `main.py` | `TAU_SENSOR_HORAS` | De `[1, 2, 2, 2, 2, 2, 3]` a `[1, 2, 2, 2, 2, 2]` (sin el valor de precipitación). |
| `main.py` | Nombre de la función principal | `correr_modelo_lif_doble_capa_sin_precipitacion()`; mensajes actualizados a "sin precipitación". |
| `main.py` | Identificador de script | `PREFIJO_SCRIPT = "LIFA"` y carpeta `datos/` (sin subcarpeta; convención LIF-A → LIFA). |
| `manejador_datos.py` | Escaladores manuales | Se reemplazaron `StandardScaler`/`RobustScaler`/`MinMaxScaler` por las clases de `escaladores.py` (misma salida numérica). |
| `neurona_lif.py`, `modelo_lif.py`, `utilidades_entrenamiento.py`, `validacion.py`, `metricas.py`, `descargar_datos_meteostat.py` | Ajustes menores | La lógica es genérica en el número de sensores; `modelo_lif.py` identifica el modelo como `Modelo LIF A`. El dataset descargado conserva `prcp` porque la variable objetivo la requiere. |

## Configuración del modelo LIF-A
- Neuronas sensor: 6 (humedad relativa, temperatura, presión, velocidad del viento, componentes u y v). Orden de la salida del `ProcesadorDatos`: `rhum_anom, temp_anom, pres_anom, wspd_robust, u_viento, v_viento`.
- Variables de contexto: 4 (`sin(doy), cos(doy), sin(hora), cos(hora)`).
- Matriz X: 6 meteorológicas + 4 contexto = 10 columnas (LIF-B tiene 11).
- Memoria por sensor (horas): `rhum=1, temp=2, pres=2, wspd=2, u=2, v=2`.
- Búsqueda: percentil de umbral de sensor (40–95 paso 5), memoria de la alerta `τ_A` (0.5–12 paso 0.5), percentil de umbral de alerta (40–95 paso 5). Total: 12×24×12 = 3.456 instancias.
- Regresión logística: entrenamiento con búsqueda de línea de Armijo (sin tasa de aprendizaje manual); máx. 1000 iteraciones, convergencia por norma del gradiente (tolerancia 1e-12).
- División por fechas (no solapada): Ajuste 2018-01-01 → 2023-01-01; Calibración 2023-01-01 → 2024-01-01; Validación 2024-01-01 → 2025-01-01. Cada sección pierde su última observación (etiqueta no confirmable).
- Variable objetivo: `next_prcp_occurrence` = `prcp[t+1] >= 0.2 mm/h`, derivada del dataset.

## Dependencias
- Modelo: `numpy` y `pandas`.
