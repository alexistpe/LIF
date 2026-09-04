# Documentación del Modelo LIF-C (monocapa sin precipitación)

## Descripción conceptual
El modelo LIF-C es una única neurona LIF de alerta que recibe una entrada ponderada (pesos aplicados a cada variable) con la suma de las variables meteorológicas y el contexto:

- Variables meteorológicas (6, sin precipitación): temperatura del aire, humedad relativa, presión atmosférica, velocidad del viento, dirección del viento en componentes `u` y `v`.
- Contexto (4): el día y la hora, codificados de forma cíclica (sin/cos del día del año y sin/cos de la hora), en la escala [-1, 1].

Objetivo: determinar la dependencia de la neurona respecto a la entrada de precipitación, comparando este modelo (LIF-C, sin precipitación) con LIF-D (con precipitación). Al eliminar la entrada de precipitación, se aísla su contribución en el rendimiento de la neurona monocapa.

## Relación con LIF-D
LIF-C se deriva del modelo LIF-D (monocapa): conserva toda la estructura, el diseño y la lógica de LIF-D, y únicamente elimina la variable precipitación como entrada de la neurona de alerta. Es el equivalente monocapa de la relación que LIF-A mantiene con LIF-B. La variable objetivo (¿llovió en la hora siguiente?, `next_prcp_occurrence`) se conserva y sigue derivándose de la precipitación del dataset.

## Flujo del modelo (funcionamiento)
```
I[t] = w·[1, X[t], contexto[t]]              (evidencia escalar instantánea)
V_A[t] = α_A·V_A[t-1] + (1−α_A)·I[t]          (memoria de la alerta: EMA escalar), α_A = exp(−1/τ_A)
alerta[t] = 1 si V_A[t] >= θ_A                (θ_A por percentil de V_A en Calibración)
```

## Estructura de carpetas (igual que LIF-D)
```
SCRIPTS/
├── dataset/                  (compartido entre modelos; incluye el CSV y el script de descarga)
└── LIF-C/
    ├── main.py, manejador_datos.py, escaladores.py, neurona_lif.py,
    ├── modelo_lif.py, utilidades_entrenamiento.py, validacion.py, metricas.py
    ├── datos/                (resultados del modelo LIF-C, prefijo LIFC)
    └── documentacion/        (DOCUMENTACION-LIF-C-...)
```
No se incluye `descargar_datos_meteostat.py` (utilidad independiente en `dataset/`); los scripts leen el `.csv`.

---

## ADAPTACIÓN respecto al modelo LIF-D (cambios de diseño por módulo)

| Módulo | LIF-D (monocapa con precipitación) | LIF-C (monocapa sin precipitación) |
| --- | --- | --- |
| `manejador_datos.py` | Produce 7 características meteorológicas (incluye `prcp_log`) + contexto. | Eliminada la precipitación como característica: el `ProcesadorDatos` produce 6 (`rhum_anom, temp_anom, pres_anom, wspd_robust, u_viento, v_viento`) en `ajustar`/`transformar`. `VARIABLE_PRECIPITACION` se conserva solo para la variable objetivo. Docstrings actualizados ("sin precipitación"). |
| `main.py` | `num_variables = 7`, prefijo `LIFD`. | `num_variables = 6`, `PREFIJO_SCRIPT = "LIFC"`; comentarios y función principal actualizados. Entrenamiento con Armijo (sin LR). |
| `modelo_lif.py` | `ModeloLIFMonocapa` (genérico). | Sin cambios de lógica (la clase es genérica en `num_variables`); el docstring se actualizó a la memoria de la alerta escalar y a "Modelo LIF C". |
| `utilidades_entrenamiento.py` | `busqueda_hiperparametros` con `prefijo_script="LIFD"`. | Sin cambios de lógica; `prefijo_script` por defecto `"LIFC"`. `nombres_pesos_modelo` = 11 (bias + 6 + 4). |
| `validacion.py` | Prefijo `LIFD`. | Prefijo por defecto `"LIFC"`. |
| `escaladores.py`, `neurona_lif.py`, `metricas.py` | Sin cambios. | Sin cambios (copias idénticas). |

Cambio fundamental: LIF-C es idéntico a LIF-D excepto por la ausencia de la entrada `prcp_log` en la neurona de alerta (de 7 a 6 variables meteorológicas, 11 pesos en vez de 12). Todo lo demás (flujo, entrenamiento de pesos, búsqueda de `τ_A`/`θ_A`, burn-in, validación) es igual.

---

## Módulos detallados

### 1. `modelo_lif.py` — `ModeloLIFMonocapa`
- `__init__(num_variables, num_vars_contexto)`: `pesos = zeros(num_variables + num_vars_contexto + 1)` (11: bias + 6 + 4).
- `calcular_evidencia_secuencia(X_secuencia, tau_alerta_horas=None, estado_alerta=None)`: aplica el readout sobre las variables y el contexto (`I[t] = w·[1, X[t], contexto[t]]`) y luego la memoria de la alerta como EMA escalar (`V_A[t] = α_A·V_A[t-1] + (1−α_A)·I[t]`); devuelve el potencial de alerta `V_A` por hora y el estado final queda en `self.estado_alerta`.

### 2. `utilidades_entrenamiento.py`
- `entrenar_regresion_logistica_monocapa(modelo, X, y, epocas=1000, tolerancia=1e-12, alpha_armijo=0.1, beta_armijo=0.5)`: contiene las dos partes (construcción de `X_lr = [1, X]` y el bucle de gradiente con BCE). Búsqueda de línea de Armijo (sin tasa de aprendizaje), convergencia por norma del gradiente, 1000 iteraciones máx.
- `aplicar_alerta(evidencia, umbral_alerta)`: regla de decisión de la alerta: predicción binaria `1` si `evidencia >= umbral_alerta`, `0` en caso contrario. La memoria de la alerta (EMA escalar) ya fue aplicada dentro de `calcular_evidencia_secuencia`.
- `busqueda_hiperparametros(...)`: entrena `w` una sola vez y, por cada `τ_A`, simula el Ajuste (calienta la memoria de la alerta) y continúa con Calibración desde el estado final; sobre el potencial `V_A` de Calibración barre los percentiles `θ_A` (24 valores de `τ_A` 0.5–12 paso 0.5 × 12 percentiles 40–95 paso 5 = 288 instancias); el mejor por máximo CSI; salida `LIFC-CALIBRACION-RESUMEN_TOTAL.json` y `LIFC-CALIBRACION-MEJOR_MODELO.json`.

### 3. `validacion.py`
- `validar_modelo(modelo, X_ajuste, X_calibracion, X_validacion, y_validacion, tau_alerta_horas, umbral_alerta, prefijo_script="LIFC", ruta_base_salida=None)`: simulación continua Ajuste → Calibración → Validación (todas las horas); evidencia + `aplicar_alerta`; guarda `LIFC-VALIDACION-MEJOR.json`.

### 4. `manejador_datos.py` / `escaladores.py` / `neurona_lif.py` / `metricas.py`
- `manejador_datos.py` adaptado (6 características, sin `prcp_log`); el resto son copias de LIF-D sin cambios.

### 5. `main.py`
- `PREFIJO_SCRIPT = "LIFC"`, `RUTA_BASE_SALIDA = .../datos/`.
- `num_variables = 6`, `num_vars_contexto = 4`.
- `RANGO_TAU_ALERTA_HORAS = np.arange(0.5, 12.001, 0.5)`, `RANGO_PERCENTIL_ALERTA = np.arange(40, 96, 5)`.
- `EPOCAS_MAXIMAS = 1000`, `TOLERANCIA = 1e-12`, `ALPHA_ARMIJO = 0.1`, `BETA_ARMIJO = 0.5`.
- Fechas no solapadas: Ajuste 2018-01-01 → 2023-01-01; Calibración 2023-01-01 → 2024-01-01; Validación 2024-01-01 → 2025-01-01.

## Normalización de las entradas directas
Las 6 variables meteorológicas ya están normalizadas por `manejador_datos.py` (anomalía mensual → z-score → min-max en [0,1]) y el contexto es cíclico en [-1,1]. No se requiere normalización adicional.

## Burn-in
No existen sensores; la memoria de la alerta (EMA escalar) se calienta durante la simulación del Ajuste. Calibración y Validación parten del estado final de la sección anterior (simulación continua), de modo que se utilizan TODAS las horas de esos conjuntos (no se descartan horas de estabilización). En el entrenamiento de pesos no hay burn-in.

## Dependencias
- Modelo: `numpy` y `pandas`.
