# Documentación del Modelo LIF-D (monocapa)

## Contexto
El modelo LIF-D es una versión monocapa del modelo LIF-B: se elimina la capa de neuronas sensor. Las 7 variables meteorológicas normalizadas (incluye precipitación) y las 4 de contexto entran ponderadas directamente (pesos del readout) a la única neurona LIF de alerta, que integra esa evidencia con memoria (`τ_A`) y dispara con un umbral por percentil (`θ_A`).

En términos de lógica, LIF-D equivale a una regresión logística sobre las variables normalizadas (los pesos `w` se entrenan con entropía cruzada binaria) a la que se le añade memoria temporal (el EMA escalar de la alerta sobre la evidencia) y un umbral de operación por percentil. Su objetivo es servir de comparación directa contra el modelo de referencia de regresión logística "con memoria" y contra las versiones con capa de sensores.

## Flujo del modelo (funcionamiento)
```
I[t] = w·[1, X[t], contexto[t]]              (evidencia escalar instantánea)
V_A[t] = α_A·V_A[t-1] + (1−α_A)·I[t]          (memoria de la alerta: EMA escalar), α_A = exp(−1/τ_A)
alerta[t] = 1 si V_A[t] >= θ_A                (θ_A por percentil de V_A en Calibración)
```
No existe memoria de sensores: la única memoria temporal es la de la neurona de alerta (el EMA escalar sobre la evidencia).

## Estructura de carpetas
```
SCRIPTS/
├── dataset/                  (compartido entre modelos; incluye el CSV y el script de descarga)
└── LIF-D/
    ├── main.py, manejador_datos.py, escaladores.py, neurona_lif.py,
    ├── modelo_lif.py, utilidades_entrenamiento.py, validacion.py, metricas.py
    ├── datos/                (resultados del modelo LIF-D, prefijo LIFD)
    └── documentacion/        (DOCUMENTACION-LIF-D-...)
```
Nota: no se incluye `descargar_datos_meteostat.py` en el modelo; ese script es una utilidad independiente que vive en la carpeta `dataset/` y se ejecuta solo para generar el CSV. Los scripts del modelo leen el archivo `.csv`.

---

## ADAPTACIÓN respecto al modelo LIF-B (cambios de diseño por módulo)

| Módulo | LIF-B (doble capa) | LIF-D (monocapa) |
| --- | --- | --- |
| `modelo_lif.py` | Clase `ModeloLIFDobleCapa`: primera capa de 7 neuronas sensor (integran con `α_i`, umbrales por percentil `θ_i`, emiten spikes 0/1) y segunda capa de readout (pesos sobre `[bias, spikes, contexto]`). | Clase `ModeloLIFMonocapa`: sin neuronas sensor. Los pesos (12: bias + 7 variables + 4 contexto) se aplican directamente a las variables normalizadas y producen la evidencia escalar `I[t] = w·[1, X[t], contexto[t]]`; luego se aplica la memoria de la alerta como EMA escalar (`V_A[t] = α_A·V_A[t-1] + (1−α_A)·I[t]`). |
| `utilidades_entrenamiento.py` | `entrenar_regresion_logistica`: simula los sensores (spikes con memoria) y excluye el burn-in de sensores del ajuste. `busqueda_hiperparametros` barre `θ_i` (percentiles de sensor), `τ_A` y `θ_A` (3456 instancias). | `entrenar_regresion_logistica_monocapa`: las características del readout son las variables normalizadas directas (`X_lr = [1, X]`); se usan todas las horas (no hay burn-in de sensores). `busqueda_hiperparametros` entrena los pesos una sola vez y barre solo `τ_A` × `θ_A` (288 instancias). Registros con `nombres_pesos_modelo` y sin `umbrales_sensor`/`percentil_umbral_sensor`. |
| `validacion.py` | Misma lógica (evidencia + `aplicar_alerta`), prefijo `LIFB`. | Misma lógica, prefijo `LIFD`; tipo del modelo `ModeloLIFMonocapa`. |
| `main.py` | `num_neuronas_sensor=7`, `TAU_SENSOR_HORAS`, `RANGO_PERCENTIL_SENSOR`; entrenamiento con Armijo (sin LR). | `num_variables=7`, `num_vars_contexto=4`, sin `TAU_SENSOR_HORAS` ni `RANGO_PERCENTIL_SENSOR`; `RANGO_TAU_ALERTA_HORAS` y `RANGO_PERCENTIL_ALERTA`; entrenamiento con Armijo (sin LR); prefijo `LIFD`. |
| `manejador_datos.py` | Produce 7 características normalizadas (incl. `prcp_log`) + contexto. | Sin cambios (idéntico). |
| `escaladores.py`, `neurona_lif.py`, `metricas.py` | Escaladores manuales; clase `NeuronaLIF`; métricas CSI/FAR/POD. | Sin cambios (copias idénticas). |

Diferencia fundamental: en LIF-B las variables pasan por la codificación no lineal de los sensores (integración con `α_i` + umbral + spike), y el readout aprende sobre esos spikes; en LIF-D las variables entran directamente (combinación lineal) a la única neurona, y la única memoria es el EMA de la alerta. LIF-D aísla el efecto de la capa de sensores y sirve de referencia frente a la regresión logística pura (con memoria añadida).

---

## Módulos detallados

### 1. `modelo_lif.py` — `ModeloLIFMonocapa`
- `__init__(num_variables, num_vars_contexto)`: `pesos = zeros(num_variables + num_vars_contexto + 1)` (12: bias + 7 + 4).
- `calcular_evidencia_secuencia(X_secuencia, tau_alerta_horas=None, estado_alerta=None)`: aplica el readout sobre las variables y el contexto (`I[t] = w·[1, X[t], contexto[t]]`) y luego la memoria de la alerta como EMA escalar (`V_A[t] = α_A·V_A[t-1] + (1−α_A)·I[t]`); devuelve el potencial de alerta `V_A` por hora y el estado final queda en `self.estado_alerta`.

### 2. `utilidades_entrenamiento.py`
- `entrenar_regresion_logistica_monocapa(modelo, X, y, epocas=1000, tolerancia=1e-12, alpha_armijo=0.1, beta_armijo=0.5)`: contiene las dos partes (construcción de `X_lr = [1, X]` y el bucle de gradiente). BCE con búsqueda de línea de Armijo (sin tasa de aprendizaje), convergencia por norma del gradiente, 1000 iteraciones máx.
- `aplicar_alerta(evidencia, umbral_alerta)`: regla de decisión de la alerta: predicción binaria `1` si `evidencia >= umbral_alerta`, `0` en caso contrario. La memoria de la alerta (EMA escalar) ya fue aplicada dentro de `calcular_evidencia_secuencia`.
- `busqueda_hiperparametros(...)`: sin hiperparámetros de sensores. Entrena `w` una sola vez y, por cada `τ_A`, simula el Ajuste (calienta la memoria de la alerta) y continúa con Calibración desde el estado final; sobre el potencial `V_A` de Calibración barre los percentiles `θ_A` (24 valores de `τ_A` 0.5–12 paso 0.5 × 12 percentiles 40–95 paso 5 = 288 instancias). El mejor se elige por máximo CSI; POD y FAR se almacenan. Salida: `{prefijo}-CALIBRACION-RESUMEN_TOTAL.json` y `{prefijo}-CALIBRACION-MEJOR_MODELO.json`.

### 3. `validacion.py`
- `validar_modelo(modelo, X_ajuste, X_calibracion, X_validacion, y_validacion, tau_alerta_horas, umbral_alerta, prefijo_script="LIFD", ruta_base_salida=None)`: simulación continua Ajuste → Calibración → Validación (todas las horas); evidencia + `aplicar_alerta`; métricas CSI/FAR/POD; guarda `LIFD-VALIDACION-MEJOR.json`.

### 4. `manejador_datos.py` / `escaladores.py` / `neurona_lif.py` / `metricas.py`
- Copias de LIF-B, sin cambios funcionales. `manejador_datos.py` produce las 7 características (`rhum_anom, temp_anom, pres_anom, wspd_robust, u_viento, v_viento, prcp_log`) + contexto, y deriva la variable objetivo descartando la última observación (etiqueta no confirmable).

### 5. `main.py`
- `PREFIJO_SCRIPT = "LIFD"`, `RUTA_BASE_SALIDA = .../datos/`.
- `num_variables = 7`, `num_vars_contexto = 4`.
- `RANGO_TAU_ALERTA_HORAS = np.arange(0.5, 12.001, 0.5)`, `RANGO_PERCENTIL_ALERTA = np.arange(40, 96, 5)`.
- `EPOCAS_MAXIMAS = 1000`, `TOLERANCIA = 1e-12`, `ALPHA_ARMIJO = 0.1`, `BETA_ARMIJO = 0.5`.
- Fechas no solapadas: Ajuste 2018-01-01 → 2023-01-01; Calibración 2023-01-01 → 2024-01-01; Validación 2024-01-01 → 2025-01-01.

## Normalización de las entradas directas
Las entradas que recibe la neurona ya están normalizadas por `manejador_datos.py` (anomalía mensual → z-score → min-max en [0,1] para las meteorológicas; sin/cos en [-1,1] para el contexto). No se requiere normalización adicional: los pesos las escalan en la evidencia `I[t]`, sobre la que actúa la memoria de la alerta.

## Burn-in
No existen sensores; la memoria de la alerta (EMA escalar) se calienta durante la simulación del Ajuste. Calibración y Validación parten del estado final de la sección anterior (simulación continua), de modo que se utilizan TODAS las horas de esos conjuntos (no se descartan horas de estabilización). En el entrenamiento de pesos no hay burn-in.

## Dependencias
- Modelo: `numpy` y `pandas`.
