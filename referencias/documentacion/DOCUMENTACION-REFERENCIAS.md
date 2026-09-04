# Documentación de los Modelos de Referencia

## Contexto
Esta carpeta contiene tres modelos de referencia (la regresión logística con dos variantes: con y sin precipitación) cuyo objetivo es servir de línea base para comparar la precisión de los modelos LIF. Se evalúan sobre la última sección del dataset (Validación): `2024-01-01 → 2025-01-01`, con las mismas métricas (CSI, FAR, POD) que utilizan los modelos LIF. No poseen memoria temporal.

## Estructura
```
SCRIPTS/referencias/
├── modelos_referencia.py    (script principal de los modelos de referencia)
├── manejador_datos.py       (copia: carga, divide y preprocesa; 7 variables)
├── escaladores.py           (copia: escaladores manuales)
├── metricas.py              (copia: CSI, FAR, POD)
├── datos/                   (resultados: REFERENCIAS-VALIDACION.json)
└── documentacion/           (DOCUMENTACION-REFERENCIAS.md)
```

## Modelos

### 1. Persistencia
- Lógica: predice lluvia en la hora `t+1` si la precipitación en la hora `t` supera el umbral:
  `lluvia[t+1] = 1` si `prcp[t] >= 0.2 mm/h`.
- Objetivo: modelo de referencia competitivo para medir la precisión de los modelos LIF (la persistencia es un baseline fuerte en nowcasting).

### 2. Sin destreza (aleatoriedad)
- Lógica: `Lluvia[t+1] ~ Bernoulli(p)`, con `p` igual a la frecuencia del evento (base rate) observada en el Ajuste: `s = P(o_t = 1) = lluvia / total_registros`. El modelo no usa información meteorológica (`I_t = ∅`); cada hora predice lluvia de forma aleatoria con probabilidad `p` (semilla fija para reproducibilidad).
- Nota de diseño: como `p < 0.5` (evento raro, ≈6 %), una regla de umbral sobre la probabilidad constante (`p_t >= τ` con `τ = 0.5`) nunca predeciría lluvia; por ello el baseline se implementa como pronóstico aleatorio Bernoulli (aleatoriedad).
- Objetivo: determinar la base del modelo LIF (el LIF debe superar la precisión de la aleatoriedad).

### 3. Regresión logística (sin memoria)
- Lógica: toma un valor de cada variable normalizada en el instante `t`, calcula la suma ponderada `z = w₀ + w₁·(humedad) + w₂·(presión) + w₃·(prcp) + ...`, aplica la función sigmoide `P(lluvia) = 1 / (1 + e^(−z))` y emite la alerta según un umbral de probabilidad calibrado por percentil: el umbral se elige como el percentil de la probabilidad que maximiza el CSI en Calibración (mismo criterio que la alerta de los modelos LIF), permitiendo una comparación justa.
- Se evalúan dos variantes:
  - `regresion_logistica` — con precipitación: todas las variables normalizadas (7 meteorológicas + 4 de contexto).
  - `regresion_logistica_sin_precipitacion` — sin precipitación: se elimina la característica `prcp_log` de las entradas (6 meteorológicas + 4 de contexto).
- Los pesos `w` se entrenan con descenso de gradiente sobre la entropía cruzada binaria en el Ajuste, usando búsqueda de línea de Armijo (sin tasa de aprendizaje manual), convergencia por norma del gradiente (`‖g‖ < 1e-12`) y tope de 1000 iteraciones.
- Objetivo: determinar la precisión que aporta la memoria del modelo LIF, comparando este modelo (sin memoria) con los LIF.

## Evaluación y salida
El script `modelos_referencia.py` carga el dataset, divide por fechas (Ajuste 2018→2023, Calibración 2023→2024, Validación 2024→2025), preprocesa con el mismo pipeline que los modelos LIF (7 variables normalizadas + contexto) y evalúa los tres modelos sobre Validación. Guarda las métricas en `referencias/datos/REFERENCIAS-VALIDACION.json`.

## Dependencias
- `numpy` y `pandas` (los escaladores son manuales).
