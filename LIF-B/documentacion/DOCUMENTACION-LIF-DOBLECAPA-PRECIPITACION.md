# Documentacion:
Se detalla la estructura y el funcionamiento del script Python diseñado para implementar, entrenar, calibrar y validar un Modelo Leaky Integrate-and-Fire (LIF) de doble capa, con la inclusión de la variable de precipitación. El enfoque principal es una implementación artesanal y modular para garantizar la interpretabilidad y auditabilidad del modelo, evitando "cajas negras" de librerías complejas.
	
	**Arquitectura General del Script**
	El script está organizado en varios módulos lógicos, cada uno con una responsabilidad específica, lo que facilita la comprensión, el mantenimiento y la depuración.
	
	-   `main.py`: Es el script principal que orquesta todo el flujo de trabajo, desde la carga de datos hasta la validación y la presentación de resultados.
	-   `manejador_datos.py`: Módulo encargado de la carga y preprocesamiento de los datos del dataset.
	-   `escaladores.py`: Define las clases `EscaladorZ`, `EscaladorRobusto` y `EscaladorMinMax` (z-score, robusto y min-max).
	-   `neurona_lif.py`: Define la clase `NeuronaLIF` y sus métodos para simular una neurona LIF individual.
	-   `modelo_lif.py`: Define la clase `ModeloLIFDobleCapa` que ensambla las neuronas en una arquitectura de doble capa.
	-   `metricas.py`: Contiene las implementaciones manuales de las métricas de evaluación del modelo (CSI, FAR, POD).
	-   `utilidades_entrenamiento.py`: Implementa la regresión logística para el ajuste de pesos y la lógica para la búsqueda de hiperparámetros.
	-   `validacion.py`: Encargado de la evaluación final del modelo en un conjunto de datos no visto.
	
	---
	
	1. **Módulo `manejador_datos.py`**
	
	**Función**
	Este módulo gestiona todas las operaciones relacionadas con la entrada de datos: carga, limpieza y preparación para su uso en el modelo.
	
	**Objetivo**
	Asegurar que el modelo reciba los datos en el formato correcto (escalados o normalizados) y que la variable objetivo (ocurrencia de lluvia al tiempo `t+1`) esté correctamente identificada.
	
	**Diseño en Código**
	-   `cargar_datos_meteostat(ruta_archivo)`: Carga un archivo CSV de Meteostat. El archivo debe contener una columna `time` datetime (que identifica la fecha y hora de cada observación) junto con las variables meteorológicas `rhum`, `temp`, `pres`, `wspd`, `wdir` y `prcp`. El resultado se ordena cronológicamente.
	-   `ProcesadorDatos` (clase): Encapsula la lógica ajustar/transformar del pipeline de normalización por variable (anomalía mensual → z-score/robusto → min-max), implementada con los escaladores manuales de `escaladores.py` (sin dependencia de librerías de preprocesamiento). Se ajusta SOLO sobre el período de Ajuste y se reutiliza para transformar Calibración y Validación (sin fuga de información). Produce 7 características meteorológicas: `rhum_anom`, `temp_anom`, `pres_anom`, `wspd_robust`, `u_viento`, `v_viento`, `prcp_log`.
	-   `preprocesar_caracteristicas(dataframe_ajuste)`: Ajusta el `ProcesadorDatos` sobre el DataFrame de Ajuste y devuelve `(X, y, procesador)`. Las columnas de salida son las 7 meteorológicas más el contexto cíclico `[sin(doy), cos(doy), sin(hora), cos(hora)]` (11 en total). Deriva la variable objetivo `next_prcp_occurrence` (`y[t] = 1` si `prcp[t+1] >= 0.2 mm/h`).
	-   `transformar_datos(procesador, dataframe)`: Aplica un `ProcesadorDatos` ya ajustado a Calibración o Validación y devuelve `(X, y)`.
	-   `derivar_variable_objetivo(dataframe)`: Define la variable objetivo `next_prcp_occurrence`; descarta la última observación de la sección, cuya etiqueta no es confirmable (no existe una hora siguiente dentro de la sección).
	-   `dividir_datos_por_fechas(dataframe, ...)`: Divide el dataset en Ajuste, Calibración y Validación por rangos de fechas (convención `inicio <= t < fin`). Los rangos deben generar tres conjuntos cronológicamente disjuntos.
	
	---
	
	2. **Módulo `neurona_lif.py`**
	
	**Función**
	Modelar el comportamiento fundamental de una neurona Leaky Integrate-and-Fire.
	
	**Objetivo**
	Proporcionar un bloque de construcción reutilizable y autónomo para crear capas de neuronas dentro del modelo LIF.
	
	**Diseño en Código**
	-   Clase `NeuronaLIF`:
	    -   `__init__(self, umbral, tasa_decaimiento)`: Inicializa la neurona con un `umbral` (potencial necesario para disparar) y una `tasa_decaimiento` (fracción del potencial retenida por paso; más cercana a 1 = decaimiento más lento). El potencial de membrana se inicializa en `0.0`. No incluye período refractario.
	    -   `calcular_potencial(self, corriente_entrada)`: Calcula el nuevo potencial de membrana: `V[t] = V[t-1] * tasa_decaimiento + (1 - tasa_decaimiento) * corriente_entrada` (el factor `(1 - tasa_decaimiento)` normaliza la ganancia de la entrada y acota el potencial en la escala de esta).
	    -   `emitir_spike(self)`: Evalúa si el potencial de membrana actual supera o iguala el `umbral`. Si es así, la neurona "dispara" (emite un `spike` binario `1`) y su potencial se reinicia a `0.0` (reset total). De lo contrario, no emite `spike` (`0`).
	
	---
	
	3. **Módulo `modelo_lif.py`**
	
	### Función
	Ensamblar las neuronas LIF individuales en una arquitectura de red de doble capa.
	
	### Objetivo
	Representar la estructura completa del Modelo LIF B, que incluye 7 neuronas en la primera capa (para variables meteorológicas, incluyendo precipitación) y 1 neurona en la segunda capa (de alerta), que integra los spikes de la primera capa y variables de contexto.
	
	### Diseño en Código
	-   Clase `ModeloLIFDobleCapa`:
	    -   `__init__(self, num_neuronas_sensor, num_vars_contexto, params_neurona_sensor)`:
	        -   `params_neurona_sensor` es una lista de dicts (una entrada por sensor) con `umbral` y `tasa_decaimiento`. Cada sensor tiene su propio umbral (percentil de su activación en el Ajuste) y su propio decaimiento (fijado por variable).
        -   Inicializa los `pesos` del readout. Estos pesos conectan `[bias, spikes de los sensores, variables de contexto]` y producen la evidencia escalar `I[t]`. Incluyen un término de sesgo (`bias`) implícito para la regresión logística.
        -   La memoria de la alerta (EMA escalar) se aplica dentro de `calcular_evidencia_secuencia` sobre la evidencia `I[t]`; la decisión binaria (comparación con `θ_A`) se aplica externamente con `aplicar_alerta` en `utilidades_entrenamiento.py`.
	    -   `calcular_horas_burn_in(factor=5.0, tau_alerta_horas=None)`: Horas de estabilización de la memoria (`τ = −1/ln(decaimiento)`); opcionalmente incluye la constante de la alerta.
	    -   `reestablecer_estados()`: Reinicia a 0 el potencial de todas las neuronas.
	    -   `calcular_activacion_subumbral(X_secuencia)`: Evolución de la membrana sin umbral ni reinicio (`A[t] = α·A[t-1] + (1−α)·x[t]`). Base para calcular los umbrales por percentil `θ_i`.
	    -   `calcular_evidencia_secuencia(X_secuencia, estados_iniciales=None, tau_alerta_horas=None, estado_alerta=None)`: Simula los sensores con memoria y aplica el readout sobre los spikes crudos y el contexto: `I[t] = w·[1, spikes[t], contexto[t]]`. Luego aplica la memoria de la alerta como EMA escalar: `V_A[t] = α_A·V_A[t-1] + (1−α_A)·I[t]`, con `α_A = exp(−1/τ_A)`. Si se indican `estados_iniciales` (potenciales de los sensores al final del Ajuste), la simulación continúa desde esos estados. Devuelve el potencial de alerta `V_A` por hora; la decisión binaria (comparación con `θ_A`) se aplica externamente.
	
	---
	
	4. **Módulo `metricas.py`**
	
	**Función**
	Calcular las métricas de evaluación de rendimiento del modelo de predicción de lluvia.
	
	**Objetivo**
	Proporcionar una forma transparente y auditable de cuantificar la efectividad del modelo, utilizando métricas comunes en la evaluación de pronósticos meteorológicos.
	
	**Diseño en Código**
	-   `calcular_tabla_contingencia(predicciones, reales)`: Calcula los componentes de la tabla de contingencia (variables `VP`, `FP`, `FN`, `VN` en el código):
	    -   `VP` (Verdaderos Positivos): Llovió y se predijo lluvia.
	    -   `FP` (Falsos Positivos): No llovió y se predijo lluvia (falsa alarma).
	    -   `FN` (Falsos Negativos): Llovió y no se predijo lluvia (fallo de detección).
	    -   `VN` (Verdaderos Negativos): No llovió y no se predijo lluvia.
	-   `calcular_csi(predicciones, reales)`: Calcula el Critical Success Index (CSI). Esta métrica es sensible tanto a las predicciones correctas de lluvia como a las falsas alarmas y fallos.
	    -   `CSI = VP / (VP + FN + FP)`
	-   `calcular_far(predicciones, reales)`: Calcula el False Alarm Ratio (FAR). Mide la proporción de predicciones de lluvia que fueron incorrectas.
	    -   `FAR = FP / (VP + FP)`
	-   `calcular_pod(predicciones, reales)`: Calcula la Probability of Detection (POD), también conocida como Tasa de Acierto. Mide la proporción de eventos de lluvia que fueron correctamente predichos.
	    -   `POD = VP / (VP + FN)`
	
	---
	
	5. **Módulo `utilidades_entrenamiento.py`**
	
	**Función**
	Implementar el algoritmo de regresión logística para el ajuste de pesos y gestionar la búsqueda de hiperparámetros del modelo LIF.
	
	**Objetivo**
	Permitir el ajuste iterativo de los parámetros del modelo para optimizar su rendimiento, siguiendo la lógica de la búsqueda descompuesta (sensores → alerta).
	
	**Diseño en Código**
	-   `sigmoide(z)`: Función de activación sigmoide, utilizada en regresión logística para convertir una combinación lineal de entradas en una probabilidad entre 0 y 1.
	-   `entrenar_regresion_logistica(modelo, X_entrenamiento, y_entrenamiento, epocas=1000, tolerancia=1e-12, alpha_armijo=0.1, beta_armijo=0.5)`:
	    -   Entrena los `pesos` del readout con regresión logística, con búsqueda de línea de Armijo (sin tasa de aprendizaje manual) y parada por norma del gradiente.
	    -   Las neuronas sensor se simulan sobre la secuencia de Ajuste con memoria (sin reinicio entre muestras); las primeras horas de estabilización (burn-in) acumulan la membrana pero no participan en el ajuste.
	    -   El entrenamiento se detiene cuando la norma del gradiente de la BCE (entropía cruzada binaria) es menor que la tolerancia o al alcanzar el máximo de épocas. El tamaño de paso lo determina la búsqueda de línea de Armijo en cada iteración (sin tasa de aprendizaje manual).
-   `aplicar_alerta(evidencia, umbral_alerta)`: Regla de decisión de la alerta: predicción binaria `1` si `evidencia >= umbral_alerta`, `0` en caso contrario. La memoria de la alerta (EMA escalar) ya fue aplicada dentro de `calcular_evidencia_secuencia`.
	-   `busqueda_hiperparametros(X_ajuste, y_ajuste, X_calibracion, y_calibracion, taus_sensor_horas, rango_percentil_sensor, rango_tau_alerta_horas, rango_percentil_alerta, nombres_variables_meteorologicas, ...)`:
	    -   Implementa la calibración por Grid Search sobre tres hiperparámetros: el percentil de umbral de los sensores (`θ_i`), la memoria de la alerta (`τ_A`) y el percentil de umbral de la alerta (`θ_A`). El decaimiento de cada sensor se fija por variable (`taus_sensor_horas`).
	    -   Por cada percentil de sensor calcula `θ_i` (percentil de la activación subumbral en el Ajuste) y entrena los pesos. Luego, por cada `τ_A` simula la evidencia del Ajuste (calienta la memoria de sensores y de la alerta) y continúa con Calibración desde el estado final; sobre el potencial de alerta `V_A` de Calibración barre los percentiles `θ_A` (búsqueda descompuesta).
	    -   Evalúa el CSI en Calibración para cada combinación; POD y FAR se almacenan. El mejor modelo se elige por máximo CSI.
	    -   Cada registro identifica la variable correspondiente a cada umbral de sensor y a cada peso (`nombres_umbrales_sensor`, `nombres_pesos_modelo`).
	    -   Genera dos archivos con el prefijo del script (LIFA, LIFB, LIFC, ...): `{prefijo}-CALIBRACION-RESUMEN_TOTAL.json` (todas las instancias) y `{prefijo}-CALIBRACION-MEJOR_MODELO.json` (registro del mejor modelo). No genera un archivo por instancia.
	    -   Retorna el `modelo` con la mejor configuración y todos los resultados.
	
	---
	
	6. **Módulo `validacion.py`**
	
	**Función**
	Realizar la evaluación final del modelo LIF seleccionado.
	
	**Objetivo**
	Proporcionar una estimación imparcial del rendimiento de generalización del modelo, ya que el conjunto de validación nunca ha sido utilizado durante las fases de ajuste o calibración.
	
	**Diseño en Código**
-   `validar_modelo(modelo, X_ajuste, X_calibracion, X_validacion, y_validacion, tau_alerta_horas, umbral_alerta, prefijo_script="LIFB", ruta_base_salida=None)`: 
    -   Toma el `modelo` LIF que resultó ser el mejor de la fase de calibración junto con los parámetros de alerta (`τ_A` y `θ_A`).
    -   Simula de forma continua Ajuste → Calibración → Validación: cada sección parte del estado final (potenciales de los sensores y estado escalar de la alerta) de la anterior, de modo que se utilizan TODAS las horas de Validación (no se descartan horas de estabilización).
    -   Calcula la evidencia con `modelo.calcular_evidencia_secuencia(...)` y aplica la alerta con `aplicar_alerta(evidencia, umbral_alerta)` para generar las predicciones binarias.
    -   Calcula y reporta las métricas CSI, FAR y POD finales; guarda `LIFB-VALIDACION-MEJOR.json`.
	
	---
	
	7. **Script Principal `main.py`**
	
	**Función**
	Actuar como el punto de entrada y coordinador de todo el proceso de desarrollo y evaluación del Modelo LIF.
	
	**Objetivo**
	Automatizar la ejecución de las fases de carga de datos, preprocesamiento, ajuste, calibración y validación, permitiendo ejecutar el análisis completo con una sola ejecución.
	
	**Diseño en Código**
	-   `correr_modelo_lif_doble_capa_precipitacion()`:
	    -   Define `rutas de archivos`, `nombres de columnas` y `parámetros iniciales` (rangos de fechas de división, rangos de hiperparámetros, parámetros de regresión logística).
	    -   Llama a las funciones del `manejador_datos.py` para cargar, preprocesar y dividir el dataset.
	    -   Llama a `busqueda_hiperparametros` de `utilidades_entrenamiento.py` para ejecutar las fases de ajuste y calibración, obteniendo el `mejor_modelo_lif`.
	    -   Llama a `validar_modelo` de `validacion.py` para realizar la evaluación final del `mejor_modelo_lif`.
	    -   Imprime un resumen de los resultados más importantes.
	
	---
	
	2. Módulo escaladores.py
	**Función**
	Implementar los escaladores manuales de normalización (z-score, robusto con mediana e IQR, y min-max) sin depender de librerías de preprocesamiento.
	
	**Objetivo**
	Proveer transformaciones de escala reutilizables y auditables con la interfaz ajustar/transformar, de modo que el ProcesadorDatos normalice cada variable según su método. Los parámetros (media, desvío, mediana, IQR, mínimos y máximos) se estiman solo sobre el período de Ajuste y se reutilizan para transformar Calibración y Validación, evitando fuga de información.
	
	**Diseño en Código**
	- Clase EscaladorZ (puntuación z): estandariza con la media y el desvío estándar poblacional: z = (x − media) / desvio.
	    - __init__(self): inicializa media y desvio en None.
	    - `ajustar(self, datos)`: calcula la media y el desvío estándar de la característica.
	    - `transformar(self, datos)`: aplica la estandarización con los parámetros ajustados.
	    - `ajustar_y_transformar(self, datos)`: ajusta y estandariza en una sola operación.
	- Clase EscaladorRobusto (mediana e IQR): escala restando la mediana y dividiendo por el rango intercuartílico: (x − mediana) / (p75 − p25).
	    - __init__(self): inicializa mediana y rango_iqr en None.
	    - `ajustar(self, datos)`: calcula la mediana y el IQR = p75 − p25; si el IQR es nulo lo fija en 1.0 para evitar división por cero.
	    - `transformar(self, datos)`: aplica el escalado robusto con los parámetros ajustados.
	    - `ajustar_y_transformar(self, datos)`: ajusta y escala en una sola operación.
	- Clase EscaladorMinMax (intervalo 0, 1): escala restando el mínimo y dividiendo por el rango: (x − mínimo) / (máximo − mínimo).
	    - __init__(self): inicializa minimo_datos, maximo_datos, escala y desplazamiento en None.
	    - `ajustar(self, datos)`: calcula el mínimo y el máximo; si el rango es nulo deja la escala en 1.0 y el desplazamiento en 0 (característica constante sin escalar).
	    - `transformar(self, datos)`: aplica el escalado al intervalo 0, 1 (x·escala + desplazamiento).
	    - `ajustar_y_transformar(self, datos)`: ajusta y escala en una sola operación.
		
	**Nomenclatura de Archivos de Resultados**
	
	El script genera archivos de resultados en la carpeta `.../SCRIPTS/LIF-B/datos/` con el prefijo `LIFB`. Cada modelo LIF usa el mismo esquema con su propia letra en su carpeta interna (p. ej. LIF-A → `LIFA`, LIF-B → `LIFB`, LIF-C → `LIFC`). El prefijo se define en `main.py` (`PREFIJO_SCRIPT`) y se propaga a la calibración y la validación. Durante la calibración se generan dos archivos; la validación genera uno:
	
	-   `LIFB-CALIBRACION-RESUMEN_TOTAL.json`: Resumen de todas las instancias evaluadas en la calibración (una por combinación de hiperparámetros), con su CSI/FAR/POD, umbrales, pesos, predicciones y reales, e indicando el nombre de la variable asociada a cada umbral y a cada peso.
	-   `LIFB-CALIBRACION-MEJOR_MODELO.json`: Registro de la instancia con el mayor CSI en calibración (misma estructura que el resumen, con los nombres de variables).
	-   `LIFB-VALIDACION-MEJOR.json`: Métricas finales del mejor modelo en Validación (CSI, FAR, POD) junto con sus parámetros de alerta, predicciones y reales.

**Dependencias**
- Modelo: `numpy` y `pandas`.
