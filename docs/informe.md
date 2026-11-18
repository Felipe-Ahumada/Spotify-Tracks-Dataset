# Inteligencia Musical: Predicción de Popularidad de Canciones (Spotify Tracks)

**Evaluación Parcial N°1 y N°2, MLY1101 Machine Learning, Duoc UC**
**Caso de Estudio C.** Inteligencia musical y predicción de popularidad de canciones (Spotify Tracks)
**Autores.** Felipe Ahumada Silva y Francisca Carrasco Lozano

Este es el informe técnico del proyecto. Documenta el
problema de negocio, los objetivos, los KPIs, las fuentes de datos, la metodología, el trabajo de
comprensión y preparación de datos, y el modelado y evaluación de resultados. El
desarrollo completo y ejecutable de cada paso está en los notebooks de la carpeta
[`notebooks/`](../notebooks/); este informe resume y referencia ese trabajo, no lo reemplaza. Para
instrucciones de instalación y ejecución del proyecto, ver [`README.md`](../README.md).

---

## 1. Descripción del problema de negocio

Un sello discográfico, una plataforma de streaming o un equipo de A&R evalúa constantemente
material nuevo y necesita priorizar en qué canciones invertir esfuerzo de promoción. Hoy esa
priorización depende en gran medida de la reputación ya instalada del artista, lo que perjudica sistemáticamente a artistas nuevos o poco conocidos, ya que sin
trayectoria previa no queda más señal sobre la canción en sí.

El dataset **Spotify Tracks Dataset** (ver [Fuentes de datos](#4-descripción-de-las-fuentes-de-datos-y-herramientas-colaborativas))
contiene más de 100.000 canciones con sus características de audio (danceability, energy,
loudness, tempo, entre otras), su género y su métrica de popularidad en Spotify. El problema de
negocio que aborda este proyecto es la siguiente pregunta. **¿Qué tan bien explican los atributos propios de una
canción, su sonido y su género, la popularidad que alcanza, independientemente de la fama de
quien la interpreta?**

Es una pregunta exploratoria, no una promesa de resultado, y el propio análisis (sección 6) la
responde de forma más matizada de lo esperado. Las características de audio por sí solas
explican muy poco de la popularidad (correlación máxima de apenas -0,18), el género aporta una
señal bastante más fuerte pero es en parte un artefacto del muestreo con que se construyó el
dataset (sección 6.4), y la señal individual más fuerte de todas, la identidad del artista o
álbum, se descarta deliberadamente pese a ser la más predictiva, por las razones éticas
detalladas en la sección 7. Usarla resolvería el problema estadístico, pero no el de negocio,
porque reproduciría exactamente el sesgo de reputación que se quiere evitar. Que "el sonido de
una canción por sí solo prediga poco su éxito" es en sí mismo un hallazgo relevante para el
negocio, ya que matiza cualquier expectativa de que un futuro modelo basado en audio y género pueda
reemplazar esto, y deja claro que su aporte es el de una señal adicional, acotada,
no un reemplazo del criterio humano. La [sección 9](#9-modelado-construcción-comparación-y-evaluación-de-modelos) cierra esta pregunta
con modelos entrenados extremo a extremo. El mejor modelo supervisado alcanza R²≈0,44
combinando sonido y género, un techo consistente con lo que el EDA ya anticipaba con
correlaciones simples, no una revisión del hallazgo.

## 2. Objetivos del proyecto

1. **Comprender la calidad y estructura del dataset**, identificando y resolviendo problemas de
   duplicación, valores inválidos y consistencia antes de sacar cualquier conclusión sobre
   popularidad.
2. **Explorar qué atributos se asocian con la popularidad de una canción** (de audio, de
   género, y de contexto de publicación o release) y cuantificar la fuerza (o debilidad) de esa
   asociación, sin asumir de antemano que los atributos musicales vayan a explicarla bien.
3. **Preparar y transformar los datos** en un pipeline reproducible (scikit-learn
   Pipeline/ColumnTransformer), sin fuga de información entre entrenamiento y prueba, dejando
   el dataset listo para una futura etapa de modelado.
4. **Identificar y documentar los sesgos y limitaciones éticas** del dataset y de las decisiones
   de preparación tomadas, antes de avanzar a modelado.
5. **Entrenar y comparar modelos supervisados de regresión** para predecir popularity,
   **complementar con modelos supervisados de clasificación** (popularity alta/baja) y
   **aplicar una técnica no supervisada** para descubrir segmentos de canciones por sonido,
   evaluando los tres con métricas apropiadas e interpretando los resultados en términos del
   problema de negocio de la sección 1.

Los objetivos 1, 2 y 4 se desarrollan en las secciones 6 y 7 (EDA y preparación de datos). El
objetivo 5 (modelado, fases 4-5 de CRISP-DM) se desarrolla en la
[sección 9](#9-modelado-construcción-comparación-y-evaluación-de-modelos). El despliegue
(fase 6) sigue fuera de alcance, ver
[Estado del proyecto](#10-estado-del-proyecto-y-próximos-pasos).

## 3. Definición de KPIs

Los KPIs se dividen en tres grupos. Los que responden directamente la pregunta de negocio de
la sección 1 (medibles ya desde el EDA, sin necesidad de un modelo entrenado), los que miden la
calidad de los datos como condición necesaria para que los primeros sean confiables, y los que
miden el desempeño del modelado (sección 9), estos últimos sí requieren los modelos
entrenados y por eso se definen con su resultado directamente en esa sección, no aquí.

### KPIs de negocio

Responden la siguiente pregunta de la sección 1. ¿Qué tan bien explica la canción misma su popularidad,
sin depender de la fama de quien la interpreta?

| KPI                                                                                                                 | Meta                                                                                                       | Resultado                                                                                                                                                                                                                                                                                                                                                                                                                    |
| ------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Techo de señal explicable por el contenido de la canción (sin usar identidad de artista/álbum)                   | Cuantificar cuánto puede llegar a explicar, en el mejor caso, un modelo que solo use sonido y género     | Atributos de audio solos: correlación máxima de apenas **-0,18** (instrumentalness). Género (codificado adecuadamente): **0,62**, muy superior a cualquier atributo de audio individual. Este 0,62 es la cota realista de lo que el contenido de la canción puede explicar sin recurrir a quién la interpreta, y fija expectativas honestas para la futura fase de modelado, antes de invertir en entrenarlo. |
| Features basadas en identidad de artista/álbum en el modelo final                                                  | 0                                                                                                          | Se midió que artista/álbum predicen popularidad mucho mejor que cualquier otro atributo (r≈0,63 artista, 0,72 álbum), pero se excluyen deliberadamente para no reforzar el sesgo de "el que ya es famoso, sigue siendo detectado como popular" (ver sección 7).                                                                                                                                                         |
| % del catálogo cuya baja popularidad es un artefacto del contexto de publicación, no de la calidad de la canción | Cuantificar para no confundir "canción poco atractiva" con "dato mal catalogado" en decisiones de negocio | **54%** del pico de canciones con popularity = 0 (5.088 de 9.413) corresponde a duplicados de release (el mismo audio con otro release sin tracción), no a que la canción en sí no guste (ver sección 6.1). Sin este KPI, el negocio podría descartar por error contenido que en realidad sí tiene tracción, solo mal catalogado.                                                                               |

### KPIs de calidad de datos (soporte)

Condición necesaria para que los KPIs de negocio de arriba sean confiables, no miden el
problema de negocio en sí.

| KPI                                          | Meta    | Resultado                                                                                                                                                                         |
| -------------------------------------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Duplicados identificados y resueltos         | 100%    | 24.259 filas por track_id repetido (mismo audio, distinto género) + 9.284 filas de audio idéntico bajo track_id distinto, el 100% consolidado, sin filas duplicadas remanentes. |
| Valores inválidos identificados y tratados  | 100%    | 129 time_signature = 0 / 124 tempo = 0 (sobre el dataset ya consolidado) marcados como faltantes para imputación; 0 valores inválidos sin tratar en el dataset preparado.       |
| Dataset final sin nulos tras la preparación | 0 nulos | X_train_prep / X_test_prep: 0 valores nulos, verificado.                                                                                                                          |

### KPIs de modelado

Miden el desempeño de los modelos entrenados en [`notebooks/modelamiento.ipynb`](../notebooks/modelamiento.ipynb)
(ver detalle y justificación en la sección 9).

| KPI                                                                                             | Meta                                                              | Resultado                                                                                                                                                                                                  |
| ----------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Desempeño del mejor modelo supervisado (regresión) frente a los otros cinco                     | Mejor MAE y R² de los seis                                       | Random Forest (`RandomizedSearchCV`): **R²=0,436**, MAE=10,03 (vs. LightGBM `RandomizedSearchCV` R²=0,430, MAE=10,19, Árbol de Decisión `GridSearchCV` R²=0,401, MAE=10,56, y Regresión Lineal/Ridge/Lasso, las tres en R²=0,387, MAE≈10,75).                                            |
| Desempeño del mejor modelo supervisado (clasificación complementaria) frente a los otros tres  | Mejor Accuracy, F1 y AUC-ROC de los cuatro                          | Random Forest Classifier (`RandomizedSearchCV`): **Accuracy=0,813**, AUC-ROC=0,894 (vs. LightGBM Classifier `RandomizedSearchCV` Accuracy=0,811, AUC-ROC=0,895, Árbol de Decisión `GridSearchCV` Accuracy=0,791, AUC-ROC=0,869, y Regresión Logística Accuracy=0,777, AUC-ROC=0,849). |
| Segmentación no supervisada de canciones por sonido con diferencias de popularidad observables | Identificar segmentos de sonido con distinta popularidad promedio | KMeans (k=4): 4 segmentos de sonido con popularidad promedio distinta, rango acotado de ~5 puntos entre el segmento más bajo (31,96) y el más alto (36,76).                                              |

## 4. Descripción de las fuentes de datos y herramientas colaborativas

**Fuente de datos.** [`data/Spotify_Tracks_Dataset.csv`](../data/Spotify_Tracks_Dataset.csv), 114.000
filas × 20 columnas. Cada fila es una canción con metadatos (track_id, artists, album_name,
track_name, track_genre), su métrica de popularidad en Spotify (popularity, 0-100), la marca
de contenido explícito (explicit) y 13 características de audio extraídas por Spotify
(duration_ms, danceability, energy, key, loudness, mode, speechiness,
acousticness, instrumentalness, liveness, valence, tempo, time_signature). La
descripción detallada de cada columna está en
[`docs/spotify_dataset.pdf`](spotify_dataset.pdf), documento de referencia de Spotify citado
a lo largo del EDA.

Es un dataset público de uso educativo, sin datos personales de usuarios de Spotify (no contiene
historial de escucha ni identificadores de oyentes), ver el detalle de privacidad en la
sección 7.

**Herramientas colaborativas y de trabajo.** El equipo se coordinó de forma presencial para
acordar avances, repartir el trabajo y discutir los hallazgos antes de incorporarlos al informe.

- **Git y GitHub.** Control de versiones y repositorio compartido. Es la herramienta central del
  trabajo colaborativo del proyecto, ya que mantiene el historial de commits del informe y de ambos
  notebooks, permite revisar qué cambió entre versiones de un análisis, y actúa como fuente única
  de verdad para el equipo. Se eligió sobre alternativas de carpeta compartida porque los
  notebooks de Jupyter son archivos JSON grandes, así que sin control de versiones, dos personas editando
  en paralelo se sobrescriben sin dejar rastro.
- **Notebooks autocontenidos.** Decisión de trabajo derivada de lo anterior, ya que cada notebook
  reconstruye su propio dataset desde el CSV crudo, sin depender de haber ejecutado el otro. Esto
  permite que dos personas trabajen sobre distintas etapas del proyecto en paralelo y que
  cualquiera pueda ejecutar y verificar el trabajo del otro sin coordinación previa.
- **Jupyter Notebook.** Entorno de análisis; el código, su salida y la justificación en Markdown
  quedan juntos en el mismo archivo, que es lo que permite que el trabajo sea revisable por otra
  persona sin explicación adicional.
- **Documentación de Spotify** ([`spotify_dataset.pdf`](spotify_dataset.pdf)). Referencia
  compartida de definiciones y rangos válidos de cada columna. Fija un criterio común, ya que sin ella,
  cada integrante decidiría por su cuenta qué valor es "inválido".

## 5. Metodología utilizada (CRISP-DM)

El proyecto sigue la metodología **CRISP-DM** (Cross-Industry Standard Process for Data Mining),
de 6 fases iterativas. Este proyecto cubre las primeras cinco.

1. **Comprensión del negocio.** Cubierta en las secciones 1-3 de este informe, con el problema,
   los objetivos y los KPIs.
2. **Comprensión de los datos.** Cubierta por [`notebooks/analisis_exploratorio.ipynb`](../notebooks/analisis_exploratorio.ipynb),
   con la carga, calidad, duplicados, valores inválidos, outliers, correlaciones y distribución del
   target.
3. **Preparación de los datos.** Cubierta por [`notebooks/preprocesamiento.ipynb`](../notebooks/preprocesamiento.ipynb),
   con la consolidación adicional, tratamiento de inválidos, ingeniería de características y pipeline de
   transformación.
4. **Modelado.** Cubierta por [`notebooks/modelamiento.ipynb`](../notebooks/modelamiento.ipynb), con
   seis modelos supervisados de regresión (Regresión Lineal, Ridge, Lasso, Árbol de Decisión,
   Random Forest, LightGBM), cuatro modelos supervisados de clasificación como complemento
   (Regresión Logística, Árbol de Decisión, LightGBM Classifier, Random Forest Classifier) y
   una técnica no supervisada (KMeans) de segmentación por sonido, ver sección 9.
5. **Evaluación.** Cubierta en la sección 9, con la comparación de modelos con métricas de regresión
   (MAE, R²) y de clasificación (accuracy, F1, AUC-ROC, precision y recall por clase), el perfilado de los segmentos obtenidos
   por clustering y la interpretación de resultados en términos del problema de negocio.
6. **Despliegue.** Fuera de alcance, se define como trabajo futuro (ver sección 10).

## 6. Preparación y análisis exploratorio de los datos (EDA)

Resumen de los hallazgos y decisiones de [`analisis_exploratorio.ipynb`](../notebooks/analisis_exploratorio.ipynb)
y [`preprocesamiento.ipynb`](../notebooks/preprocesamiento.ipynb), ambos notebooks son
autocontenidos, están documentados en español celda a celda, y fueron verificados ejecutándolos
de punta a punta (no solo escritos y asumidos correctos).

### 6.1 Calidad de datos: dos niveles de duplicación

El dataset crudo (114.000 filas) tiene **24.259 filas repetidas por track_id**. La misma
canción aparece una vez por cada género bajo el que Spotify la cataloga. Se consolida con
groupby('track_id'), uniendo los géneros con ; (mismo criterio que ya usa artists para
colaboraciones) y promediando popularity (variaba en apenas 4,33% de los casos). El resultado son
**89.740 canciones únicas**. Se elimina además 1 fila con `duration_ms = 0` y metadatos nulos
(error de carga).

El EDA identifica un **segundo nivel de duplicación que la consolidación por track_id no resuelve**. track_id identifica un *release* (álbum, single, recopilatorio), no la canción en
sí. Agrupando por las 13 columnas de audio se encuentran **9.284 filas (10,35%) en 3.022 grupos**
de audio idéntico bajo distinto track_id, reediciones o recopilatorios de la misma grabación.
El caso más extremo, la canción *"I'm Good (Blue)"*, tiene popularity = 0 en un release y 98
en otro para el mismo audio exacto. En 466 de esos grupos el rango de popularity es ≥30 puntos.
Esto es relevante porque explica parte del pico de canciones con popularity = 0 (9.413 filas,
10,49% del dataset). El 54% de esas filas (5.088) pertenece a estos grupos de audio duplicado, y
en 779 de los 3.022 grupos la misma canción tiene popularity = 0 en un release y > 0 en otro, es decir, buena parte del "pico en cero" no es "esta canción no se reproduce", es "este release
específico no se reproduce".

El EDA deja este segundo nivel **documentado pero sin resolver** (es una decisión de preparación,
no de exploración). preprocesamiento.ipynb lo resuelve colapsando cada grupo de audio duplicado
a una fila, **promediando** popularity entre releases (mismo criterio ya usado para el primer
nivel de duplicación, en vez de quedarse con el máximo, que introduciría un sesgo optimista
sistemático). El dataset preparado final queda en **83.478 canciones únicas**.

Como el modelado (sección 9.2) usa, además de la regresión, una clasificación complementaria de
`popularity` alta/baja respecto a la mediana, el EDA verifica de entrada si esa partición está
desbalanceada. Sobre las 89.740 canciones únicas por track_id, la mediana de popularity es
**33,0**. 50,32% de las filas queda bajo la mediana (baja) y 49,68% sobre ella (alta), una razón
mayoría/minoría de **1,01**. Es decir, prácticamente sin desbalance, esperable al partir una
variable continua por su propia mediana, lo que respalda usar accuracy sin remuestreo en la
clasificación de la sección 9.2.

### 6.2 Valores inválidos

Contra los rangos documentados por Spotify, se encuentran tres inconsistencias reales (sobre el
dataset ya consolidado por audio, 83.478 filas).

- **tempo = 0** (124 filas). Ninguna canción tiene 0 BPM, es físicamente imposible.
- **time_signature = 0** (129 filas). 96% de estas filas también tiene tempo = 0, y el 81%
  pertenece al género sleep, es el mismo fallo del algoritmo de Spotify al no poder estimar
  pulso musical en sonidos ambientales, no un compás real.
- **loudness > 0** (66 filas, hasta 4,53 dB). Másters muy comprimidos que superan el techo
  digital de 0 dBFS, inusual pero válido, no un error.

time_signature = 1 (788 filas sobre el dataset final) se investiga aparte y se confirma
**válido** (tempo coherente ~110 BPM, géneros variados, 0 casos con tempo = 0), se conserva
sin modificar. Los casos de tempo/time_signature
inválidos se marcan como NaN (no se eliminan las filas) para que el SimpleImputer del pipeline
los complete, conservando el resto de los atributos válidos de esas canciones.

### 6.3 Outliers y correlaciones

Boxplots muestran outliers esperables en duration_ms (canciones de varias horas), loudness
(hasta -49 dB) y en speechiness/instrumentalness/liveness (variables concentradas cerca de
0 con colas largas). La matriz de correlación muestra multicolinealidad moderada
(energy-loudness=0,76; acousticness vs. energy/loudness=-0,73/-0,58), pero el hallazgo
central es que **ninguna variable de audio correlaciona fuertemente con popularity**. La más
alta es instrumentalness con apenas **-0,18**. La popularidad no se explica linealmente por
cómo suena la canción.

### 6.4 Género: la señal más fuerte, con una trampa de diseño

track_genre es multietiqueta (114 géneros posibles, separados por ; 82% de las canciones
tiene un solo género, promedio 1,27). Su distribución (~1.000 canciones por género) no es
orgánica, sino la huella del muestreo estratificado con que se construyó el dataset, y no refleja
popularidad real de cada género en Spotify, una limitación a tener presente si se usa como
predictor (ver sección 7).

Pese a eso, un chequeo de la popularidad promedio por género muestra la relación más fuerte
encontrada en todo el dataset. Pop promedia 47,6 de popularidad, grindcore 14,6, un rango de
más de 3x, y muy superior a cualquier variable de audio. Por eso se incorpora como predictor (ver
6.5) en vez de descartarla.

### 6.5 Ingeniería de características y pipeline de preparación

Se evaluaron y descartaron varias transformaciones (interacciones entre variables de audio, todas
con |r| < 0,06) y se retuvieron dos, con señal verificada.

- **distancia_duracion_optima** (`|duración_min − 3,75|`). La popularidad sube y baja en forma
  de U invertida según la duración (pico 34,4-35,5 entre 3-4 min, mínima en canciones muy cortas
  o muy largas), relación que duration_ms en bruto no captura por ser lineal
  (r=-0,02 vs. r=-0,09 con la distancia).
- **es_calmado_positivo** (valence ≥ 0,5 y energy < 0,5). El único de los 4 cuadrantes del
  modelo de ánimo (valence×energy) cuya popularidad real queda sistemáticamente por debajo
  (-2,84 puntos) de lo que predice un modelo lineal simple con valence y energy por separado.

Para incorporar track_genre se evaluaron tres opciones, multi-hot completo (~114 columnas),
agrupación manual en familias, o codificación por promedio de género (*mean/target encoding*). Se
eligió la última (GenreMeanEncoder, transformador personalizado), el promedio de popularidad
histórica del/los género(s) de la canción, calculado **solo con el conjunto de entrenamiento**
para no filtrar información hacia el conjunto de prueba. Se verificó (comparando la versión que
usa el pipeline contra una versión *out-of-fold*) que la fuga de información dentro del propio
entrenamiento es despreciable (diferencia de r ≈ 0,002), dado que los 114 géneros tienen cientos
de canciones cada uno. La consolidación por audio (6.1) mejora además esta señal, ya que la correlación
de GenreMeanEncoder con popularity en el conjunto de prueba sube de 0,568 a **0,618**.

El pipeline final (scikit-learn Pipeline + ColumnTransformer) combina Winsorizer
(recorte de outliers) → SimpleImputer → StandardScaler para las 11 variables numéricas,
SimpleImputer → OneHotEncoder para las 5 categóricas, GenreMeanEncoder → StandardScaler
para el género, y un CorrelationFilter final (umbral 0,9) como resguardo de multicolinealidad.

El recorte de Winsorizer no es uniforme. Se calibró un límite propio por columna según el %
real de outliers medido con la regla IQR (1,5× el rango intercuartílico), en vez de aplicar el
mismo 5%/5% a las 11 columnas por igual. Por ejemplo, energy, acousticness y valence no tienen
outliers reales (0% ambos lados) y no se recortan; instrumentalness sí concentra una cantidad
grande (21,5%, toda hacia arriba), y se le aplica un tope de 10% en vez del % real completo,
para no generar un pico artificial de valores idénticos en la columna.
track_id, artists, album_name y track_name se excluyen de las features por altísima
cardinalidad (73.261 nombres de canción y 45.880 álbumes distintos sobre 83.478 filas, casi cada
valor aparece una sola vez). El resultado es **83.478 canciones → 66.782 de entrenamiento / 16.696
de prueba → 29 columnas finales, sin valores nulos**, verificado ejecutando el pipeline de punta
a punta.

## 7. Evaluación de sesgos, ética y privacidad de los datos

**Privacidad.** El dataset no contiene datos personales de oyentes (sin historial de escucha ni
identificadores de usuarios). Las únicas variables descriptivas de personas (`artists`) son
nombres de artistas públicos asociados a un catálogo musical comercial, no datos de individuos
privados. De todas formas se excluyen de las features del pipeline (sección 6.5), pero por falta
de valor predictivo utilizable (altísima cardinalidad), no por motivos de privacidad, es una
distinción importante, ya que no es una medida de anonimización, sino una decisión de ingeniería de
características.

**Sesgo de diseño del dataset (género).** La distribución casi uniforme de track_genre
(~1.000 canciones por género, sección 6.4) es un artefacto del muestreo con que se construyó el
dataset, no la popularidad real de cada género en el mundo real. Un modelo entrenado con
GenreMeanEncoder aprende "qué tan popular es un género **en esta muestra**", que no es lo mismo
que "qué tan popular es ese género entre oyentes reales de Spotify", una limitación a comunicar
si el modelo se usa para decisiones reales de programación musical.

**Sesgo de contexto de publicación.** El hallazgo de la sección 6.1 (misma canción, distinta
popularidad según el release) muestra que popularity no es una propiedad pura del contenido
musical, sino también del contexto de publicación (álbum, momento, promoción). Un modelo que
prediga popularidad a partir solo de audio y género tiene un techo de precisión inherente por
esta razón, independiente de qué tan bueno sea el modelo.

**Sesgo de "el que ya es famoso, sigue siendo detectado como popular".** Se evaluó (fuera de los
notebooks, como parte del diseño de features) codificar artists/album_name con la misma
técnica de promedio histórico usada para género. El álbum resulta la señal más fuerte de todo el
dataset (r≈0,72 en conjunto de prueba, contra 0,62 de género); el artista queda en un nivel
similar al género (r≈0,63). Se decidió **no incorporarlas**, por dos razones. (1) Predecirían
principalmente "qué tan famoso es ya el artista/álbum", no "qué hace que una canción suene
popular", que es el problema de negocio real (sección 1), un modelo así sería inútil para
descubrir artistas nuevos por ejemplo, y (2) reforzarían un sesgo de retroalimentación
("rich get richer"), ya que un modelo que aprende que un artista ya famoso seguirá siendo popular
perpetúa esa ventaja en vez de evaluar la canción en sus propios méritos. Este es el KPI ético de
la sección 3 (0 features de identidad de artista/álbum en el modelo).

**Multicolinealidad y calidad del pipeline** (sección 6.3, 6.5). Se trata como una cuestión de
calidad de datos y no de ética, pero se documenta acá por completitud. El CorrelationFilter
(umbral 0,9) actúa como resguardo automático y verificable ante variables redundantes que
pudieran sesgar el ajuste de un futuro modelo lineal.

## 8. Estructura del proyecto

```text
Spotify-Tracks-Dataset/
├── README.md                          — presentación del proyecto: qué es y cómo ejecutarlo
├── requirements.txt                   — dependencias de Python del proyecto
├── data/
│   └── Spotify_Tracks_Dataset.csv     — dataset crudo (114.000 filas)
├── docs/
│   ├── spotify_dataset.pdf            — documentación de cada columna del dataset
│   └── informe.md                     — este documento: informe técnico del proyecto
├── notebooks/
│   ├── analisis_exploratorio.ipynb    — Fase 2 CRISP-DM: comprensión de los datos (EDA)
│   ├── preprocesamiento.ipynb         — Fase 3 CRISP-DM: preparación de datos
│   └── modelamiento.ipynb             — Fases 4-5 CRISP-DM: modelado y evaluación
└── images/                            — figuras exportadas por el EDA y el modelado
```

No existe una carpeta `models/` porque los modelos de esta entrega no se serializan a disco (el
notebook `modelamiento.ipynb` los entrena y evalúa de punta a punta, de forma reproducible); se
incorporaría junto con los artefactos serializados del pipeline si el proyecto avanzara a la fase
de despliegue (fase 6, ver [Estado del proyecto](#10-estado-del-proyecto-y-próximos-pasos)).

**Para reproducir.** Ver las instrucciones de instalación y ejecución en [`README.md`](../README.md).

## 9. Modelado: construcción, comparación y evaluación de modelos

Esta sección resume [`notebooks/modelamiento.ipynb`](../notebooks/modelamiento.ipynb) (fases 4-5
de CRISP-DM), verificado ejecutándolo de punta a punta. Como los notebooks anteriores, es
autocontenido, ya que reconstruye `X_train_prep`, `X_test_prep`, `y_train`, `y_test` con el mismo
pipeline de `preprocesamiento.ipynb` (sección 6.5) en vez de depender de haberlo ejecutado antes.

### 9.1 Modelos supervisados: predicción de popularity

Se comparan seis modelos supervisados contra `popularity`.

| Modelo                                           | MAE            | R²             |
| ------------------------------------------------ | -------------- | --------------- |
| Regresión Lineal                                | 10,75          | 0,387           |
| Ridge (por defecto, alpha=1,0)                   | 10,75          | 0,387           |
| Ridge (`RandomizedSearchCV`)                     | 10,75          | 0,387           |
| Lasso (por defecto, alpha=1,0)                   | 11,00          | 0,380           |
| Lasso (`RandomizedSearchCV`)                     | 10,75          | 0,387           |
| Árbol de Decisión (`GridSearchCV`)           | 10,56          | 0,401           |
| Random Forest (por defecto, sin ajuste)          | 10,05          | 0,428           |
| LightGBM (por defecto, sin ajuste)               | 10,39          | 0,418           |
| LightGBM (`RandomizedSearchCV`)                  | 10,19          | 0,430           |
| **Random Forest (`RandomizedSearchCV`)** | **10,03** | **0,436** |

- **Regresión Lineal**, elegida por ser el modelo más simple e interpretable y por servir de
  referencia directa de las correlaciones lineales del EDA (sección 6.3, ninguna variable de
  audio supera |r|=0,18). Al combinar todas las variables (incluido el género, codificado como
  `genero_popularidad_media`) ya explica el 38,7% de la varianza, muy por encima de lo que
  sugería cualquier correlación individual. Su coeficiente estandarizado de mayor magnitud es,
  por lejos, el del género (11,7), seguido de `explicit` y `energy`.
- **Ridge y Lasso, las variantes regularizadas de la Regresión Lineal.** Se incluyen para
  verificar directamente si la Regresión Lineal sin regularizar está desaprovechando desempeño
  por no controlar la magnitud de sus coeficientes. Ambas se ajustaron con
  `RandomizedSearchCV` sobre `alpha` (penalización L2 para Ridge, L1 para Lasso), mismo criterio
  metodológico que los demás modelos (validación cruzada explícita de 5 folds). **El resultado es
  que no hay ninguna mejora sobre la Regresión Lineal sin regularizar.** Ridge encuentra `alpha=100` como
  "mejor" pero da exactamente el mismo R²=0,387 (de hecho, cualquier `alpha` entre 0,001 y 1.000
  da el mismo resultado, verificado aparte); Lasso con su configuración por defecto (`alpha=1,0`)
  empeora (R²=0,380), pero la búsqueda encuentra que el mejor `alpha` es casi nulo (0,005, apenas
  lleva 3 de 29 coeficientes a cero), recuperando el mismo R²=0,387 de la Regresión Lineal sin
  regularizar. Es la confirmación empírica directa de lo que ya anticipaba el diagnóstico de
  subajuste (sección 9.3), ya que el problema de la Regresión Lineal es falta de capacidad para
  explotar más señal, no sobreajuste, así que ninguna forma de regularización puede mejorarla,
  porque regularizar solo puede restarle capacidad a un modelo, nunca sumarle.

  **Esto no significa que Ridge/Lasso sean, en general, inútiles para mejorar una regresión
  lineal** (son una técnica legítima y ampliamente usada), sino que las tres condiciones bajo
  las que mejoran no se cumplen en este proyecto. (1) **Sobreajuste**, que Ridge/Lasso corrigen
  achicando coeficientes, pero acá no hay (sección 9.3, brecha train-test de solo 0,007); (2)
  **multicolinealidad severa**, que Ridge en particular estabiliza, pero acá el propio pipeline
  ya la resuelve antes de llegar al modelo (`CorrelationFilter`, |r|>0,9, sección 6.5); y (3)
  **pocas filas frente a muchas variables**, el escenario donde Lasso más aporta seleccionando
  variables automáticamente, pero acá hay 66.782 filas de entrenamiento contra solo 29 variables,
  una proporción muy alejada de ese caso. Con ninguna de las tres condiciones presentes, el
  resultado (ninguna mejora) es exactamente lo que la teoría predeciría, no una sorpresa ni un
  fallo de la técnica.
- **Árbol de Decisión**, elegido como punto intermedio entre la Regresión Lineal y el ensamble
  de Random Forest, ya que un único árbol captura no linealidades, pero sin el promedio de muchos
  árboles que reduce varianza. Ajustado con `GridSearchCV` **exhaustivo** sobre `max_depth` y
  `min_samples_leaf` (espacio chico, 7×7=49 combinaciones, así que se recorre completo en vez de
  muestrear al azar; mejor combinación `max_depth=8, min_samples_leaf=50`), queda apenas por
  encima de la Regresión Lineal (R²=0,401 vs. 0,387). Un único árbol, aunque no lineal, no
  alcanza a capturar mucha más señal que el modelo lineal. Depende casi por completo del género
  (94,2% de la importancia de variables), más concentrado que el Random Forest.
- **Random Forest Regressor, una técnica de ensamble (*bagging*).** En vez de ajustar un único
  modelo, combina las predicciones de 300 árboles de decisión independientes, cada uno entrenado
  sobre una muestra bootstrap distinta del conjunto de entrenamiento. La predicción final es el
  promedio de las 300 predicciones individuales. Se elige frente a un único árbol porque
  promediar muchos modelos entrenados con distintas muestras reduce la varianza de cualquier
  árbol individual (inestable ante pequeños cambios en los datos) sin perder su capacidad de
  capturar interacciones y relaciones no lineales entre variables de audio y género que un
  modelo lineal no puede representar. El aporte del ensamble es medible y no marginal, pues mejora el
  R² en +0,035 frente al Árbol de Decisión individual (0,436 vs. 0,401, +8,7%) y en +0,049
  frente a la Regresión Lineal (vs. 0,387), con un MAE 0,53 puntos menor que el árbol y 0,72
  menor que la lineal. En vez de elegir los hiperparámetros del ensamble a mano, se ajustaron
  con `RandomizedSearchCV` (24 combinaciones de un espacio de 600 posibles sobre `n_estimators`,
  `max_depth`, `min_samples_leaf`, `max_features`; validación cruzada explícita de 5 folds con
  `KFold(shuffle=True, random_state=42)`, ver sección 9.3). La mejor combinación encontrada
  (`n_estimators=300, min_samples_leaf=2, max_features=1.0, max_depth=16`) mejora el R² en
  +0,008 frente a la configuración por defecto de scikit-learn (`n_estimators=100`, sin límite de
  profundidad), confirmando que la búsqueda justifica el costo frente a dejar esos valores sin
  ajustar.
  Un detalle relevante del resultado de esa búsqueda es que `max_features=1.0` significa que, a
  diferencia de un Random Forest "clásico", cada división considera **todas** las variables
  disponibles, no un subconjunto aleatorio, por lo que la única fuente de aleatoriedad (y de reducción de
  varianza) entre los 300 árboles es el muestreo bootstrap de filas, no de columnas. Es, en la
  práctica, *bagging* de árboles de decisión puro; se sigue llamando Random Forest porque es la
  clase de scikit-learn usada (`RandomForestRegressor`), que permite esa configuración como caso
  particular.
- **LightGBM, una técnica de ensamble (*boosting*).** A diferencia del *bagging* de Random
  Forest (300 árboles independientes entrenados en paralelo y promediados), el *boosting*
  entrena árboles en secuencia, donde cada árbol nuevo corrige los errores del anterior, reduciendo
  sesgo en vez de varianza. Se incluye como contraste porque es la otra gran familia de
  ensambles de árboles de decisión. Se ajustó con el mismo criterio que Random Forest
  (`RandomizedSearchCV`, 24 combinaciones, validación cruzada explícita de 5 folds con
  `KFold(shuffle=True, random_state=42)`) sobre un espacio de `n_estimators`, `max_depth`,
  `learning_rate`, `num_leaves` y `min_child_samples`. La mejor combinación encontrada
  (`num_leaves=127, n_estimators=100, min_child_samples=50, max_depth=8, learning_rate=0.1`)
  mejora el R² en +0,012 frente a su configuración por defecto (0,430 vs. 0,418), confirmando
  otra vez que la búsqueda justifica el costo. Es, además, la configuración menos compleja de las tres
  evaluadas en esta sección (solo 100 árboles, frente a los 300 de Random Forest). Con menos
  árboles, `min_child_samples=50` (cada hoja necesita 50 canciones, bastante conservador) y
  `max_depth=8`, es la que menos sobreajusta de los dos ensambles por un margen amplio (brecha
  train-test de solo 0,092, frente a 0,257 de Random Forest, sección 9.3). Pero incluso ya
  optimizado, LightGBM queda por debajo de Random Forest (R²=0,430 vs. 0,436, -0,006), más
  cerca que en la búsqueda inicial más acotada, pero sin alcanzarlo. Es consistente con el
  diagnóstico de la sección 6.4, ya que con el género concentrando la mayor parte de la señal
  explicable, hay poco margen residual para que la corrección secuencial de errores del
  *boosting* aporte algo que el promedio de 300 árboles bootstrap de Random Forest no capture ya.
  Su importancia de variables confirma esto de forma aún más marcada que Random Forest, ya que
  `genero_popularidad_media` domina con **79,6%** de la importancia total (`importance_type='gain'`),
  seguida de variables de audio similares entre sí (`acousticness`, `loudness`, `danceability`,
  `duration_ms`, `energy`, cada una entre 1,5-2,2%), una concentración todavía mayor en género
  que el 61,5% de Random Forest, consistente con que el *boosting*, al corregir error de forma
  secuencial, termina apoyándose más en la señal ya fuerte del género que en explorar variables
  de audio adicionales.
- **Importancia de variables** (Random Forest con hiperparámetros de la búsqueda).
  `genero_popularidad_media` domina con ~61,5% de la importancia total; le siguen varias
  variables de audio similares entre sí (`acousticness`, `danceability`, `loudness`, `energy`,
  `valence`, `tempo`, `liveness`, `speechiness`, `duration_ms`, cada una entre 2,9-4,0%). Es
  consistente con el hallazgo del EDA (sección 6.4), ya que el género es, por lejos, la señal más
  fuerte del contenido de la canción, y ninguna variable de audio individual domina por sí sola
  (sección 6.3).

### 9.2 Clasificación complementaria: ¿popularidad alta o baja?

Como complemento a la regresión, se entrena un segundo tipo de modelo supervisado sobre el
mismo problema de fondo. En vez de predecir el puntaje exacto, se clasifica cada canción como
`popularity` **alta** o **baja** respecto a la mediana del conjunto de entrenamiento (35),
reutilizando el mismo pipeline y las mismas 29 columnas. No es una repetición de la regresión,
ya que responde una pregunta de negocio distinta y más directamente accionable, *"¿esta canción nueva
suena como que va a tener mejor o peor desempeño que la mediana del catálogo?"*, con sus
propias métricas (accuracy, F1 y AUC-ROC para los cuatro modelos, y precision/recall por clase
para el modelo elegido).

| Modelo                                                      | Accuracy        | F1 macro        | AUC-ROC         |
| ----------------------------------------------------------- | --------------- | --------------- | --------------- |
| Regresión Logística                                       | 0,777           | 0,777           | 0,849           |
| Árbol de Decisión (`GridSearchCV`)                      | 0,791           | 0,791           | 0,869           |
| LightGBM Classifier (por defecto, sin ajuste)                | 0,802          | 0,802          | 0,885          |
| LightGBM Classifier (`RandomizedSearchCV`)                   | 0,811          | 0,811          | **0,895**  |
| Random Forest Classifier (por defecto, sin ajuste)          | 0,795          | 0,795          | 0,873           |
| **Random Forest Classifier (`RandomizedSearchCV`)** | **0,813** | **0,813** | 0,894 |

El Árbol de Decisión (`max_depth=15, min_samples_leaf=50`, ajustado con `GridSearchCV`
exhaustivo igual que en regresión, sección 9.1) queda entre ambos extremos, igual que en la
regresión, mejorando sobre la Regresión Logística pero sin llegar al desempeño del ensamble. **El
Random Forest Classifier es, igual que en regresión, una técnica de ensamble** (300 árboles
entrenados por *bagging*, cada uno sobre una muestra bootstrap de filas; la búsqueda de
hiperparámetros encontró, igual que en regresión, `max_features=1.0` como mejor combinación,
donde cada árbol considera todas las variables en cada división, sin subconjunto aleatorio de
columnas, así que la predicción final es la clase mayoritaria entre 300 árboles que solo
difieren en qué filas vieron). Su aporte frente al árbol individual es medible, con +0,022 de
Accuracy (0,813 vs. 0,791, +2,8%) y +0,025 de AUC-ROC (0,894 vs. 0,869), y frente a la Regresión
Logística, +0,036 de Accuracy y +0,045 de AUC-ROC, y tanto él como el Árbol de Decisión se
ajustaron con el mismo criterio (Árbol, `GridSearchCV` exhaustivo; Random Forest,
`RandomizedSearchCV`, 8 combinaciones de un espacio de 600 posibles; ambos con validación
cruzada explícita de 5 folds con `StratifiedKFold(shuffle=True, random_state=42)`, ver sección
9.3). **Comparación contra la configuración por defecto** (igual que en regresión, sección
9.1). El Random Forest Classifier sin ajustar ya queda por encima de los otros dos modelos
(0,795 vs. 0,791 y 0,777), y la búsqueda de hiperparámetros todavía mejora +0,018 de Accuracy y
+0,021 de AUC-ROC sobre esos valores sin ajustar, lo que confirma, igual que en regresión, que la
búsqueda justifica el costo y no es solo el ensamble en sí lo que explica el resultado.

**LightGBM Classifier, el mismo contraste *bagging* vs. *boosting* de la sección 9.1, ahora en
clasificación.** Se ajustó con el mismo criterio que el Random Forest Classifier
(`RandomizedSearchCV`, 8 combinaciones, `StratifiedKFold` de 5 folds) sobre el mismo espacio de
`n_estimators`, `max_depth`, `learning_rate`, `num_leaves` y `min_child_samples` usado en
regresión. La mejor combinación encontrada (`num_leaves=127, n_estimators=250,
min_child_samples=20, max_depth=8, learning_rate=0.1`) mejora sobre su configuración por
defecto (Accuracy=0,811 vs. 0,802, AUC-ROC=0,895 vs. 0,885). Queda **muy cerca** de Random
Forest, perdiendo por apenas 0,002 de Accuracy (0,811 vs. 0,813) pero **ganándole** en AUC-ROC (0,895
vs. 0,894), y sobreajustando considerablemente menos (brecha train-test de 0,092 vs. 0,120 de
Random Forest, sección 9.3), el mismo patrón que en regresión, donde LightGBM también quedaba
cerca del Random Forest con menos sobreajuste. Es la confirmación, con un segundo método, de que
ninguna de las dos grandes familias de ensamble (*bagging* ni *boosting*) encuentra una ventaja
decisiva sobre la otra en este problema, ya que ambas convergen a un desempeño similar, consistente
con que el techo de señal disponible (dominado por el género, sección 6.4) deja poco margen para
que la forma específica del ensamble marque una diferencia grande. Su importancia de variables
refuerza esa lectura, ya que `genero_popularidad_media` domina con **62,2%** de la importancia total
(`importance_type='gain'`), seguida de variables de audio similares entre sí (`acousticness`,
`danceability`, `valence`, `energy`, `loudness`, cada una entre 3,4-4,1%), muy en línea con el
57,3% del Random Forest Classifier (sección 9.2) y con el 79,6% que mostró LightGBM en
regresión (sección 9.1). El *boosting* tiende a concentrarse algo más en género que el
*bagging*, en ambas tareas.

**Nota metodológica, por qué se mantiene esta búsqueda de 8 combinaciones.** Se probó ampliar
esta búsqueda a 24 combinaciones (dos veces, una incluyendo `max_depth=None` en la grilla, otra
excluyéndolo). En ambos casos el resultado generalizó peor en el conjunto de prueba y
sobreajustó más que esta búsqueda original (ver sección 9.3 para el detalle completo), a pesar
de tener un score de validación cruzada apenas distinto (dentro del margen de ruido entre
folds). Es decir, para este modelo en este dataset, varias combinaciones de hiperparámetros
quedan prácticamente empatadas en validación cruzada, y ampliar la búsqueda no garantiza elegir
la que mejor generaliza, solo multiplica el riesgo de elegir una peor por ruido. Por eso se
reporta esta búsqueda de 8 combinaciones como la definitiva, ya que es la que mejor se sostiene contra
el conjunto de prueba de las tres probadas.

**Precision y Recall por clase, en los cuatro modelos** (además de Accuracy/F1/AUC-ROC ya
reportados). Se desglosan porque Accuracy por sí sola puede esconder que un modelo favorezca
una clase sobre la otra, algo que acá sí importa decidir, dado que las dos clases representan
decisiones de negocio distintas (invertir o no esfuerzo de promoción en una canción nueva).

| Modelo | Clase | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| Regresión Logística | alta | 0,7657 | 0,7962 | 0,7806 |
| Regresión Logística | baja | 0,7889 | 0,7576 | 0,7729 |
| Árbol de Decisión | alta | 0,7870 | 0,7968 | 0,7919 |
| Árbol de Decisión | baja | 0,7953 | 0,7855 | 0,7904 |
| LightGBM Classifier | alta | 0,8120 | 0,8071 | 0,8095 |
| LightGBM Classifier | baja | 0,8093 | 0,8141 | 0,8117 |
| **Random Forest Classifier** | **alta** | **0,8148** | **0,8098** | **0,8123** |
| **Random Forest Classifier** | **baja** | **0,8119** | **0,8170** | **0,8144** |

En los cuatro modelos, Precision y Recall quedan similares entre clases (ninguno difiere en más
de 0,04 dentro del mismo modelo), así que ninguno sacrifica una clase por la otra. Para Random
Forest Classifier (el modelo elegido) los cuatro números quedan prácticamente empatados entre
sí (0,81-0,82), sin favorecer "alta" sobre "baja" ni al revés, algo relevante porque un
desbalance aquí significaría, en términos de negocio, perderse canciones realmente
prometedoras (falsos negativos de "alta", que bajarían el recall de esa clase) o invertir
esfuerzo de promoción en canciones que no lo justifican (falsos positivos, que bajarían su
precision). Ninguno de los dos errores domina por sobre el otro.

**Importancia de variables en clasificación** (no reportada en el notebook original, calculada
aparte sobre los modelos ya entrenados, sin reajustar nada). El patrón es el mismo que en
regresión (sección 9.1). En el Random Forest Classifier, `genero_popularidad_media` domina con
**57,3%** de la importancia total, seguida de variables de audio similares entre sí
(`acousticness`, `danceability`, `energy`, `valence`, `loudness`, cada una entre 4-4,5%). En el
Árbol de Decisión la concentración es aún mayor, con **84,4%** en género, consistente con que un
único árbol, al no promediar entre muestras ni variables como el ensamble, depende casi por
completo de la señal más fuerte disponible. Es la misma conclusión del EDA (sección 6.4) y de la
regresión, confirmada con un tercer método (clasificación) y dos algoritmos distintos (árbol y
ensamble). El género es, por lejos, la señal más fuerte del contenido de la canción, independiente
de si la tarea es predecir un puntaje exacto o una categoría binaria.

**Por qué la accuracy (0,81) es tan superior al R² de la regresión (0,44), sin ser un modelo
"mejor".** Se verificó tomando el Random Forest de regresión de la sección 9.1 (ya entrenado,
sin ajustar nada nuevo) y agrupando su propia predicción continua en la misma mediana, lo que da una
accuracy de 0,796, cercana al clasificador entrenado específicamente para esta tarea (0,813).
La diferencia entre ambos no es una señal nueva descubierta por el clasificador, es solo que
optimiza un poco mejor la misma frontera que ya estaba implícita en el regresor. La razón de
fondo es que clasificar en 2 categorías balanceadas es una pregunta mucho más tolerante al
error que acertar el valor exacto, ya que solo hay **una frontera** que cruzar (la mediana), y al
estar justo al medio de la distribución es la configuración más permisiva posible. No es que
el sonido y el género "expliquen mejor" si una canción es popular que cuánto exactamente, es
la misma señal (techo ≈0,44 de R²), vista con una vara de medir distinta.

### 9.3 Validación cruzada y capacidad de generalización de los modelos

Las secciones 9.1 y 9.2 reportan MAE/R²/Accuracy sobre el conjunto de prueba, que ya es una
medida de generalización en sí misma (son datos que ningún modelo vio durante el entrenamiento).
Pero esa única cifra no responde dos preguntas que sí importan para confiar en el número. **¿El
modelo memoriza sus propios datos de entrenamiento en vez de aprender un patrón que generaliza?**
(sobreajuste) y **¿el resultado depende de qué partición específica de los datos se usó para
elegir sus hiperparámetros?** (inestabilidad de la validación). Se responden con dos chequeos
adicionales, ninguno de los cuales cambia los modelos ya entrenados, ya que solo evalúan qué tan
confiable es el número que ya se reportó.

**Chequeo 1, estabilidad de la validación cruzada.** El `RandomizedSearchCV` de las secciones
9.1 y 9.2 usa validación cruzada para elegir hiperparámetros, por una razón adicional a la ya
mencionada (evitar elegir una combinación que solo funciona bien en una partición particular de
los datos). Se instancia explícitamente el divisor de folds en vez de pasar un entero
(`cv=5`). Para los dos clasificadores se usa `StratifiedKFold(n_splits=5, shuffle=True,
random_state=42)` (cada fold mantiene la misma proporción de clases "alta"/"baja" que el
conjunto completo, evitando que un fold quede por azar con una proporción distinta que
distorsione la validación); para los dos regresores, `KFold(n_splits=5, shuffle=True,
random_state=42)` (no hay clases que preservar en un target continuo). Se instancia a mano, en
vez de dejar que scikit-learn lo decida automáticamente a partir de un entero, por una
diferencia concreta. Pasar `cv=5` como entero sí produce el mismo tipo de splitter
(`StratifiedKFold`/`KFold`) para un estimador de clasificación/regresión respectivamente, pero
con `shuffle=False`
por defecto, dejando los folds en el orden original de las filas; `shuffle=True` los mezcla
antes de dividir, para que ningún fold quede sesgado por algún orden residual en los datos, y
`random_state=42` deja esa mezcla fija para que el resultado sea reproducible.

El resultado de esa validación cruzada, para la combinación de hiperparámetros ganadora de
Random Forest, es el siguiente.

| Tarea          | Media de la métrica en los 5 folds | Desviación estándar |
| -------------- | ----------------------------------- | -------------------- |
| Regresión (R²) | 0,4373                             | 0,0086                |
| Clasificación (Accuracy) | 0,8119                  | 0,0026                |

La desviación estándar es baja en ambos casos (2,0% y 0,3% del valor medio). Los 5 folds dan
resultados cercanos entre sí, sin que ningún fold "raro" infle o arrastre el promedio de forma
relevante. Esto importa porque, comparando esa media de validación cruzada contra el resultado
real en el conjunto de prueba (R²=0,4363 en regresión, Accuracy=0,8134 en clasificación, sección
9.1/9.2), ambos números quedan muy cerca (diferencia de 0,0010 y 0,0015 respectivamente), lo que
confirma que el resultado reportado no depende de qué partición específica se usó para elegir
los hiperparámetros, es un desempeño estable y reproducible.

**Chequeo 2, sobreajuste y subajuste (comparación train vs. test).** La validación cruzada del
chequeo 1 evalúa la *elección de hiperparámetros*, no si el modelo *final* (reentrenado sobre el
100% del conjunto de entrenamiento con esos hiperparámetros) memoriza sus propios datos. Para
eso se compara cada modelo contra los datos que ya vio (entrenamiento) y contra los que nunca vio
(prueba). Si el desempeño es parecido en ambos, el modelo aprendió un patrón real; si es mucho
mejor en entrenamiento que en prueba, memorizó en vez de generalizar (sobreajuste); si es
parecido pero bajo en los dos, el modelo es demasiado simple para aprovechar toda la señal
disponible (subajuste).

| Modelo (regresión) | R² entrenamiento | R² prueba | Brecha | Lectura |
| --- | --- | --- | --- | --- |
| Regresión Lineal | 0,394 | 0,387 | 0,007 | **Subajuste.** Similar pero bajo, el modelo lineal no tiene la flexibilidad para capturar más señal, aunque no memoriza. |
| Árbol de Decisión | 0,423 | 0,401 | 0,022 | **Subajuste leve.** Con `max_depth=8` (el óptimo de la búsqueda exhaustiva) el árbol sigue siendo poco profundo a propósito. |
| LightGBM (`RandomizedSearchCV`) | 0,522 | 0,430 | 0,092 | **Sobreajuste leve**, bastante menor que Random Forest, y con un R² de prueba cercano. |
| Random Forest | 0,693 | 0,436 | **0,257** | **Sobreajuste.** Predice con mayor precisión las canciones que ya vio que las nuevas, memorizando parte del entrenamiento. |

| Modelo (clasificación) | Accuracy entrenamiento | Accuracy prueba | Brecha | Lectura |
| --- | --- | --- | --- | --- |
| Regresión Logística | 0,781 | 0,777 | 0,004 | Generaliza bien, sin señales de memorización. |
| Árbol de Decisión | 0,817 | 0,791 | 0,026 | Generaliza razonablemente bien, brecha moderada. |
| LightGBM (`RandomizedSearchCV`) | 0,902 | 0,811 | 0,092 | **Sobreajuste leve**, menor que Random Forest, con Accuracy de prueba muy cercana. |
| Random Forest | 0,933 | 0,813 | **0,120** | **Sobreajuste.** Mismo patrón que en regresión. |

**Por qué esto no cambia la elección del modelo (ver justificación completa en 9.5).** El Random
Forest sobreajusta de forma clara en los dos problemas (brecha de 0,257 en R² y 0,120 en
Accuracy), pero sigue siendo el mejor de los cuatro modelos de regresión y de los cuatro de
clasificación *en el conjunto de prueba*, que es la métrica que de verdad importa para decidir
qué tan bien funcionaría con una canción nueva. Los modelos que no sobreajustan (Regresión
Lineal/Logística, Árbol de Decisión) no lo hacen porque son demasiado simples, no porque
generalicen mejor en términos absolutos. Además, el chequeo 1 ya mostró que el número de prueba
del Random Forest (0,436 / 0,813) queda muy cerca de su score de validación cruzada (0,437 /
0,812), calculado íntegramente sobre particiones de entrenamiento que nunca tocan el conjunto de
prueba, lo que confirma que, pese al sobreajuste visible en la comparación train/test, el número que
efectivamente se reporta y se usa para decidir no está inflado por él.

**Nota metodológica, por qué el Random Forest Classifier usa una búsqueda de 8 combinaciones
en vez de 24.** Se probó ampliar esta búsqueda a 24 combinaciones, dos veces (una incluyendo
`max_depth=None` en la grilla, otra excluyéndolo). En ambas, la combinación "ganadora" por
validación cruzada generalizaba **peor** en el conjunto de prueba y sobreajustaba **más** que
esta búsqueda de 8 combinaciones (brechas de 0,163 y 0,150 respectivamente, ambas por encima del
0,120 reportado acá), a pesar de tener un score de CV apenas distinto (0,0001 a 0,0003 de
diferencia, bien dentro del margen de ruido entre folds, desviación estándar ≈0,0026, sección
Chequeo 1). La causa es que, para este modelo en este dataset, varias combinaciones de hiperparámetros
quedan prácticamente empatadas en validación cruzada (se verificó variando solo
`min_samples_leaf` con el resto fijo, ya que los 5 valores probados dieron scores de CV entre 0,8106 y
0,8120, una diferencia menor que el propio ruido entre folds). Cuando eso pasa, ampliar la
búsqueda no mejora las chances de encontrar la combinación que mejor generaliza, solo multiplica
la oportunidad de que el ruido elija, entre los empatados, uno que generaliza peor (más
capacidad, más propenso a ajustar también el ruido de los folds de validación, no solo la señal
real). Por eso se mantiene esta búsqueda de 8 combinaciones como la reportada, ya que de las tres
probadas, es la que mejor se sostiene contra el conjunto de prueba (ver también sección 9.6).

**Chequeo 3, ¿la optimización de hiperparámetros mejora o empeora el sobreajuste?** La
pregunta del chequeo 2 fue si el Random Forest *ya ajustado* sobreajusta; esta pregunta es
distinta. ¿El propio proceso de búsqueda de hiperparámetros (`RandomizedSearchCV`, secciones
9.1-9.2) empeora el sobreajuste frente a usar la configuración original de scikit-learn, a
cambio de la mejora de desempeño ya reportada? Se responde comparando la brecha train-test del
Random Forest sin ajustar (`n_estimators=100`, sin límite de profundidad) contra la del
Random Forest ajustado, en ambas tareas.

| Tarea | Configuración | Entrenamiento | Prueba | Brecha |
| --- | --- | --- | --- | --- |
| Regresión (R²) | Por defecto | 0,920 | 0,428 | 0,493 |
| Regresión (R²) | Ajustado (`RandomizedSearchCV`) | 0,693 | 0,436 | **0,257** |
| Clasificación (Accuracy) | Por defecto | 1,000 | 0,795 | 0,205 |
| Clasificación (Accuracy) | Ajustado (`RandomizedSearchCV`) | 0,933 | 0,813 | **0,120** |

La respuesta es clara. La optimización **reduce el sobreajuste**, no lo empeora, y mejora el
desempeño en prueba al mismo tiempo, así que no es un trade-off, sino dos mejoras en la misma dirección.
El Random Forest sin ajustar, sin límite de profundidad, literalmente memoriza por completo el
conjunto de entrenamiento en clasificación (Accuracy=1,000, cada hoja termina con una sola
canción), lo que explica una brecha más del doble de grande que la del modelo ajustado. La
búsqueda encontró `max_depth=16` en ambas tareas (en vez de profundidad ilimitada) y
`min_samples_leaf=2` (en vez de 1, su valor original). Limitar cuánto puede crecer cada árbol
individual, aunque sea levemente, no solo mejora la métrica de prueba, sino que reduce
directamente cuánto memoriza cada árbol su propia muestra bootstrap. Confirma que la búsqueda de
hiperparámetros no es solo "buscar el número más alto en test", ya que acá también produjo, como efecto
colateral deseable, un modelo que generaliza mejor.

### 9.4 Aprendizaje no supervisado: segmentación de canciones por sonido (KMeans)

Se agrupan las canciones usando únicamente sus 9 variables de audio ya escaladas (`danceability`,
`energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`,
`tempo`; se excluyen duración, género, variables categóricas y `popularity`, para que el
agrupamiento refleje solo el perfil sonoro, sin ver la variable objetivo ni usar identidad). Es
una generalización no supervisada de la idea manual de "cuadrantes de ánimo" (`valence` ×
`energy`) explorada en el EDA (sección 6.4-6.5).

Se evaluó k entre 2 y 8 con el método del codo (inercia) y el coeficiente de silhouette. La
inercia decrece de forma monótona sin un codo marcado, y el silhouette más alto es con k=2
(0,239, una separación binaria poco accionable); de k=3 en adelante se aplana entre 0,16 y 0,18
sin diferencias relevantes. Se confirmó esta lectura visual con una detección automatizada del
codo (`KneeLocator` de la librería `kneed`, que ajusta la curva y busca el punto de máxima
curvatura), que no encuentra ninguno (`elbow=None`) en el rango evaluado, descartando que exista un
codo matemáticamente inequívoco que simplemente no se vio a ojo. Se eligió **k=4** por ofrecer
una cohesión comparable a valores de k mayores con un número de segmentos manejable para una
decisión de negocio, una decisión metodológica que prioriza interpretabilidad sobre un óptimo
matemático que, en este caso, ni el propio algoritmo de detección puede señalar (ver
limitaciones en 9.6).

Perfil de los 4 segmentos (promedios de audio sin escalar).

| Cluster | Perfil                                                                                                     | n canciones | Popularidad promedio      |
| ------- | ---------------------------------------------------------------------------------------------------------- | ----------- | ------------------------- |
| 0       | Acústico/calmado (`acousticness`≈0,77, `energy`/`danceability` bajos, volumen ≈-15 dB)            | 14.607      | **31,96** (mínima) |
| 1       | Hablado/urbano (`speechiness`≈0,29, notoriamente más alto que el resto)                                | 9.169       | 34,84                     |
| 2       | Pop/dance alegre (`danceability`/`energy`/`valence` altos, `acousticness` y `speechiness` bajos) | 24.401      | **36,76** (máxima) |
| 3       | Intenso/instrumental (`energy` y tempo los más altos, `instrumentalness` elevado, `valence` bajo)   | 18.605      | 33,90                     |

La diferencia de popularidad promedio entre los 4 segmentos es acotada (~5 puntos entre el
segmento más bajo y el más alto, sobre una media global de 34,65), coherente con el hallazgo
central del EDA (sección 6.3), ya que el sonido por sí solo tiene una relación real pero débil con la
popularidad. El segmento
de menor popularidad (acústico, `valence` bajo) es además consistente con `es_calmado_positivo`
(sección 6.5). Ahí el hallazgo fue sobre un cuadrante de *valence* alto/*energy* bajo; aquí,
usando las 9 variables de audio en conjunto, el segmento más penalizado tiene igualmente
*valence* bajo, reforzando que un sonido poco enérgico y poco positivo se asocia con menor
tracción, más allá de qué par de variables se use para definirlo.

**Contraste exploratorio, clustering ponderado por importancia de popularidad (no
independiente).** El clustering de arriba es deliberadamente independiente de `popularity` (no
la usa para nada), lo que le da fuerza al hallazgo de que separa popularidad sin haberla
mirado. Como ejercicio aparte, se prueba qué pasa si en vez de tratar las 9 variables de audio
por igual, se les da más peso (proporcional a su importancia en el Random Forest de regresión,
sección 9.1) a las que más se relacionan con `popularity`. Esto **sí** inyecta información del
target en la construcción de los clusters, así que no reemplaza al clustering independiente de
arriba, solo contrasta contra él.

| | Clustering independiente (actual) | Clustering ponderado por importancia |
| --- | --- | --- |
| Rango de popularidad entre el segmento más bajo y más alto | 4,80 puntos | **6,95 puntos** |

El clustering ponderado separa la popularidad entre segmentos de forma más marcada (6,95 vs.
4,80 puntos), algo esperable, porque ya se le dio de antemano más peso a las variables que se sabía
que importaban para `popularity`. El `Adjusted Rand Index` entre ambos clustering es 0,4692
(moderadamente distinto, ni idéntico ni al azar), lo que muestra que la ponderación sí reordena de forma
apreciable qué canciones quedan en cada grupo. Este resultado no debe leerse como una
confirmación independiente adicional (es, por construcción, más probable que separe
popularidad), sino como evidencia de que, si se acepta usar información de popularidad en el
proceso, los segmentos de sonido pueden volverse más accionables para ese fin específico.

**Clustering separado dentro de canciones "alta" y "baja" popularidad.** Otra forma de
explorar el sonido sin romper la independencia. En vez de pesar variables por importancia, se
**separan las canciones según su etiqueta real** (`popularity` por encima o por debajo de la
mediana, ya conocida, no predicha) y se agrupan por sonido **dentro de cada mitad por
separado**, sin usar `popularity` para nada en ese paso, exactamente la misma lógica del
clustering principal, solo aplicada a cada mitad.

Dentro de "alta" (32.525 canciones) aparecen 4 perfiles de sonido distintos, todos con
popularidad similar entre sí (48-52 puntos), acústico/instrumental calmado, pop/dance
energético, energético con vibra en vivo, y bailable con voz marcada. No hay una única
"fórmula sonora" del éxito, ya que varios caminos sonoros muy distintos funcionan igual de bien.

Dentro de "baja" (34.257 canciones) aparece el hallazgo más útil, un grupo de **12.781
canciones** (el más grande de las cuatro) con un perfil de sonido **prácticamente idéntico**
al pop/dance energético que sí triunfa dentro de "alta" (`danceability` y `energy` altos,
`valence` medio-alto), pero con popularidad promedio de solo 19,23, muy por debajo de los
51,97 de su equivalente exitoso. Como el sonido de estas canciones no las distingue de las que
sí funcionaron, lo que las frenó fue algo ajeno al audio (promoción, reputación del artista,
timing de lanzamiento). Son el grupo más concreto de candidatas con sonido ya comprobado que
no explotó por razones fuera del alcance de este modelo.

### 9.5 Selección de la solución analítica óptima, evaluación de resultados y recomendaciones de negocio

**Selección y justificación de la solución analítica óptima.** De los diez modelos
supervisados entrenados (seis de regresión, cuatro de clasificación complementaria), el Random
Forest es la solución elegida en ambos casos, por tres razones que se sostienen juntas, no por
separado.

1. **Es el mejor en la métrica que de verdad importa (el conjunto de prueba)**, comparado
   directamente contra las otras alternativas entrenadas sobre los mismos datos. R²=0,436
   frente a 0,430 (LightGBM, también ajustado con `RandomizedSearchCV`), 0,401 (Árbol de
   Decisión) y 0,387 (Regresión Lineal, Ridge y Lasso, las tres prácticamente empatadas) en
   regresión; Accuracy=0,813 frente a 0,811 (LightGBM Classifier), 0,791 y 0,777 en
   clasificación (secciones 9.1-9.2). No es una mejora marginal en regresión, ya que el ensamble
   aporta una mejora de +8,7% de R² sobre el árbol individual, y le gana a los dos ensambles
   de *boosting* evaluados (LightGBM en ambas tareas) a pesar de que ambos también fueron
   optimizados (secciones 9.1-9.2), aunque en clasificación la ventaja sobre LightGBM es chica
   (+0,002 de Accuracy) y este último incluso le gana en AUC-ROC (0,895 vs. 0,894) y sobreajusta
   bastante menos (sección 9.3).
2. **Su ventaja no es un artefacto de sobreajuste ni de una partición de validación
   particular**, verificado en la sección 9.3. El score de validación cruzada (que nunca toca el
   conjunto de prueba) queda muy cerca del resultado real en test, y es razonablemente estable
   entre folds (desviación estándar ≈2,0% en regresión, ≈0,3% en clasificación). La sección 9.3
   también documenta por qué se mantuvo la búsqueda original de 8 combinaciones para el Random
   Forest Classifier en vez de ampliarla. Dos intentos de ampliarla a 24 combinaciones
   encontraron candidatos con un score de CV apenas distinto (dentro del margen de ruido), pero
   que generalizaban peor contra el conjunto de prueba, precisamente el tipo de verificación
   que este chequeo está diseñado para habilitar, y la razón por la que no se adoptó la búsqueda
   ampliada solo por "verse mejor" en la validación cruzada.
3. **Es la configuración que más aporta al objetivo de negocio de la sección 1**, ya que al depender
   menos de una sola variable que los modelos más simples (61,5% de la importancia en género,
   vs. 94,2% del Árbol de Decisión, sección 9.1), captura mejor la señal residual de las variables
   de audio, que es justamente la parte del problema que no se puede resolver con la fama ya
   instalada de un artista, el objetivo explícito del proyecto.

La Regresión Lineal/Logística y el Árbol de Decisión no se descartan por ser "peores" en
abstracto. Se mantienen en la comparación porque son el punto de referencia que permite
cuantificar cuánto aporta cada nivel de complejidad (lineal → árbol único → ensamble), y porque
su mayor interpretabilidad (un coeficiente por variable, o un único camino de decisiones) sigue
siendo valiosa si en el futuro se necesita explicar una predicción individual a alguien no
técnico, algo que un ensamble de 300 árboles no puede hacer con la misma simplicidad.

Retomando la pregunta de negocio de la sección 1, los tres modelos (regresión, clasificación y
clustering) confirman de forma cuantitativa y extremo a extremo lo que el EDA ya anticipaba con
correlaciones simples. Sonido y género explican una fracción real pero acotada de la
popularidad (R²≈0,44 en regresión; ~5 puntos de rango entre segmentos de sonido), con el género
como la señal individual más fuerte en todos los enfoques. La clasificación binaria (accuracy
0,81) no contradice este techo (es la misma señal, vista con una pregunta más tolerante al
error, sección 9.2) y es, de los tres, la más directamente accionable para una decisión de
negocio binaria ("¿invertir esfuerzo de promoción en esto o no?").

**Recomendación de negocio.** Usar el modelo (y los segmentos de sonido) como señal de apoyo para
priorizar qué escuchar primero entre material nuevo de artistas sin trayectoria (el caso de uso
real de la sección 1), nunca como filtro automático ni sustituto del criterio humano. Con
R²≈0,44, más de la mitad de la varianza de `popularity` queda sin explicar por el contenido de
la canción, y parte de esa varianza no observada corresponde a contexto de publicación (sección
6.1) y a la fama del artista, excluida deliberadamente del modelo por las razones éticas de la
sección 7. El modelo responde "qué tan prometedora suena esta canción en sí misma", no "qué tan
exitosa será", ya que son preguntas distintas y el negocio debe tratarlas como tales.

**Qué error de clasificación le cuesta más al negocio (y por qué el umbral de decisión no
debería subirse por encima de 0,5).** El clasificador de la sección 9.2 usa, como cualquier
clasificador binario, un umbral de probabilidad para decidir "alta" o "baja" (0,5 por defecto).
Subir ese umbral (ej. a 0,60, como se consideró en una entrega anterior) reduce los Falsos
Positivos (canciones marcadas "alta" que en realidad no lo eran) a costa de más Falsos
Negativos (canciones realmente populares marcadas "baja"). Para este problema de negocio
específico, el Falso Negativo es el error más costoso, por tres razones que se sostienen
juntas.

1. **El objetivo del proyecto es corregir una asimetría que hoy perjudica a artistas nuevos**
   (sección 1). La priorización actual depende de la reputación ya instalada, lo que
   sistemáticamente descarta a quienes no la tienen. Si el modelo que se construye para
   corregir eso termina generando muchos Falsos Negativos, reproduce el mismo resultado práctico
   que el problema original (una canción de un artista nuevo queda descartada sin oportunidad
   real), solo que la causa pasa de ser la reputación a ser el límite del modelo.
2. **Los dos errores no son igual de recuperables.** Un Falso Positivo cuesta el tiempo de un
   humano escuchando una canción que no resultó tan buena (un recurso barato y ya asignado a
   esta tarea). Un Falso Negativo significa que una oportunidad real puede perderse de forma
   permanente, ya que con un flujo constante de material nuevo, una canción deprioritizada puede no
   llegar nunca a revisión humana.
3. **La señal ya es débil** (R²≈0,44, sin identidad de artista), así que exigirle más certeza al modelo
   antes de decir "alta" (un umbral más alto) solo agrava cuántas canciones genuinamente buenas
   quedan del lado equivocado de una incertidumbre que ya es considerable de por sí.

Por estas razones, el umbral se mantiene en 0,5 (el valor por defecto usado en toda la sección
9.2) en vez de subirlo. De hecho, verificado directamente sobre el Random Forest Classifier, el
umbral de 0,5 es el que maximiza la Accuracy total (0,8134, contra 0,8054 con umbral=0,4 y
0,7998 con umbral=0,6) porque el modelo está razonablemente bien calibrado (la probabilidad
predicha cruza el 50% justo donde la mayoría real de las canciones pasa a ser "alta"), así que
cualquier desvío de 0,5 sacrifica Accuracy agregada a cambio de mover el balance entre ambos
tipos de error. Si el negocio decidiera que evitar Falsos Negativos vale ese costo (consistente
con los tres puntos anteriores), bajar el umbral (ej. a 0,4, con lo que los Falsos Negativos caen de 1.584 a
995, a cambio de que los Falsos Positivos suban de 1.532 a 2.254) sería la dirección correcta,
nunca subirlo.

### 9.6 Limitaciones del modelado

- **No se usó remuestreo (SMOTE o similar) en la clasificación, deliberadamente.** SMOTE existe
  para corregir el desbalance de clases generando ejemplos sintéticos de la clase minoritaria;
  acá no se justifica porque no hay desbalance que corregir. La partición por la mediana deja
  32.525 canciones "alta" y 34.257 "baja" en el conjunto de entrenamiento (48,7%/51,3%, razón
  mayoría/minoría ≈1,05), y el propio EDA ya había verificado esto antes de modelar (sección 6.1).
  Aplicar SMOTE sobre un dataset ya balanceado no mejoraría el modelo y añadiría ejemplos
  sintéticos innecesarios al entrenamiento, por eso se omite a propósito en vez de omitirse por
  descuido.
- **Justificación de usar `GridSearchCV` para el Árbol de Decisión y `RandomizedSearchCV` para
  los ensambles, y por qué esta diferencia no invalida la comparación entre modelos.** Cada
  modelo usa la técnica
  de búsqueda más exhaustiva que es computacionalmente viable para el tamaño de su propio
  espacio de hiperparámetros, que difiere mucho entre modelos.

  | Modelo | Espacio total de combinaciones | Combinaciones probadas | % cubierto |
  | --- | --- | --- | --- |
  | Árbol de Decisión | 49 (7×7) | 49 (`GridSearchCV`) | **100%** |
  | Ridge / Lasso | 9-10 (solo `alpha`) | 8 (`RandomizedSearchCV`) | ~80-89% |
  | Random Forest (regresión) / LightGBM | 600 (5×6×5×4) | 24 (`RandomizedSearchCV`) | 4% |
  | Random Forest Classifier | 600 (5×6×5×4) | 8 (`RandomizedSearchCV`) | 1,3% |

  Con solo 49 combinaciones posibles, probarlas todas para el Árbol de Decisión es barato
  (49×5 folds=245 ajustes de un solo árbol); para un ensamble de hasta 300 árboles, probar las
  600 combinaciones completas (600×5=3.000 ajustes) sería demasiado lento para esta entrega, por
  eso se usa una muestra al azar. Esto significa que, si hay una asimetría en la comparación, va
  **en contra** de los ensambles, no a su favor. El Árbol de Decisión exploró el 100% de su
  espacio mientras que Random Forest exploró apenas 1,3-4% del suyo, y aun así Random Forest le
  gana en las dos tareas (R²=0,436 vs. 0,401 en regresión; Accuracy=0,813 vs. 0,791 en
  clasificación, secciones 9.1-9.2). Que el modelo elegido gane pese a una búsqueda
  relativamente más restringida que la de su competidor más cercano es evidencia más fuerte a
  favor de la elección, no más débil.
- La búsqueda de hiperparámetros para los Árboles de Decisión (regresión y clasificación) se
  amplió a `GridSearchCV` **exhaustivo** (las 49 combinaciones posibles de su espacio, sección
  9.1), y la de Random Forest de regresión/LightGBM de `RandomizedSearchCV` con 8 a 24
  combinaciones de un espacio de 600, una cobertura más amplia que la búsqueda original, pero
  todavía parcial (4% del espacio total) para estos dos últimos, ya que probarlo completo sería
  computacionalmente muy costoso; una búsqueda aún más amplia (`GridSearchCV` sobre el entorno
  de la mejor combinación encontrada, o más iteraciones) sigue quedando como trabajo futuro
  (sección 10). El Random Forest Classifier es la excepción deliberada, ya que se mantuvo en 8
  combinaciones (ver bullet siguiente).
- **Ampliar la búsqueda de 8 a 24 combinaciones no mejoró todo por igual.** En regresión
  encontró una mejor combinación tanto para el Árbol de Decisión como para LightGBM, pero para
  el Random Forest Classifier se probó dos veces (con y sin `max_depth=None` en la grilla) y en
  ambos casos el resultado generalizó *peor* en el conjunto de prueba y sobreajustó *más* que la
  búsqueda original de 8 combinaciones (brechas de 0,163 y 0,150, ambas por encima del 0,120
  reportado en las secciones 9.2/9.3), a pesar de tener scores de validación cruzada apenas
  distintos (diferencias de 0,0001 a 0,0003, bien dentro del margen de ruido entre folds, sección
  9.3). Por eso se mantiene la búsqueda original de 8 combinaciones para este modelo, ya que de las
  tres probadas, es la que mejor se sostiene contra el conjunto de prueba. Es evidencia concreta
  de que una búsqueda más amplia no es gratuita ni estrictamente mejor. Cuando varias
  combinaciones quedan empatadas en CV, ampliar la búsqueda solo multiplica la oportunidad de
  que el ruido elija una que generaliza peor, un riesgo a tener presente al decidir cuánto
  ampliar una búsqueda, no solo "más es mejor".
- El Random Forest sobreajusta de forma medible en ambos problemas (brecha train-test de 0,257
  en R² y 0,120 en Accuracy, sección 9.3); se mantiene como la solución elegida porque ese
  sobreajuste no infla el número reportado (verificado contra la validación cruzada, sección
  9.3) y sigue siendo superior a las alternativas en el conjunto de prueba, pero una búsqueda de
  hiperparámetros más agresiva contra el sobreajuste (ej. `max_depth` más bajo, `min_samples_leaf`
  más alto) es una mejora posible no explorada en esta entrega.
- El techo de R²≈0,44 refleja el límite del contenido+género para explicar `popularity`, no
  necesariamente el límite del algoritmo. Coincide con que la señal más fuerte medida (identidad
  de artista/álbum, r≈0,63/0,72 respectivamente, sección 7) fue deliberadamente excluida por razones éticas. Un
  modelo que la incluyera llegaría a un R² mucho mayor, pero respondería la pregunta de negocio
  equivocada.
- La segmentación por sonido (k=4) no es la óptima según silhouette puro (k=2 tiene mayor
  cohesión, 0,239 vs 0,166 de k=4); se privilegió un número de segmentos más accionable para el
  negocio, una decisión metodológica explícita, no el resultado matemáticamente óptimo. Además,
  el valor absoluto de silhouette en k=4 (0,1663) cae bajo el umbral de 0,25 que suele
  considerarse "sin estructura sustancial" (escala de Kaufman & Rousseeuw). Los 4 segmentos
  reflejan diferencias de sonido reales e interpretables, pero sus límites no son estadísticamente
  nítidos, es un continuo, no categorías tajantes.

## 10. Estado del proyecto y próximos pasos

Este proyecto cubre comprensión del negocio, comprensión de los datos, preparación de
datos, modelado y evaluación (fases 1-5 de CRISP-DM). Queda explícitamente fuera de alcance, como
trabajo futuro, lo siguiente.

- **Despliegue** (fase 6). Servir el modelo (por ejemplo, una API), monitoreo de desempeño en
  producción y reentrenamiento periódico, junto con los artefactos serializados del pipeline y
  del modelo elegido.
- **Búsqueda de hiperparámetros más exhaustiva** (`GridSearchCV` sobre el entorno de la mejor
  combinación ya encontrada, o más iteraciones de `RandomizedSearchCV`), más allá de la búsqueda
  acotada de la sección 9, tanto para el Random Forest de regresión elegido como para LightGBM
  (sección 9.1), que ya ampliaron su búsqueda de 8 a 24 combinaciones de un espacio de 600, pero
  siguen cubriendo solo una fracción chica (4%) de todas las posibles. Para el Random Forest
  Classifier, en cambio, se probó ampliar la búsqueda y **se descartó**, ya que dos intentos con 24
  combinaciones generalizaron peor que la búsqueda original de 8 (sección 9.3/9.6), así que una
  búsqueda más amplia para este modelo específico no es prioritaria. Lo que sí queda pendiente
  es investigar *por qué* el paisaje de validación cruzada es tan plano para este modelo en
  particular (varias combinaciones quedan empatadas dentro del margen de ruido, sección 9.3),
  posiblemente con más folds o repeticiones de la validación cruzada para reducir ese ruido.

## 11. Matriz de responsabilidades del equipo

Tareas a cargo de cada integrante, según el historial de commits del repositorio.

| Tarea                                                                                                                                                                                                                | Responsable               |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------- |
| Configuración inicial del repositorio (estructura de carpetas, dataset, README base, `requirements.txt`)                                                                                                          | Felipe Ahumada Silva      |
| Primera versión de `notebooks/analisis_exploratorio.ipynb` (EDA)                                                                                                                                                  | Felipe Ahumada Silva      |
| Primera versión de `notebooks/preprocesamiento.ipynb`                                                                                                                                                             | Felipe Ahumada Silva      |
| Primera versión de `docs/informe.md`                                                                                                                                                                              | Felipe Ahumada Silva      |
| Correcciones de rutas y ajustes del README                                                                                                                                                                           | Felipe Ahumada Silva      |
| Ajustes y refinamiento del informe                                                                                                                                                                                   | Felipe Ahumada Silva      |
| Corrección de inconsistencias detectadas en el EDA, el preprocesamiento y el informe                                                                                                                                | Francisca Carrasco Lozano |
| Corrección final del informe                                                                                                                                                                                        | Francisca Carrasco Lozano |
| Corrección de enlaces del informe, justificación de herramientas colaborativas y cierre de la primera entrega                                                                                                      | Felipe Ahumada Silva      |
| Modelado (`notebooks/modelamiento.ipynb`: regresión, clasificación complementaria y clustering), actualización del informe (sección 9) y corrección de inconsistencias detectadas en la preparación de datos | Francisca Carrasco Lozano |
