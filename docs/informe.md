# Inteligencia Musical: Predicción de Popularidad de Canciones (Spotify Tracks)

**Evaluación Parcial N°1 y N°2 · MLY1101 Machine Learning · Duoc UC**
**Caso de Estudio C:** Inteligencia musical y predicción de popularidad de canciones (Spotify Tracks)
**Por:** Felipe Ahumada Silva y Francisca Carrasco Lozano

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
priorización depende en gran medida de la reputación ya instalada del artista, lo que perjudica sistemáticamente a artistas nuevos o poco conocidos: sin
trayectoria previa, no queda más señal sobre la canción en sí.

El dataset **Spotify Tracks Dataset** (ver [Fuentes de datos](#4-descripción-de-las-fuentes-de-datos-y-herramientas-colaborativas))
contiene más de 100.000 canciones con sus características de audio (danceability, energy,
loudness, tempo, entre otras), su género y su métrica de popularidad en Spotify. El problema de
negocio que aborda este proyecto es: **¿qué tan bien explican los atributos propios de una
canción su sonido y su género la popularidad que alcanza, independientemente de la fama de
quien la interpreta?**

Es una pregunta exploratoria, no una promesa de resultado, y el propio análisis (sección 6) la
responde de forma más matizada de lo esperado: las características de audio por sí solas
explican muy poco de la popularidad (correlación máxima de apenas -0,13), el género aporta una
señal bastante más fuerte pero es en parte un artefacto del muestreo con que se construyó el
dataset (sección 6.4), y la señal individual más fuerte de todas, la identidad del artista o
álbum, se descarta deliberadamente pese a ser la más predictiva, por las razones éticas
detalladas en la sección 7: usarla resolvería el problema estadístico, pero no el de negocio,
porque reproduciría exactamente el sesgo de reputación que se quiere evitar. Que "el sonido de
una canción por sí solo prediga poco su éxito" es en sí mismo un hallazgo relevante para el
negocio: matiza cualquier expectativa de que un futuro modelo basado en audio y género pueda
reemplazar esto, y deja claro que su aporte es el de una señal adicional, acotada,
no un reemplazo del criterio humano. La [sección 9](#9-modelado-construcción-comparación-y-evaluación-de-modelos) cierra esta pregunta
con modelos entrenados extremo a extremo: el mejor modelo supervisado alcanza R²≈0,44
combinando sonido y género, un techo consistente con lo que el EDA ya anticipaba con
correlaciones simples, no una revisión del hallazgo.

## 2. Objetivos del proyecto

1. **Comprender la calidad y estructura del dataset**: identificar y resolver problemas de
   duplicación, valores inválidos y consistencia antes de sacar cualquier conclusión sobre
   popularidad.
2. **Explorar qué atributos se asocian con la popularidad de una canción**: de audio, de
   género y de contexto de publicación (release), cuantificando la fuerza (o debilidad) de esa
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

Los objetivos 1, 2, 3 y 4 se desarrollan en las secciones 6 y 7 (EDA y preparación de datos). El
objetivo 5 (modelado, fases 4-5 de CRISP-DM) se desarrolla en la
[sección 9](#9-modelado-construcción-comparación-y-evaluación-de-modelos). El despliegue
(fase 6) sigue fuera de alcance, ver
[Estado del proyecto](#10-estado-del-proyecto-y-próximos-pasos).

## 3. Definición de KPIs

Los KPIs se dividen en tres grupos: los que responden directamente la pregunta de negocio de
la sección 1 (medibles ya desde el EDA, sin necesidad de un modelo entrenado), los que miden la
calidad de los datos como condición necesaria para que los primeros sean confiables, y los que
miden el desempeño del modelado (sección 9): estos últimos sí requieren los modelos
entrenados y por eso se definen con su resultado directamente en esa sección, no aquí.

### KPIs de negocio

Responden la pregunta de la sección 1: ¿qué tan bien explica la canción misma su popularidad,
sin depender de la fama de quien la interpreta?

| KPI | Meta | Resultado |
|---|---|---|
| Techo de señal explicable por el contenido de la canción (sin usar identidad de artista/álbum) | Cuantificar cuánto puede llegar a explicar, en el mejor caso, un modelo que solo use sonido y género | Atributos de audio solos: correlación máxima de apenas **-0,13** (instrumentalness). Género (codificado adecuadamente): **0,62**, muy superior a cualquier atributo de audio individual. Este 0,62 es la cota realista de lo que el contenido de la canción puede explicar sin recurrir a quién la interpreta: fija expectativas honestas para la futura fase de modelado, antes de invertir en entrenarlo. |
| Features basadas en identidad de artista/álbum en el modelo final | 0 | Se midió que artista/álbum predicen popularidad mucho mejor que cualquier otro atributo (r≈0,63 artista, 0,72 álbum), pero se excluyen deliberadamente para no reforzar el sesgo de "el que ya es famoso, sigue siendo detectado como popular" (ver sección 7). |
| % del catálogo cuya baja popularidad es un artefacto del contexto de publicación, no de la calidad de la canción | Cuantificar para no confundir "canción poco atractiva" con "dato mal catalogado" en decisiones de negocio | **54%** del pico de canciones con popularity = 0 (5.088 de 9.413) corresponde a duplicados de release (el mismo audio con otro release sin tracción), no a que la canción en sí no guste (ver sección 6.1). Sin este KPI, el negocio podría descartar por error contenido que en realidad sí tiene tracción, solo mal catalogado. |

### KPIs de calidad de datos (soporte)

Condición necesaria para que los KPIs de negocio de arriba sean confiables, no miden el
problema de negocio en sí:

| KPI | Meta | Resultado |
|---|---|---|
| Duplicados identificados y resueltos | 100% | 24.259 filas por track_id repetido (mismo audio, distinto género) + 9.284 filas de audio idéntico bajo track_id distinto, el 100% consolidado, sin filas duplicadas remanentes. |
| Valores inválidos identificados y tratados | 100% | 129 time_signature = 0 / 124 tempo = 0 (sobre el dataset ya consolidado) marcados como faltantes para imputación; 0 valores inválidos sin tratar en el dataset preparado. |
| Dataset final sin nulos tras la preparación | 0 nulos | X_train_prep / X_test_prep: 0 valores nulos, verificado. |

### KPIs de modelado

Miden el desempeño de los modelos entrenados en [`notebooks/modelamiento.ipynb`](../notebooks/modelamiento.ipynb)
(ver detalle y justificación en la sección 9):

| KPI | Meta | Resultado |
|---|---|---|
| Desempeño del mejor modelo supervisado (regresión) frente a los otros dos | Mejor MAE y R² de los tres | Random Forest (`RandomizedSearchCV`): **R²=0,438**, MAE=9,98 (vs. Árbol de Decisión R²=0,393, MAE=10,67, y Regresión Lineal R²=0,387, MAE=10,75). |
| Desempeño del mejor modelo supervisado (clasificación complementaria) frente a los otros dos | Mejor Accuracy, F1 y AUC-ROC de los tres | Random Forest Classifier (`RandomizedSearchCV`): **Accuracy=0,813**, AUC-ROC=0,894 (vs. Árbol de Decisión Accuracy=0,789, AUC-ROC=0,866, y Regresión Logística Accuracy=0,777, AUC-ROC=0,849). |
| Segmentación no supervisada de canciones por sonido con diferencias de popularidad observables | Identificar segmentos de sonido con distinta popularidad promedio | KMeans (k=4): 4 segmentos de sonido con popularidad promedio distinta, rango acotado de ~5 puntos entre el segmento más bajo (31,96) y el más alto (36,76). |

## 4. Descripción de las fuentes de datos y herramientas colaborativas

**Fuente de datos:** [`data/Spotify_Tracks_Dataset.csv`](../data/Spotify_Tracks_Dataset.csv): 114.000
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

**Herramientas colaborativas y de trabajo:** el equipo se coordinó de forma presencial para
acordar avances, repartir el trabajo y discutir los hallazgos antes de incorporarlos al informe.

- **Git y GitHub**: control de versiones y repositorio compartido. Es la herramienta central del
  trabajo colaborativo del proyecto: mantiene el historial de commits del informe y de ambos
  notebooks, permite revisar qué cambió entre versiones de un análisis, y actúa como fuente única
  de verdad para el equipo. Se eligió sobre alternativas de carpeta compartida porque los
  notebooks de Jupyter son archivos JSON grandes: sin control de versiones, dos personas editando
  en paralelo se sobrescriben sin dejar rastro.
- **Notebooks autocontenidos**: decisión de trabajo derivada de lo anterior: cada notebook
  reconstruye su propio dataset desde el CSV crudo, sin depender de haber ejecutado el otro. Esto
  permite que dos personas trabajen sobre distintas etapas del proyecto en paralelo y que
  cualquiera pueda ejecutar y verificar el trabajo del otro sin coordinación previa.
- **Jupyter Notebook**: entorno de análisis; el código, su salida y la justificación en Markdown
  quedan juntos en el mismo archivo, que es lo que permite que el trabajo sea revisable por otra
  persona sin explicación adicional.
- **Documentación de Spotify** ([`spotify_dataset.pdf`](spotify_dataset.pdf)): referencia
  compartida de definiciones y rangos válidos de cada columna. Fija un criterio común: sin ella,
  cada integrante decidiría por su cuenta qué valor es "inválido".

## 5. Metodología utilizada (CRISP-DM)

El proyecto sigue la metodología **CRISP-DM** (Cross-Industry Standard Process for Data Mining),
de 6 fases iterativas. Este proyecto cubre las primeras cinco:

1. **Comprensión del negocio**: cubierta en las secciones 1-3 de este informe: problema,
   objetivos y KPIs.
2. **Comprensión de los datos**: cubierta por [`notebooks/analisis_exploratorio.ipynb`](../notebooks/analisis_exploratorio.ipynb):
   carga, calidad, duplicados, valores inválidos, outliers, correlaciones, distribución del
   target.
3. **Preparación de los datos**: cubierta por [`notebooks/preprocesamiento.ipynb`](../notebooks/preprocesamiento.ipynb):
   consolidación adicional, tratamiento de inválidos, ingeniería de características, pipeline de
   transformación.
4. **Modelado**: cubierta por [`notebooks/modelamiento.ipynb`](../notebooks/modelamiento.ipynb):
   tres modelos supervisados de regresión (Regresión Lineal, Árbol de Decisión, Random Forest),
   tres modelos supervisados de clasificación como complemento (Regresión Logística, Árbol de
   Decisión, Random Forest Classifier) y una técnica no supervisada (KMeans) de segmentación por
   sonido, ver sección 9.
5. **Evaluación**: cubierta en la sección 9, comparación de modelos con métricas de regresión
   (MAE, R²) y de clasificación (accuracy, F1, AUC-ROC), perfilado de los segmentos obtenidos
   por clustering e interpretación de resultados en términos del problema de negocio.
6. **Despliegue**: fuera de alcance, se define como trabajo futuro (ver sección 10).

## 6. Preparación y análisis exploratorio de los datos (EDA)

Resumen de los hallazgos y decisiones de [`analisis_exploratorio.ipynb`](../notebooks/analisis_exploratorio.ipynb)
y [`preprocesamiento.ipynb`](../notebooks/preprocesamiento.ipynb), ambos notebooks son
autocontenidos, están documentados en español celda a celda, y fueron verificados ejecutándolos
de punta a punta (no solo escritos y asumidos correctos).

### 6.1 Calidad de datos: dos niveles de duplicación

El dataset crudo (114.000 filas) tiene **24.259 filas repetidas por track_id**: la misma
canción aparece una vez por cada género bajo el que Spotify la cataloga. Se consolida con
groupby('track_id'), uniendo los géneros con ; (mismo criterio que ya usa artists para
colaboraciones) y promediando popularity (variaba en apenas 4,33% de los casos). Resultado:
**89.740 canciones únicas**. Se elimina además 1 fila con `duration_ms = 0` y metadatos nulos
(error de carga).

El EDA identifica un **segundo nivel de duplicación que la consolidación por track_id no resuelve**: track_id identifica un *release* (álbum, single, recopilatorio), no la canción en
sí. Agrupando por las 13 columnas de audio se encuentran **9.284 filas (10,35%) en 3.022 grupos**
de audio idéntico bajo distinto track_id, reediciones o recopilatorios de la misma grabación.
El caso más extremo, la canción *"I'm Good (Blue)"*, tiene popularity = 0 en un release y 98
en otro para el mismo audio exacto. En 466 de esos grupos el rango de popularity es ≥30 puntos.
Esto es relevante porque explica parte del pico de canciones con popularity = 0 (9.413 filas,
10,49% del dataset): el 54% de esas filas (5.088) pertenece a estos grupos de audio duplicado, y
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
**33,0**: 50,32% de las filas queda bajo la mediana (baja) y 49,68% sobre ella (alta), una razón
mayoría/minoría de **1,01**. Es decir, prácticamente sin desbalance, esperable al partir una
variable continua por su propia mediana, lo que respalda usar accuracy sin remuestreo en la
clasificación de la sección 9.2.

### 6.2 Valores inválidos

Contra los rangos documentados por Spotify, se encuentran tres inconsistencias reales (sobre el
dataset ya consolidado por audio, 83.478 filas):

- **tempo = 0** (124 filas): ninguna canción tiene 0 BPM, es físicamente imposible.
- **time_signature = 0** (129 filas): 96% de estas filas también tiene tempo = 0, y el 81%
  pertenece al género sleep, es el mismo fallo del algoritmo de Spotify al no poder estimar
  pulso musical en sonidos ambientales, no un compás real.
- **loudness > 0** (66 filas, hasta 4,53 dB): másters muy comprimidos que superan el techo
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
(energy-loudness: 0,76; acousticness vs. energy/loudness: -0,73/-0,58), pero el hallazgo
central es que **ninguna variable de audio correlaciona fuertemente con popularity**: la más
alta es instrumentalness con apenas **-0,13**. La popularidad no se explica linealmente por
cómo suena la canción.

### 6.4 Género: la señal más fuerte, con una trampa de diseño

track_genre es multietiqueta (114 géneros posibles, separados por `;`: el 82% de las canciones
tiene un solo género, promedio 1,27). Su distribución (~1.000 canciones por género) no es
orgánica: es la huella del muestreo estratificado con que se construyó el dataset, no refleja
popularidad real de cada género en Spotify, una limitación a tener presente si se usa como
predictor (ver sección 7).

Pese a eso, un chequeo de la popularidad promedio por género muestra la relación más fuerte
encontrada en todo el dataset: pop promedia 47,9 de popularidad, grindcore 14,6, un rango de
más de 3x, y muy superior a cualquier variable de audio. Por eso se incorpora como predictor (ver
6.5) en vez de descartarla.

### 6.5 Ingeniería de características y pipeline de preparación

Se evaluaron y descartaron varias transformaciones (interacciones entre variables de audio, todas
con |r| < 0,06) y se retuvieron dos, con señal verificada:

- **distancia_duracion_optima** (`|duración_min − 3,75|`): la popularidad sube y baja en forma
  de U invertida según la duración (pico 34,4-35,5 entre 3-4 min, mínima en canciones muy cortas
  o muy largas), relación que duration_ms en bruto no captura por ser lineal
  (r=-0,02 vs. r=-0,09 con la distancia).
- **es_calmado_positivo** (valence ≥ 0,5 y energy < 0,5): el único de los 4 cuadrantes del
  modelo de ánimo (valence×energy) cuya popularidad real queda sistemáticamente por debajo
  (-2,84 puntos) de lo que predice un modelo lineal simple con valence y energy por separado.

Para incorporar track_genre se evaluaron tres opciones: multi-hot completo (~114 columnas),
agrupación manual en familias, o codificación por promedio de género (*mean/target encoding*). Se
eligió la última (GenreMeanEncoder, transformador personalizado): el promedio de popularidad
histórica del/los género(s) de la canción, calculado **solo con el conjunto de entrenamiento**
para no filtrar información hacia el conjunto de prueba. La consolidación por audio (6.1) mejora
además esta señal: la correlación de GenreMeanEncoder con popularity en el conjunto de prueba
sube de 0,568 a **0,618**.

El pipeline final (scikit-learn Pipeline + ColumnTransformer) combina: Winsorizer
(recorte de outliers) → SimpleImputer → StandardScaler para las 11 variables numéricas;
SimpleImputer → OneHotEncoder para las 5 categóricas; GenreMeanEncoder → StandardScaler
para el género; y un CorrelationFilter final (umbral 0,9) como resguardo de multicolinealidad.

El recorte de Winsorizer no es uniforme: se calibró un límite propio por columna según el %
real de outliers medido con la regla IQR (1,5× el rango intercuartílico), en vez de aplicar el
mismo 5%/5% a las 11 columnas por igual. Por ejemplo, energy, acousticness y valence no tienen
outliers reales (0% ambos lados) y no se recortan; instrumentalness sí concentra una cantidad
grande (21,5%, toda hacia arriba), y se le aplica un tope de 10% en vez del % real completo,
para no generar un pico artificial de valores idénticos en la columna.
track_id, artists, album_name y track_name se excluyen de las features por altísima
cardinalidad (73.261 nombres de canción y 45.880 álbumes distintos sobre 83.478 filas, casi cada
valor aparece una sola vez). El resultado: **83.478 canciones → 66.782 de entrenamiento / 16.696
de prueba → 29 columnas finales, sin valores nulos**, verificado ejecutando el pipeline de punta
a punta.

## 7. Evaluación de sesgos, ética y privacidad de los datos

**Privacidad:** el dataset no contiene datos personales de oyentes (sin historial de escucha ni
identificadores de usuarios). Las únicas variables descriptivas de personas (`artists`) son
nombres de artistas públicos asociados a un catálogo musical comercial, no datos de individuos
privados. De todas formas se excluyen de las features del pipeline (sección 6.5), pero por falta
de valor predictivo utilizable (altísima cardinalidad), no por motivos de privacidad, es una
distinción importante: no es una medida de anonimización, es una decisión de ingeniería de
características.

**Sesgo de diseño del dataset (género):** la distribución casi uniforme de track_genre
(~1.000 canciones por género, sección 6.4) es un artefacto del muestreo con que se construyó el
dataset, no la popularidad real de cada género en el mundo real. Un modelo entrenado con
GenreMeanEncoder aprende "qué tan popular es un género **en esta muestra**", que no es lo mismo
que "qué tan popular es ese género entre oyentes reales de Spotify", una limitación a comunicar
si el modelo se usa para decisiones reales de programación musical.

**Sesgo de contexto de publicación:** el hallazgo de la sección 6.1 (misma canción, distinta
popularidad según el release) muestra que popularity no es una propiedad pura del contenido
musical, sino también del contexto de publicación (álbum, momento, promoción). Un modelo que
prediga popularidad a partir solo de audio y género tiene un techo de precisión inherente por
esta razón, independiente de qué tan bueno sea el modelo.

**Sesgo de "el que ya es famoso, sigue siendo detectado como popular":** se evaluó (fuera de los
notebooks, como parte del diseño de features) codificar artists/album_name con la misma
técnica de promedio histórico usada para género. El álbum resulta la señal más fuerte de todo el
dataset (r≈0,72 en conjunto de prueba, contra 0,62 de género); el artista queda en un nivel
similar al género (r≈0,63). Se decidió **no incorporarlas**, por dos razones: (1) predeciría principalmente "qué tan
famoso es ya el artista", no "qué hace que una canción suene popular", que es el problema de
negocio real (sección 1), un modelo así sería inútil para descubrir artistas nuevos por ejemplo, y (2) reforzaría un sesgo de retroalimentación
("rich get richer"): un modelo que aprende que un artista ya famoso seguirá siendo popular
perpetúa esa ventaja en vez de evaluar la canción en sus propios méritos. Este es el KPI ético de
la sección 3 (0 features de identidad de artista/álbum en el modelo).

**Multicolinealidad y calidad del pipeline** (sección 6.3, 6.5): se trata como una cuestión de
calidad de datos y no de ética, pero se documenta acá por completitud: el CorrelationFilter
(umbral 0,9) actúa como resguardo automático y verificable ante variables redundantes que
pudieran sesgar el ajuste de un futuro modelo lineal.

## 8. Estructura del proyecto

```text
Spotify-Tracks-Dataset/
├── README.md                          # presentación del proyecto: qué es y cómo ejecutarlo
├── requirements.txt                   # dependencias de Python del proyecto
├── data/
│   └── Spotify_Tracks_Dataset.csv     # dataset crudo (114.000 filas)
├── docs/
│   ├── spotify_dataset.pdf            # documentación de cada columna del dataset
│   └── informe.md                     # este documento: informe técnico del proyecto
├── notebooks/
│   ├── analisis_exploratorio.ipynb    # Fase 2 CRISP-DM: comprensión de los datos (EDA)
│   ├── preprocesamiento.ipynb         # Fase 3 CRISP-DM: preparación de datos
│   └── modelamiento.ipynb             # Fases 4-5 CRISP-DM: modelado y evaluación
└── images/                            # figuras exportadas por el EDA y el modelado
```

No existe una carpeta `models/` porque los modelos de esta entrega no se serializan a disco (el
notebook `modelamiento.ipynb` los entrena y evalúa de punta a punta, de forma reproducible); se
incorporaría junto con los artefactos serializados del pipeline si el proyecto avanzara a la fase
de despliegue (fase 6, ver [Estado del proyecto](#10-estado-del-proyecto-y-próximos-pasos)).

**Para reproducir:** ver las instrucciones de instalación y ejecución en [`README.md`](../README.md).

## 9. Modelado: construcción, comparación y evaluación de modelos

Esta sección resume [`notebooks/modelamiento.ipynb`](../notebooks/modelamiento.ipynb) (fases 4-5
de CRISP-DM), verificado ejecutándolo de punta a punta. Como los notebooks anteriores, es
autocontenido: reconstruye `X_train_prep`, `X_test_prep`, `y_train`, `y_test` con el mismo
pipeline de `preprocesamiento.ipynb` (sección 6.5) en vez de depender de haberlo ejecutado antes.

### 9.1 Modelos supervisados: predicción de popularity

Se comparan tres modelos supervisados contra `popularity`:

| Modelo | MAE | R² |
|---|---|---|
| Regresión Lineal | 10,75 | 0,387 |
| Árbol de Decisión (`RandomizedSearchCV`) | 10,67 | 0,393 |
| Random Forest (por defecto, sin ajuste) | 10,05 | 0,428 |
| **Random Forest (`RandomizedSearchCV`)** | **9,98** | **0,438** |

- **Regresión Lineal**, elegida por ser el modelo más simple e interpretable y por servir de
  referencia directa de las correlaciones lineales del EDA (sección 6.3: ninguna variable de
  audio supera |r|=0,13): al combinar todas las variables (incluido el género, codificado como
  `genero_popularidad_media`) ya explica el 38,7% de la varianza, muy por encima de lo que
  sugería cualquier correlación individual. Su coeficiente estandarizado de mayor magnitud es,
  por lejos, el del género (11,7), seguido de `explicit` y `energy`.
- **Árbol de Decisión**, elegido como punto intermedio entre la Regresión Lineal y el ensamble
  de Random Forest: un único árbol captura no linealidades, pero sin el promedio de muchos
  árboles que reduce varianza. Ajustado con `RandomizedSearchCV` sobre `max_depth` y
  `min_samples_leaf` (mejor combinación: `max_depth=6, min_samples_leaf=50`), queda apenas por
  encima de la Regresión Lineal (R²=0,393 vs. 0,387): un único árbol, aunque no lineal, no
  alcanza a capturar mucha más señal que el modelo lineal. Depende casi por completo del género
  (97% de la importancia de variables), mucho más concentrado que el Random Forest.
- **Random Forest Regressor**, elegido por capturar interacciones y relaciones no lineales entre
  variables de audio y género que un modelo lineal no puede representar, y por ser robusto a los
  atípicos residuales del Winsorizer. En vez de elegir los hiperparámetros a mano, se ajustaron
  con `RandomizedSearchCV` (8 combinaciones de un espacio de 600 posibles: `n_estimators`,
  `max_depth`, `min_samples_leaf`, `max_features`; validación cruzada de 3 folds): la mejor
  combinación encontrada (`n_estimators=300, max_depth=22, min_samples_leaf=5,
  max_features=0.7`) mejora el R² en +0,010 frente a la configuración por defecto, confirmando
  que vale la pena la búsqueda en vez de usar los valores de fábrica.
- **Importancia de variables** (Random Forest con hiperparámetros de la búsqueda):
  `genero_popularidad_media` domina con ~56% de la importancia total; le siguen varias
  variables de audio parejas entre sí (`acousticness`, `danceability`, `loudness`, `energy`,
  `instrumentalness`, `valence`, `tempo`, `speechiness`, `liveness`, cada una entre 3,4-4,5%).
  Es consistente con el hallazgo del EDA (sección 6.4): el género es, por lejos, la señal más
  fuerte del contenido de la canción, y ninguna variable de audio individual domina por sí sola
  (sección 6.3).

### 9.2 Clasificación complementaria: ¿popularidad alta o baja?

Como complemento a la regresión, se entrena un segundo tipo de modelo supervisado sobre el
mismo problema de fondo: en vez de predecir el puntaje exacto, se clasifica cada canción como
`popularity` **alta** o **baja** respecto a la mediana del conjunto de entrenamiento (35),
reutilizando el mismo pipeline y las mismas 29 columnas. No es una repetición de la regresión:
responde una pregunta de negocio distinta y más directamente accionable: *"¿esta canción nueva
suena como que va a tener mejor o peor desempeño que la mediana del catálogo?"*, con sus
propias métricas (accuracy, F1, AUC-ROC).

| Modelo | Accuracy | F1 macro | AUC-ROC |
|---|---|---|---|
| Regresión Logística | 0,777 | 0,777 | 0,849 |
| Árbol de Decisión (`RandomizedSearchCV`) | 0,789 | 0,789 | 0,866 |
| **Random Forest Classifier (`RandomizedSearchCV`)** | **0,813** | **0,813** | **0,894** |

El Árbol de Decisión (`max_depth=10, min_samples_leaf=50`) queda entre ambos extremos, igual
que en la regresión: mejora sobre la Regresión Logística pero sin llegar al desempeño del
ensamble. El Random Forest queda bien balanceado entre ambas clases (precision/recall
~0,81-0,82 tanto para "alta" como para "baja", sin favorecer una por sobre la otra), y tanto él
como el Árbol de Decisión se ajustaron con el mismo criterio (`RandomizedSearchCV`, 8
combinaciones de un espacio de 600 posibles, validación cruzada de 3 folds).

**Por qué la accuracy (0,81) es tan superior al R² de la regresión (0,44), sin ser un modelo
"mejor":** se verificó tomando el Random Forest de regresión de la sección 9.1 (ya entrenado,
sin ajustar nada nuevo) y agrupando su propia predicción continua en la misma mediana: da una
accuracy de 0,800, muy cerca del clasificador entrenado específicamente para esta tarea (0,813).
La diferencia entre ambos no es una señal nueva descubierta por el clasificador, es solo que
optimiza un poco mejor la misma frontera que ya estaba implícita en el regresor. La razón de
fondo es que clasificar en 2 categorías balanceadas es una pregunta mucho más tolerante al
error que acertar el valor exacto: solo hay **una frontera** que cruzar (la mediana), y al
estar justo al medio de la distribución es la configuración más permisiva posible. No es que
el sonido y el género "expliquen mejor" si una canción es popular que cuánto exactamente: es
la misma señal (techo ≈0,44 de R²), vista con una vara de medir distinta.

### 9.3 Aprendizaje no supervisado: segmentación de canciones por sonido (KMeans)

Se agrupan las canciones usando únicamente sus 9 variables de audio ya escaladas (`danceability`,
`energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`,
`tempo`; se excluyen duración, género, variables categóricas y `popularity`, para que el
agrupamiento refleje solo el perfil sonoro, sin ver la variable objetivo ni usar identidad). Es
una generalización no supervisada de la idea manual de "cuadrantes de ánimo" (`valence` ×
`energy`) explorada en el EDA (sección 6.4-6.5).

Se evaluó k entre 2 y 8 con el método del codo (inercia) y el coeficiente de silhouette: la
inercia decrece de forma monótona sin un codo marcado, y el silhouette más alto es con k=2
(0,239, una separación binaria poco accionable); de k=3 en adelante se aplana entre 0,16 y 0,18
sin diferencias relevantes. Se eligió **k=4** por ofrecer una cohesión comparable a valores de k
mayores con un número de segmentos manejable para una decisión de negocio: una decisión
metodológica que prioriza interpretabilidad sobre el óptimo matemático puro (ver limitaciones en
9.5).

Perfil de los 4 segmentos (promedios de audio sin escalar):

| Cluster | Perfil | n canciones | Popularidad promedio |
|---|---|---|---|
| 0 | Acústico/calmado (`acousticness`≈0,77, `energy`/`danceability` bajos, volumen ≈-15 dB) | 14.607 | **31,96** (mínima) |
| 1 | Hablado/urbano (`speechiness`≈0,29, notoriamente más alto que el resto) | 9.169 | 34,84 |
| 2 | Pop/dance alegre (`danceability`/`energy`/`valence` altos, `acousticness` y `speechiness` bajos) | 24.401 | **36,76** (máxima) |
| 3 | Intenso/instrumental (`energy` y tempo los más altos, `instrumentalness` elevado, `valence` bajo) | 18.605 | 33,90 |

La diferencia de popularidad promedio entre los 4 segmentos es acotada (~5 puntos entre el
segmento más bajo y el más alto, sobre una media global de 34,65), coherente con el hallazgo
central del EDA (sección 6.3): el sonido por sí solo tiene una relación real pero débil con la
popularidad. El segmento
de menor popularidad (acústico, `valence` bajo) es además consistente con `es_calmado_positivo`
(sección 6.5): ahí el hallazgo fue sobre un cuadrante de *valence* alto/*energy* bajo; aquí,
usando las 9 variables de audio en conjunto, el segmento más penalizado tiene igualmente
*valence* bajo, reforzando que un sonido poco enérgico y poco positivo se asocia con menor
tracción, más allá de qué par de variables se use para definirlo.

### 9.4 Evaluación de resultados y recomendaciones de negocio

Retomando la pregunta de negocio de la sección 1, los tres modelos (regresión, clasificación y
clustering) confirman, de forma cuantitativa y de extremo a extremo, lo que el EDA ya anticipaba con
correlaciones simples: sonido y género explican una fracción real pero acotada de la
popularidad (R²≈0,44 en regresión; ~5 puntos de rango entre segmentos de sonido), con el género
como la señal individual más fuerte en todos los enfoques. La clasificación binaria (accuracy
0,81) no contradice este techo (es la misma señal, vista con una pregunta más tolerante al
error, sección 9.2) y es, de los tres, la más directamente accionable para una decisión de
negocio binaria ("¿invertir esfuerzo de promoción en esto o no?").

**Recomendación de negocio:** usar el modelo (y los segmentos de sonido) como señal de apoyo para
priorizar qué escuchar primero entre material nuevo de artistas sin trayectoria (el caso de uso
real de la sección 1), nunca como filtro automático ni sustituto del criterio humano: con
R²≈0,44, más de la mitad de la varianza de `popularity` queda sin explicar por el contenido de
la canción, y parte de esa varianza no observada corresponde a contexto de publicación (sección
6.1) y a la fama del artista, excluida deliberadamente del modelo por las razones éticas de la
sección 7. El modelo responde "qué tan prometedora suena esta canción en sí misma", no "qué tan
exitosa será": son preguntas distintas y el negocio debe tratarlas como tales.

### 9.5 Limitaciones del modelado

- La búsqueda de hiperparámetros (`RandomizedSearchCV`) se acotó a 8 combinaciones, tanto para
  los Árboles de Decisión como para los Random Forest, de regresión (sección 9.1) y de
  clasificación (sección 9.2); es suficiente para mejorar sobre la configuración por defecto en
  el caso del Random Forest, pero una búsqueda más amplia (`GridSearchCV` sobre el entorno de la
  mejor combinación encontrada, o más iteraciones) queda como trabajo futuro (sección 10).
- El techo de R²≈0,44 refleja el límite del contenido+género para explicar `popularity`, no
  necesariamente el límite del algoritmo: coincide con que la señal más fuerte medida (identidad
  de artista/álbum, r≈0,72, sección 7) fue deliberadamente excluida por razones éticas: un
  modelo que la incluyera llegaría a un R² mucho mayor, pero respondería la pregunta de negocio
  equivocada.
- La segmentación por sonido (k=4) no es la óptima según silhouette puro (k=2 tiene mayor
  cohesión, 0,239 vs 0,166 de k=4); se privilegió un número de segmentos más accionable para el
  negocio, una decisión metodológica explícita, no el resultado matemáticamente óptimo.

## 10. Estado del proyecto y próximos pasos

Este proyecto cubre comprensión del negocio, comprensión de los datos, preparación de
datos, modelado y evaluación (fases 1-5 de CRISP-DM). Queda explícitamente fuera de alcance, como
trabajo futuro:

- **Despliegue** (fase 6): servir el modelo (por ejemplo, una API), monitoreo de desempeño en
  producción y reentrenamiento periódico, junto con los artefactos serializados del pipeline y
  del modelo elegido.
- **Búsqueda de hiperparámetros más exhaustiva** (`GridSearchCV` sobre el entorno de la mejor
  combinación ya encontrada, o más iteraciones de `RandomizedSearchCV`) y modelos
  adicionales (por ejemplo, gradient boosting), más allá de la búsqueda acotada de la
  sección 9.

## 11. Matriz de responsabilidades del equipo

Tareas a cargo de cada integrante, según el historial de commits del repositorio.

| Tarea | Responsable |
|---|---|
| Configuración inicial del repositorio (estructura de carpetas, dataset, README base, `requirements.txt`) | Felipe Ahumada Silva |
| Primera versión de `notebooks/analisis_exploratorio.ipynb` (EDA) | Felipe Ahumada Silva |
| Primera versión de `notebooks/preprocesamiento.ipynb` | Felipe Ahumada Silva |
| Primera versión de `docs/informe.md` | Felipe Ahumada Silva |
| Correcciones de rutas y ajustes del README | Felipe Ahumada Silva |
| Ajustes y refinamiento del informe | Felipe Ahumada Silva |
| Corrección de inconsistencias detectadas en el EDA, el preprocesamiento y el informe | Francisca Carrasco Lozano |
| Corrección final del informe | Francisca Carrasco Lozano |
| Corrección de enlaces del informe, justificación de herramientas colaborativas y cierre de la primera entrega | Felipe Ahumada Silva |
| Modelado (`notebooks/modelamiento.ipynb`: regresión, clasificación complementaria y clustering), actualización del informe (sección 9) y corrección de inconsistencias detectadas en la preparación de datos | Francisca Carrasco Lozano |