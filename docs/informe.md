# Inteligencia Musical: Predicción de Popularidad de Canciones (Spotify Tracks)

**Evaluación Parcial N°1 — MLY1101 Machine Learning — Duoc UC**
**Caso de Estudio C:** Inteligencia musical y predicción de popularidad de canciones (Spotify Tracks)
**Por:** Felipe Ahumada Silva y Francisca Carrasco Lozano

Este es el informe técnico del proyecto, entregable formal de esta evaluación. Documenta el
problema de negocio, los objetivos, los KPIs, las fuentes de datos, la metodología y el trabajo de
comprensión y preparación de datos realizado hasta esta entrega. El desarrollo completo y
ejecutable de cada paso está en los notebooks de la carpeta [`notebooks/`](../notebooks/); este
informe resume y referencia ese trabajo, no lo reemplaza. Para instrucciones de instalación y
ejecución del proyecto, ver [`README.md`](../README.md).

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
quien la interpreta?**, como comprensión de datos previa a una futura etapa de modelado
predictivo.

Es una pregunta exploratoria, no una promesa de resultado, y el propio análisis (sección 6) la
responde de forma más matizada de lo esperado: las características de audio por sí solas
explican muy poco de la popularidad (correlación máxima de apenas -0,13), el género aporta una
señal bastante más fuerte pero es en parte un artefacto del muestreo con que se construyó el
dataset (sección 6.4), y la señal individual más fuerte de todas la identidad del artista o
álbum se descarta deliberadamente pese a ser la más predictiva, por las razones éticas
detalladas en la sección 7: usarla resolvería el problema estadístico, pero no el de negocio,
porque reproduciría exactamente el sesgo de reputación que se quiere evitar. Que "el sonido de
una canción por sí solo prediga poco su éxito" es en sí mismo un hallazgo relevante para el
negocio: matiza cualquier expectativa de que un futuro modelo basado en audio y género pueda
reemplazar esto, y deja claro que su aporte es el de una señal adicional, acotada,
no un reemplazo del criterio humano.

## 2. Objetivos del proyecto

1. **Comprender la calidad y estructura del dataset**: identificar y resolver problemas de
   duplicación, valores inválidos y consistencia antes de sacar cualquier conclusión sobre
   popularidad.
2. **Explorar qué atributos se asocian con la popularidad de una canción** — de audio, de
   género, y de contexto de publicación (release) — y cuantificar la fuerza (o debilidad) de esa
   asociación, sin asumir de antemano que los atributos musicales vayan a explicarla bien.
3. **Preparar y transformar los datos** en un pipeline reproducible (scikit-learn
   Pipeline/ColumnTransformer), sin fuga de información entre entrenamiento y prueba, dejando
   el dataset listo para una futura etapa de modelado.
4. **Identificar y documentar los sesgos y limitaciones éticas** del dataset y de las decisiones
   de preparación tomadas, antes de avanzar a modelado.

Los objetivos 1, 2 y 4 están cubiertos por esta entrega (EDA + preparación). El modelado
predictivo (entrenar y evaluar modelos de regresión sobre popularity) queda explícitamente
fuera del alcance de este informe y de los notebooks actuales, ver
[Estado del proyecto](#9-estado-del-proyecto-y-próximos-pasos).

## 3. Definición de KPIs

Los KPIs se dividen en dos grupos: los que responden directamente la pregunta de negocio de
la sección 1, y los que miden la calidad de los datos como condición necesaria para que los
primeros sean confiables. Ambos grupos son medibles con el trabajo de esta etapa (EDA +
preparación de datos); ninguno requiere un modelo entrenado, esa métrica (precisión predictiva
del modelo final) es un KPI de la fase 4, fuera de alcance de esta entrega (ver sección 9).

### KPIs de negocio

Responden la pregunta de la sección 1: ¿qué tan bien explica la canción misma su popularidad,
sin depender de la fama de quien la interpreta?

| KPI | Meta | Resultado |
|---|---|---|
| Techo de señal explicable por el contenido de la canción (sin usar identidad de artista/álbum) | Cuantificar cuánto puede llegar a explicar, en el mejor caso, un modelo que solo use sonido y género | Atributos de audio solos: correlación máxima de apenas **-0,13** (instrumentalness). Género (codificado adecuadamente): **0,62**, muy superior a cualquier atributo de audio individual. Este 0,62 es la cota realista de lo que el contenido de la canción puede explicar sin recurrir a quién la interpreta — fija expectativas honestas para la futura fase de modelado, antes de invertir en entrenarlo. |
| Features basadas en identidad de artista/álbum en el modelo final | 0 | Se midió que artista/álbum predicen popularidad mucho mejor que cualquier otro atributo (r≈0,71–0,72), pero se excluyen deliberadamente para no reforzar el sesgo de "el que ya es famoso, sigue siendo detectado como popular" (ver sección 7). |
| % del catálogo cuya baja popularidad es un artefacto del contexto de publicación, no de la calidad de la canción | Cuantificar para no confundir "canción poco atractiva" con "dato mal catalogado" en decisiones de negocio | **54%** del pico de canciones con popularity = 0 (5.088 de 9.413) corresponde a duplicados de release — el mismo audio con otro release sin tracción — no a que la canción en sí no guste (ver sección 6.1). Sin este KPI, el negocio podría descartar por error contenido que en realidad sí tiene tracción, solo mal catalogado. |

### KPIs de calidad de datos (soporte)

Condición necesaria para que los KPIs de negocio de arriba sean confiables, no miden el
problema de negocio en sí:

| KPI | Meta | Resultado |
|---|---|---|
| Duplicados identificados y resueltos | 100% | 24.259 filas por track_id repetido (mismo audio, distinto género) + 9.284 filas de audio idéntico bajo track_id distinto, el 100% consolidado, sin filas duplicadas remanentes. |
| Valores inválidos identificados y tratados | 100% | 129 time_signature = 0 / 124 tempo = 0 (sobre el dataset ya consolidado) marcados como faltantes para imputación; 0 valores inválidos sin tratar en el dataset preparado. |
| Dataset final sin nulos tras la preparación | 0 nulos | X_train_prep / X_test_prep: 0 valores nulos, verificado. |

## 4. Descripción de las fuentes de datos y herramientas colaborativas

**Fuente de datos:** [`data/Spotify_Tracks_Dataset.csv`](../data/Spotify_Tracks_Dataset.csv) — 114.000
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

- **Git y GitHub** — control de versiones y repositorio compartido. Es la herramienta central del
  trabajo colaborativo del proyecto: mantiene el historial de commits del informe y de ambos
  notebooks, permite revisar qué cambió entre versiones de un análisis, y actúa como fuente única
  de verdad para el equipo. Se eligió sobre alternativas de carpeta compartida porque los
  notebooks de Jupyter son archivos JSON grandes: sin control de versiones, dos personas editando
  en paralelo se sobrescriben sin dejar rastro.
- **Notebooks autocontenidos** — decisión de trabajo derivada de lo anterior: cada notebook
  reconstruye su propio dataset desde el CSV crudo, sin depender de haber ejecutado el otro. Esto
  permite que dos personas trabajen sobre distintas etapas del proyecto en paralelo y que
  cualquiera pueda ejecutar y verificar el trabajo del otro sin coordinación previa.
- **Jupyter Notebook** — entorno de análisis; el código, su salida y la justificación en Markdown
  quedan juntos en el mismo archivo, que es lo que permite que el trabajo sea revisable por otra
  persona sin explicación adicional.
- **Documentación de Spotify** ([`spotify_dataset.pdf`](spotify_dataset.pdf)) — referencia
  compartida de definiciones y rangos válidos de cada columna. Fija un criterio común: sin ella,
  cada integrante decidiría por su cuenta qué valor es "inválido".

## 5. Metodología utilizada (CRISP-DM)

El proyecto sigue la metodología **CRISP-DM** (Cross-Industry Standard Process for Data Mining),
de 6 fases iterativas. Esta entrega cubre las primeras tres:

1. **Comprensión del negocio** — cubierta en las secciones 1-3 de este informe: problema,
   objetivos y KPIs.
2. **Comprensión de los datos** — cubierta por [`notebooks/analisis_exploratorio.ipynb`](../notebooks/analisis_exploratorio.ipynb):
   carga, calidad, duplicados, valores inválidos, outliers, correlaciones, distribución del
   target.
3. **Preparación de los datos** — cubierta por [`notebooks/preprocesamiento.ipynb`](../notebooks/preprocesamiento.ipynb):
   consolidación adicional, tratamiento de inválidos, ingeniería de características, pipeline de
   transformación.
4. **Modelado** — fuera de alcance de esta entrega, se define como trabajo futuro.
5. **Evaluación** — depende de la fase de modelado.
6. **Despliegue** — depende de la fase de evaluación.

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

El EDA identifica un **segundo nivel de duplicación que la consolidación por track_id no resuelve** track_id identifica un *release* (álbum, single, recopilatorio), no la canción en
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

track_genre es multietiqueta (114 géneros posibles, separados por ; 82% de las canciones
tiene un solo género, promedio 1,27). Su distribución (~1.000 canciones por género) no es
orgánica: es la huella del muestreo estratificado con que se construyó el dataset, no refleja
popularidad real de cada género en Spotify, una limitación a tener presente si se usa como
predictor (ver sección 7).

Pese a eso, un chequeo de la popularidad promedio por género muestra la relación más fuerte
encontrada en todo el dataset: pop promedia 47,6 de popularidad, grindcore 14,6, un rango de
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
para no filtrar información hacia el conjunto de prueba. Se verificó (comparando la versión que
usa el pipeline contra una versión *out-of-fold*) que la fuga de información dentro del propio
entrenamiento es despreciable (diferencia de r ≈ 0,002), dado que los 114 géneros tienen cientos
de canciones cada uno. La consolidación por audio (6.1) mejora además esta señal: la correlación
de GenreMeanEncoder con popularity en el conjunto de prueba sube de 0,568 a **0,618**.

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
técnica de promedio histórico usada para género. La señal resultante es mucho más fuerte que
cualquier otra en el dataset (r≈0,71 artista, r≈0,72 álbum en conjunto de prueba, contra 0,62 de
género). Se decidió **no incorporarla**, por dos razones: (1) predeciría principalmente "qué tan
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

```
Spotify-Tracks-Dataset/
├── README.md                          — presentación del proyecto: qué es y cómo ejecutarlo
├── requirements.txt                   — dependencias de Python del proyecto
├── data/
│   └── Spotify_Tracks_Dataset.csv     — dataset crudo (114.000 filas)
├── docs/
│   ├── spotify_dataset.pdf            — documentación de cada columna del dataset
│   └── informe.md                     — este documento: informe técnico exigido por la rúbrica
├── notebooks/
│   ├── analisis_exploratorio.ipynb    — Fase 2 CRISP-DM: comprensión de los datos (EDA)
│   └── preprocesamiento.ipynb         — Fase 3 CRISP-DM: preparación de datos
└── images/                            — figuras exportadas por el EDA
```

No existe una carpeta `models/` porque el entrenamiento de modelos queda fuera del alcance de esta
entrega (ver [Estado del proyecto](#9-estado-del-proyecto-y-próximos-pasos)); se incorporará en la
fase 4 de CRISP-DM junto con los artefactos serializados del pipeline.

**Para reproducir:** ver las instrucciones de instalación y ejecución en [`README.md`](../README.md).

## 9. Estado del proyecto y próximos pasos

Esta entrega cubre comprensión del negocio, comprensión de los datos y preparación de datos
(fases 1-3 de CRISP-DM). Quedan explícitamente fuera de alcance, como trabajo futuro:

- **Modelado** (fase 4): entrenar y comparar modelos de regresión sobre popularity usando el
  dataset preparado (X_train_prep, X_test_prep, y_train, y_test), definiendo ahí, dentro
  de un notebook, la línea base ingenua (predecir siempre el promedio de popularity) como punto
  de comparación.
- **Evaluación** (fase 5) y **despliegue** (fase 6), dependientes del modelado.