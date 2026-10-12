# Spotify Tracks — Predicción de Popularidad

Proyecto de la asignatura **MLY1101 Machine Learning (Duoc UC)** — Evaluaciones Parciales N°1 y
N°2, Caso de Estudio C: *Inteligencia musical y predicción de popularidad de canciones*.

**Por:** Felipe Ahumada Silva y Francisca Carrasco Lozano

## ¿De qué trata?

Analiza el [Spotify Tracks Dataset](data/Spotify_Tracks_Dataset.csv) (~114.000 canciones con
características de audio y metadatos) para entender qué atributos musicales se asocian con la
popularidad de una canción en Spotify, y entrena/compara modelos de Machine Learning sobre esos
datos: dos modelos supervisados de regresión para predecir `popularity`, dos modelos
supervisados de clasificación (como complemento, `popularity` alta/baja respecto a la mediana)
y una técnica no supervisada (KMeans) para segmentar canciones por su perfil de sonido. No es
una aplicación: el
resultado es el análisis en sí, expresado en notebooks de Jupyter con el razonamiento detrás de
cada decisión.

Para el informe técnico completo (problema de negocio, objetivos, KPIs, metodología CRISP-DM,
hallazgos del EDA, decisiones de preparación de datos y evaluación de sesgos/ética), ver
**[`informe.md`](./docs/informe.md)**.

## Estructura del proyecto

```
Spotify-Tracks-Dataset/
├── README.md                          — este archivo
├── requirements.txt                   — dependencias de Python del proyecto
├── data/
│   └── Spotify_Tracks_Dataset.csv     — dataset crudo (114.000 filas)
├── docs/
│   ├── spotify_dataset.pdf            — documentación de cada columna del dataset
│   └── informe.md                     — informe técnico del proyecto
├── notebooks/
│   ├── analisis_exploratorio.ipynb    — Fase 2 CRISP-DM: comprensión de los datos (EDA)
│   ├── preprocesamiento.ipynb         — Fase 3 CRISP-DM: preparación de datos
│   └── modelamiento.ipynb             — Fases 4-5 CRISP-DM: modelado y evaluación
└── images/                            — figuras exportadas por el EDA y el modelado
```

## Cómo usarlo

**Requisitos:** Python 3.12 y las librerías listadas en [`requirements.txt`](requirements.txt)
(`pandas`, `numpy`, `matplotlib`, `scikit-learn`, `joblib` y `jupyter`). Para verificar la
versión instalada de una librería puntual, ejecutar `import <lib>; print(<lib>.__version__)` como
se indica en la primera celda de cada notebook.

```bash
pip install -r requirements.txt
```

**Ejecución:** abrir y correr cada notebook de punta a punta desde la carpeta `notebooks/`
(es la carpeta de trabajo que esperan las rutas relativas como `../data/...`):

```bash
cd notebooks
jupyter notebook
```

- [`analisis_exploratorio.ipynb`](notebooks/analisis_exploratorio.ipynb) — carga el CSV crudo,
  limpia duplicados, valida rangos, analiza outliers/correlaciones y explora `track_genre`.
- [`preprocesamiento.ipynb`](notebooks/preprocesamiento.ipynb) — reconstruye el dataset limpio de
  forma independiente (no depende de haber corrido el otro notebook antes), agrega ingeniería de
  características y arma el pipeline de `scikit-learn` que deja los datos listos para modelar.
- [`modelamiento.ipynb`](notebooks/modelamiento.ipynb) — reconstruye el dataset preparado
  (mismo pipeline que `preprocesamiento.ipynb`), entrena y compara dos modelos supervisados de
  regresión (Regresión Lineal, Random Forest) contra una línea base ingenua, dos modelos
  supervisados de clasificación complementarios (Regresión Logística, Random Forest Classifier)
  sobre `popularity` alta/baja, y aplica KMeans para segmentar canciones por perfil de sonido.

Los tres notebooks son autocontenidos: cada uno reconstruye el dataset que necesita desde el CSV
crudo.

## Estado

Esta entrega cubre las cinco primeras fases de CRISP-DM: comprensión del negocio, comprensión de
los datos, preparación de datos, modelado y evaluación. El despliegue (fase 6) queda fuera de
alcance, ver la sección 10 de [`informe.md`](docs/informe.md) para el detalle de qué falta.
