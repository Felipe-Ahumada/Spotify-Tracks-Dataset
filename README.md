# Spotify Tracks — Predicción de Popularidad

Proyecto de la asignatura **MLY1101 Machine Learning (Duoc UC)** — Evaluación Parcial N°1, Caso de
Estudio C: *Inteligencia musical y predicción de popularidad de canciones*.

## ¿De qué trata?

Analiza el [Spotify Tracks Dataset](data/Spotify_Tracks_Dataset.csv) (~114.000 canciones con
características de audio y metadatos) para entender qué atributos musicales se asocian con la
popularidad de una canción en Spotify, como preparación para una futura etapa de modelado
predictivo. No es una aplicación: el resultado es el análisis en sí, expresado en notebooks de
Jupyter con el razonamiento detrás de cada decisión.

Para el informe técnico completo (problema de negocio, objetivos, KPIs, metodología CRISP-DM,
hallazgos del EDA, decisiones de preparación de datos y evaluación de sesgos/ética), ver
**[`informe.md`](./docs/informe.md)**.

## Estructura del proyecto

```
EV_1/
├── README.md                          — este archivo
├── requirements.txt                   — dependencias de Python del proyecto
├── data/
│   └── Spotify_Tracks_Dataset.csv     — dataset crudo (114.000 filas)
├── docs/
│   ├── spotify_dataset.pdf            — documentación de cada columna (Spotify)
│   └── informe.md                     — informe técnico del proyecto
├── notebooks/
│   ├── analisis_exploratorio.ipynb    — Fase 2 CRISP-DM: comprensión de los datos (EDA)
│   └── preprocesamiento.ipynb         — Fase 3 CRISP-DM: preparación de datos
└── images/                            — figuras exportadas por el EDA 
```

## Cómo usarlo

**Requisitos:** Python 3.12 y las librerías listadas en [`requirements.txt`](requirements.txt)
(`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn` y `jupyter`). Para verificar la
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

Ambos notebooks son autocontenidos: cada uno reconstruye el dataset que necesita desde el CSV
crudo.

## Estado

Esta entrega cubre comprensión del negocio, comprensión de los datos y preparación de datos. El
entrenamiento de modelos predictivos queda fuera de alcance, ver la sección 9 de
[`informe.md`](informe.md) para el detalle de qué falta.
