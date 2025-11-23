# Librería de Análisis Exploratorio de Datos (CTG)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Enabled-red)
![Tests](https://img.shields.io/badge/Pytest-Passing-green)

Librería personalizada en Python para el análisis, limpieza y visualización del conjunto de datos **Cardiotocography (Kaggle)**. Este proyecto implementa un flujo completo de Ciencia de Datos, desde el preprocesamiento automatizado hasta un dashboard interactivo.

##  Características Principales

* **Preprocesamiento Robusto:**
    * Eliminación automática de columnas con >20% de nulos.
    * Imputación de valores faltantes (Moda para categóricos, KNN/Mediana para numéricos).
    * Detección y tratamiento de outliers mediante Rango Intercuartílico (IQR).
* **Visualización Avanzada:**
    * Gráficos interactivos con **Plotly** (Zoom, Pan, Tooltips).
    * Soporte para Histogramas, Boxplots Facetados, Violines y Heatmaps de correlación.
* **Validación:**
    * Batería de pruebas unitarias con `pytest`.
    * Clasificación automática de variables (Continuas vs Discretas).

##  Estructura del Proyecto

```text
├── ctg_viz/               # Paquete principal
│   ├── plots/             # Módulo de visualización (Plotly)
│   ├── preprocessing.py   # Lógica de limpieza y outliers
│   └── utils.py           # Reportes de completitud
├── notebooks/             # Análisis y Reportes
│   └── demo_analysis.ipynb # Notebook principal (Generador del PDF)
├── tests/                 # Pruebas Unitarias
├── app.py                 # Aplicación Web (Streamlit)
├── setup.py               # Configuración de empaquetado
├── requirements.txt       # Dependencias
├── LICENSE                # Licencia de uso
└── README.md              # Documentación
```

# 🛠️ Instalación

Clonar el repositorio:

git clone https://github.com/JosueJimenezA/Practica3_DCD.git



##  Instalar dependencias:

```python
pip install -r requirements.txt
```

## Uso Básico (Librería)

Lo siguiente son ejemplos de uso de las funciones, se puede explorar el uso de estas funciones directamente en el notebook principal demo_analysis.ipynb. Este archivo es el insumo del pdf por lo que dentro del notebook se pueden encontrar comentarios de como se aplican las funciones, asi como comentarios respecto a los datos.

#### Ejemplo de la importación de funciones de la libreria personalizada

```python
import pandas as pd
from ctg_viz.preprocessing import remove_null_columns, detect_handle_outliers
from ctg_viz.plots.histograms import plot_histogram_interactive
```

### 1. Cargar datos y Limpiar con las funciones personalizadas
Para descargar los datos se puede realizar en el siguiente link
https://www.kaggle.com/code/akshat0007/cardiotocology/input


```python
df = pd.read_csv('data/CTG.csv')
df_clean = remove_null_columns(df, threshold=0.2)
df_final = detect_handle_outliers(df_clean, method='iqr')
```
Sin embargo, si se hace uso de la herramienta interactiva se puede seleccionar visualmente otro archivo.

### 2. Ejemplo de visualización de los gráficos personalizados

```python
fig = plot_histogram_interactive(df_final, col='LB', group_by='NSP')
fig.show()
```

##  Dashboard Interactivo
Este proyecto incluye una aplicación web para explorar los datos dinámicamente. Para iniciarla:

```python
streamlit run app.py
```


##  Ejecución de Pruebas
Para validar la lógica de limpieza y procesamiento:

```python
pytest tests/ -v
```


##  Bonus
Para instalar localmente la libreria solo debemos correr desde la consola el siquiente comando

```bash
pip install -e .
```

Despues de ya es posible importarla en cualquier proyecto

```bash
import ctg_viz
```


##  Autores

Josue Jimenez Apodaca

Diplomado de Ciencia de Datos, FES Acatlán

Fecha: Noviembre 2025