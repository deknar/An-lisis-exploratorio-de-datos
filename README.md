# Análisis Exploratorio de Datos (EDA)

Este repositorio contiene el proyecto final del **Módulo 4: Análisis Exploratorio de Datos**. El objetivo principal es analizar un dataset de autos BMW usados para identificar los factores clave que influyen en su precio de venta, utilizando herramientas estadísticas y de visualización en Python.

El dataset `bmw.csv` se encuentra incluido en este repositorio.

## Descripción del Proyecto

El análisis se estructura en 6 lecciones progresivas que cubren desde la carga de datos hasta la creación de modelos predictivos y visualizaciones avanzadas:

1.  **Carga y Limpieza**: Importación de datos y manejo de tipos de variables.
2.  **Estadística Descriptiva**: Cálculo de medidas de tendencia central, dispersión y detección de outliers.
3.  **Correlación**: Análisis de relaciones entre variables numéricas (mapas de calor).
4.  **Regresión Lineal**: Modelo predictivo para estimar el precio basado en el año de fabricación (incluye métricas MSE, MAE y P-values).
5.  **Visualización Avanzada**: Gráficos complejos con `seaborn` (Jointplots, Pairplots).
6.  **Reporte Gráfico**: Comparativas finales y exportación de gráficos.

## Tecnologías Utilizadas

El proyecto está desarrollado en **Python** y utiliza las siguientes librerías:

-   **pandas**: Manipulación y análisis de datos.
-   **numpy**: Operaciones numéricas y cálculo de arrays.
-   **matplotlib**: Creación de gráficos estáticos.
-   **seaborn**: Visualización de datos estadísticos atractiva.
-   **statsmodels**: Modelado estadístico y pruebas de hipótesis.
-   **os**: Gestión de rutas de archivos compatible con diferentes sistemas operativos.

## Instalación y Uso

1.  **Clonar el repositorio**:
    ```bash
    git clone https://github.com/deknar/An-lisis-exploratorio-de-datos
    cd An-lisis-exploratorio-de-datos
    ```

2.  **Instalar dependencias**:
    Asegúrate de tener Python instalado y ejecuta:
    ```bash
    pip install pandas numpy matplotlib seaborn statsmodels
    ```

3.  **Ejecutar el análisis**:
    Corre el script principal `eda_bmw.py`:
    ```bash
    python eda_bmw.py
    ```

## Estructura del Script

El archivo `eda_bmw.py` realiza automáticamente las siguientes acciones:

-   Carga el dataset `bmw.csv` usando rutas relativas.
-   Renombra las columnas a español (`Precio`, `Año`, `Kilometraje`, etc.).
-   Genera y guarda 6 gráficos clave en formato PNG:
    -   `distribucion_precios.png`
    -   `boxplot_precios.png`
    -   `mapa_calor_correlacion.png`
    -   `regresion_lineal.png`
    -   `jointplot_km_precio.png`
    -   `precio_por_transmision.png`

## Hallazgos Clave

-   **Depreciación**: Existe una fuerte correlación negativa entre el kilometraje y el precio; a mayor uso, menor valor.
-   **Correlación Año-Precio**: Existe una correlación positiva moderada (0.62). Los autos más nuevos tienden a ser más caros y el año de fabricación es un predictor significativo (R² ≈ 0.39).
-   **Transmisión**: Los vehículos con transmisión automática y semi-automática tienden a tener un precio promedio superior a los manuales.

---

