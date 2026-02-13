import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import os

# configuración visual básica
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.formatter.useoffset'] = False # evitar notación científica en ejes
plt.ticklabel_format(style='plain', axis='y') # precios completos

# para que funcione en github (ruta relativa al script)
archivo = os.path.join(os.path.dirname(__file__), 'bmw.csv')

# lección 1: análisis exploratorio de datos
print("Lección 1: Cargando y explorando datos")

try:
    df = pd.read_csv(archivo)
    print(f"Datos cargados correctamente. Filas: {df.shape[0]}, Columnas: {df.shape[1]}")
except FileNotFoundError:
    print(f"No se encontró el archivo en: {archivo}")
    print("Asegúrate de que bmw.csv esté en la misma carpeta que este script")
    exit()

# renombrar columnas a español
df.rename(columns={
    'model': 'Modelo',
    'year': 'Año',
    'price': 'Precio',
    'transmission': 'Transmision',
    'mileage': 'Kilometraje',
    'fuelType': 'Tipo combustible',
    'tax': 'Impuesto',
    'mpg': 'Mpg',
    'engineSize': 'Tamaño motor'
}, inplace=True)
print("Columnas renombradas")

# traducir valores de transmisión
mapa_transmision = {
    'Automatic': 'Automático',
    'Manual': 'Manual',
    'Semi-Auto': 'Semi-Automático',
    'Other': 'Otro'
}
df['Transmision'] = df['Transmision'].map(mapa_transmision)
print("Valores de transmisión traducidos")

print("Precios en Libras Esterlinas (GBP)")

# vemos qué tipo de datos tenemos
print("Tipos de variables:")
print(df.dtypes)

# buscamos valores nulos
nulos = df.isnull().sum()
if nulos.sum() == 0:
    print("No hay valores faltantes en el dataset")
else:
    print("Valores faltantes por columna:")
    print(nulos[nulos > 0])

# lección 2: estadística descriptiva
print("Lección 2: Estadísticas básicas")

# resumen estadístico (precios en GBP):
print(df.describe().round(2))

# gráfico de distribución del precio
plt.figure()
sns.histplot(df['Precio'], kde=True, color='skyblue')
plt.title('Distribución de precios de los autos')
plt.xlabel('Precio (£)')
plt.ylabel('Frecuencia')
plt.tight_layout()
plt.savefig('distribucion_precios.png')
plt.close()
print("Gráfico guardado: distribucion_precios.png")

# diagrama de caja para detectar valores atípicos en el precio
plt.figure()
sns.boxplot(x=df['Precio'], color='orange')
plt.title('Detectando precios atípicos (outliers)')
plt.xlabel('Precio (£)')
plt.tight_layout()
plt.savefig('boxplot_precios.png')
plt.close()
print("Gráfico guardado: boxplot_precios.png")

# lección 3: correlación
print("Lección 3: Analizando relaciones")

# seleccionamos solo columnas numéricas para la correlación
cols_numericas = df.select_dtypes(include=[np.number]).columns
matriz_corr = df[cols_numericas].corr()

print("Matriz de correlación (primeras filas):")
print(matriz_corr.head())

# mapa de calor
plt.figure(figsize=(10, 8))
sns.heatmap(matriz_corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlación entre variables numéricas')
plt.savefig('mapa_calor_correlacion.png')
plt.close()
print("Gráfico guardado: mapa_calor_correlacion.png")

# lección 4: regresión lineal
print("Lección 4: Modelo de regresión (Precio vs Año)")

# preparamos los datos: queremos predecir precio basado en el año
X = df['Año']
y = df['Precio']

# añadimos una constante para el intercepto (necesario en statsmodels)
X_cte = sm.add_constant(X)

# ajustamos el modelo
modelo = sm.OLS(y, X_cte).fit()

# predicciones para calcular errores
y_pred = modelo.predict(X_cte)

# métricas de error
mse = np.mean((y - y_pred) ** 2)
mae = np.mean(np.abs(y - y_pred))

# resumen estadístico
print("Resultados de la Regresión:")
print(f"Coeficiente R2: {modelo.rsquared:.4f}")
print(f"MSE: {mse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"Intercepto: {modelo.params['const']:.2f}")
print(f"Pendiente (Coeficiente Año): {modelo.params['Año']:.2f} GBP")

print("Significancia estadística (P-values):")
print(modelo.pvalues.round(4))

if modelo.pvalues['Año'] < 0.05:
    print("-> El año es un predictor significativo (p < 0.05).")
else:
    print("-> El año NO es significativo.")

# gráfico de la regresión
plt.figure()
sns.regplot(x='Año', y='Precio', data=df, line_kws={"color": "red"})
plt.title('Regresión lineal: Precio según el Año')
plt.xlabel('Año')
plt.ylabel('Precio (£)')
plt.ticklabel_format(style='plain', axis='y')
plt.tight_layout()
plt.savefig('regresion_lineal.png')
plt.close()
print("Gráfico guardado: regresion_lineal.png")

# lección 5: análisis visual avanzado
print("Lección 5: Visualizaciones avanzadas")

# jointplot para ver relación kilometraje vs precio
print("Generando gráfico combinado de Kilometraje vs Precio...")
g = sns.jointplot(x='Kilometraje', y='Precio', data=df, kind='hex', height=8)
g.fig.suptitle('Densidad: Kilometraje vs Precio (GBP)', y=1.02)
plt.savefig('jointplot_km_precio.png')
plt.close()
print("Gráfico guardado: jointplot_km_precio.png")

# lección 6: matplotlib y reporte final
print("Lección 6: Gráfico comparativo final")

# comparamos precio promedio por tipo de transmisión
precio_transmision = df.groupby('Transmision')['Precio'].mean().sort_values()

plt.figure(figsize=(10, 6))
precio_transmision.plot(kind='bar', color='teal')
plt.title('Precio promedio por tipo de transmisión (GBP)')
plt.xlabel('Transmisión')
plt.ylabel('Precio promedio (£)')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.ticklabel_format(style='plain', axis='y')
plt.tight_layout()
plt.savefig('precio_por_transmision.png')
plt.close()
print("Gráfico guardado: precio_por_transmision.png")

print("Proyecto finalizado. Gráficos guardados.")
