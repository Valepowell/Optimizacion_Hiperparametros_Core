# Optimizacion_Hiperparametros_Core
Análisis Exploratorio de Datos y Modelado Predictivo en Datos de Seguros
Descripción del Proyecto
Este proyecto tiene como objetivo realizar un Análisis Exploratorio de Datos (EDA) exhaustivo y desarrollar modelos predictivos utilizando un conjunto de datos relacionado con seguros. Se abordan problemas de clasificación y regresión para obtener insights sobre los datos y construir modelos capaces de realizar predicciones relevantes.

Conjunto de Datos
El análisis se basa en el archivo insurance.csv, que contiene información sobre asegurados, incluyendo edad, sexo, índice de masa corporal (BMI), número de hijos, estado de fumador, región y cargos médicos.

Análisis Exploratorio de Datos (EDA)
Se realizó un EDA inicial para comprender la estructura de los datos, identificar valores nulos y duplicados, y visualizar las distribuciones y relaciones entre las variables.

Carga de Datos: Los datos se cargaron en un DataFrame de pandas.
Limpieza de Datos: Se identificó y manejó un duplicado en el conjunto de datos. Se verificó la ausencia de valores nulos.
Análisis Descriptivo: Se obtuvieron estadísticas descriptivas para las variables numéricas y se analizaron las distribuciones de las variables categóricas.
Visualizaciones: Se generaron visualizaciones (histogramas, boxplots, mapas de calor de correlación) para explorar las distribuciones de las variables y las relaciones entre ellas.
Modelado de Clasificación: Predicción del Estado de Fumador
Se abordó un problema de clasificación para predecir si un individuo es fumador ('smoker') basándose en otras características.

Preparación de Datos: Se dividieron los datos en conjuntos de entrenamiento y prueba. Las características numéricas fueron escaladas y las categóricas codificadas utilizando un ColumnTransformer dentro de un Pipeline.
Modelos Implementados: Se entrenaron y evaluaron los siguientes modelos de clasificación:
Regresión Logística
Árbol de Decisión
K-Nearest Neighbors (KNN)
Optimización de Hiperparámetros: Se utilizó GridSearchCV con validación cruzada para encontrar los mejores hiperparámetros para cada modelo.
Evaluación del Modelo: Los mejores modelos se evaluaron en el conjunto de prueba utilizando métricas como la matriz de confusión, reporte de clasificación (precision, recall, f1-score) y ROC-AUC con la curva ROC.
Resultados de Clasificación:

(Se pueden añadir aquí los resultados clave de cada modelo, por ejemplo: "El modelo de Regresión Logística optimizado alcanzó un accuracy del X% y un ROC-AUC de Y en el conjunto de prueba, mostrando un alto rendimiento en la predicción del estado de fumador.")

Próximos Pasos: Modelado de Regresión
El siguiente objetivo del proyecto es realizar un análisis de regresión para predecir la variable 'charges'. Los pasos a seguir incluyen:

Preparación de datos para regresión.
Preprocesamiento de datos para regresión.
División de datos.
Implementación y entrenamiento de modelos de regresión.
Evaluación de modelos de regresión.
Análisis de resultados y conclusiones.
Cómo Ejecutar el Código
El código se encuentra en un notebook de Jupyter/Colab. Para ejecutarlo:

Clonar o descargar el repositorio.
Abrir el notebook en un entorno con Python y las librerías necesarias (pandas, numpy, scikit-learn, matplotlib, seaborn).
Asegurarse de que el archivo insurance.csv esté accesible desde la ruta especificada en el notebook.
Ejecutar las celdas secuencialmente.
Dependencias
pandas
numpy
scikit-learn
matplotlib
seaborn
Autor
