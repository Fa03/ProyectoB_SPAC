# Proyecto B: Sistema de Predicción de Abandono de Clientes

[![Carga de datos](https://img.shields.io/badge/Cargar%20datos-CSV-4CAF50?style=for-the-badge&logo=files)]()
[![Limpieza de datos](https://img.shields.io/badge/Limpieza%20y%20EDA-Procesamiento-2196F3?style=for-the-badge&logo=databricks)]()   
[![Modelado supervisado](https://img.shields.io/badge/Modelado-Supervisado-FF9800?style=for-the-badge&logo=mlflow)]()  
[![Documentación](https://img.shields.io/badge/Documentación-Notebook-795548?style=for-the-badge&logo=jupyter)]()  
[![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-E91E63?style=for-the-badge&logo=streamlit)]()  
[![API](https://img.shields.io/badge/🔌_API-Disponible-blue?style=for-the-badge)]()

## 👥 Equipo
- **Integrante 1**: Sharon Obando Gómez 
- **Integrante 2**: Marco Alvarez Quiros
- **Integrante 3**: Fabián Brenes Loría

## 📋 Descripción del Proyecto
Creación de un sistema que prediga qué clientes tienen mayor probabilidad de
abandonar un servicio de telecomunicaciones e identificar clientes en
riesgo clasificandolos por nivel de urgencia para implementar estrategias de retención.

## 🎯 Objetivos
- Realizar análisis exploratorio de datos (EDA) con visualizaciones profesionales
- Implementar preprocesamiento de datos para modelos de machine learning
- Diseñar, entrenar y evaluar redes neuronales artificiales (ANN)
- Comparar múltiples modelos y seleccionar el óptimo según métricas
- Desplegar modelos en producción mediante API REST con FastAPI
- Crear interfaces de usuario con Streamlit
- Documentar proyectos de forma profesional

## 📊 Dataset
- **Fuente**: Telco Customer Churn KAGGLE
- **URL**: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- **Registros**: 7,043 clientes
- **Variables**: 21 (demográficas, servicios contratados, información de cuenta)
- **Variables principales**: antigüedad del cliente, tipo de contrato, método de pago, cargos
mensuales, servicios adicionales (internet, streaming, seguridad),

## 🔧 Instalación

### Requisitos Previos
- Python 3.8+
- pip

### Pasos de Instalación
```bash
# 1. Clonar o descargar el repositorio
cd ProyectoB_SPAC

# 2. Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Descargar el dataset
python data/raw/download_data.py
```

## 🚀 Uso

### Ejecutar Notebooks
```bash
jupyter notebook notebooks/
```
Ejecutar en orden:
1. `01_EDA_CreditRisk.ipynb` - Análisis exploratorio
2. `02_ANN_BinaryClass.ipynb` - Modelo binario
3. `03_ANN_Regression.ipynb` - Modelo regresión
4. `04_Comparacion_Modelos.ipynb` - Evaluación

### Entrenar Modelos
```bash
# Entrenar modelo de clasificación binaria
python src/train/binary_classifier.py

# Entrenar modelo de clasificación multiclase
python src/train/multiclass_classifier.py
```

### Ejecutar API
```bash
cd api
uvicorn main:app --reload
```
La API estará disponible en: http://localhost:8000
Documentación automática: http://localhost:8000/docs

### Ejecutar Frontend
```bash
cd app
streamlit run Home.py
```
El frontend estará disponible en: http://localhost:8501

## 📁 Estructura del Proyecto
```
ProyectoA_RiesgoCrediticio/
├── README.md                          # Este archivo
├── requirements.txt                   # Dependencias Python
├── .gitignore                        # Archivos a ignorar en Git
│
├── data/
│   ├── raw/                          # Datos originales
│   │   └── download_data.py          # Script de descarga
│   └── processed/                    # Datos procesados
│
├── notebooks/                        # Jupyter notebooks
│   ├── 01_EDA_CreditRisk.ipynb      # Análisis exploratorio
│   ├── 02_Preprocesamiento.ipynb     # Preprocesamiento
│   ├── 03_ANN_BinaryClass.ipynb      # Modelo binario
│   ├── 04_ANN_MultiClass.ipynb       # Modelo multiclase
│   └── 05_Comparacion_Modelos.ipynb  # Comparación
│
├── src/                              # Código fuente
│   ├── __init__.py
│   ├── data_prep.py                  # Funciones de preprocesamiento
│   ├── config.py                     # Configuraciones
│   └── train/                        # Scripts de entrenamiento
│       ├── binary_classifier.py
│       ├── Regression_classifier.py
│       └── utils.py
│
├── models/                           # Modelos entrenados
│   ├── binary_model.h5
│   ├── regression_model.keras
│   ├── scaler.pkl
│   └── label_encoder.pkl
│
├── api/                              # API REST
│   ├── main.py                       # Aplicación FastAPI
│   ├── schemas.py                    # Modelos Pydantic
│   └── predict.py                    # Lógica de predicción
│
└── app/                              # Frontend Streamlit
    ├── Home.py                       # Página principal
    └── pages/
        ├── 1_Prediccion_Individual.py
        ├── 2_Analisis_Batch.py
        └── 3_Metricas_Modelos.py
```

## 🧪 Modelos Implementados

### Modelo 1: Clasificación Binaria
- **Objetivo**: Predecir aprobación de crédito (Bueno/Malo)
- **Arquitectura**: Red neuronal artificial (ANN) tipo MLP 
(Multilayer Perceptron) para clasificación binaria
- **Métricas**: 
  - Accuracy: 0.72
  - Precision: 0 = 0.89 y 1 = 0.49
  - Recall: 0 = 0.70 y 1 = 0.76
  - F1-Score: 0 = 0.79 y 1 = 0.59

### Modelo 2: Clasificación Regresión
- **Objetivo**: Clasificar nivel de riesgo (Bajo/Medio/Alto)
- **Arquitectura**: Para crear el Modelo se utilizó **ANN Regresión** secuencial,
la cual está compuesta por 4 capas ocultas con la activación **Relu** y se agrego
una capa de salida, por último utilizando la función de pérdida de **MSE** y **ADAM**.
- **Métricas**: 
  - Accuracy: 0.77
  - Precision macro: 0 = 0.81 y 1 = 0.60
  - Recall macro: 0 = 0.89 y 1 = 0.45
  - F1-Score macro: 0 = 0.85 y 1 = 0.52

## 📈 Resultados

### Conclusiones
Como conclusión, A lo largo de este proyecto se exploró el uso de redes neuronales artificiales para resolver problemas tanto de predicción continua como de clasificación binaria, evidenciando su potencial en contextos reales. Los modelos implementados lograron resultados satisfactorios, permitiendo no solo comprender el comportamiento de los datos, sino también generar predicciones útiles.
Sin embargo, el proceso también permitió identificar limitaciones, como la necesidad de un adecuado preprocesamiento de datos y la sensibilidad del modelo a la elección de parámetros. Estas dificultades resaltan la importancia de una etapa de análisis previo sólida y de la validación constante de los resultados obtenidos.
En conclusión, el uso de ANN constituye una metodología poderosa dentro del aprendizaje automático, pero requiere un enfoque cuidadoso y estructurado para maximizar su efectividad.

### Recomendaciones
Como recomendación, como trabajo futuro, se sugiere explorar arquitecturas más complejas, 
así como comparar con otros algoritmos para fortalecer la toma de decisiones.

## 🛠️ Tecnologías Utilizadas
- **Python 3.x**
- **TensorFlow/Keras**: Redes neuronales
- **Pandas/NumPy**: Manipulación de datos
- **Matplotlib/Seaborn**: Visualización
- **Scikit-learn**: Preprocesamiento y métricas
- **FastAPI**: API REST
- **Streamlit**: Frontend web
- **Uvicorn**: Servidor ASGI

## 📝 Notas de Desarrollo
**Problemas Encontrados**

- Como equipo encontramos que nuestros dataset estaba totalmente limpio, pero creando un gráfico nos dimos cuando que podiamos llegar a tener algunas fallas con los modelos, ya que, nuestra variable objetivo 'Churn' nos presentaba que los 0 = 5174 y de 1 = 1869, dandonos a entender que el modelo iba a llegar a predecir mas un 0 que un 1.

---
**Proyecto desarrollado para el curso de Inteligencia Artificial Aplicada**  
**Colegio Universitario de Cartago (CUC) - 2026**
