# Modelo Híbrido para Predicción de COVID-19

Este repositorio contiene el código y los archivos del Trabajo Fin de Máster sobre predicción de casos de COVID-19 utilizando un modelo híbrido XGBoost-LSTM.

## Estructura del Proyecto

```
TFM_definitivo/
├── README.md                  # Este archivo
├── memoria/                   # Capítulos de la memoria en LaTeX
│   ├── capitulo0_introduccion.tex
│   ├── capitulo1_estado_arte.tex
│   ├── capitulo2_datos_analisis.tex
│   ├── capitulo3_metodologia.tex
│   └── capitulo4_resultados.tex
├── modelo/                    # Código fuente del modelo híbrido
│   ├── config.py              # Configuración de hiperparámetros
│   ├── hybrid_model.py        # Modelo híbrido completo
│   ├── regime_classifier.py   # Clasificador XGBoost de régimen
│   ├── magnitude_predictor.py # Predictores LSTM especializados
│   ├── feature_engineering.py # Extracción de características temporales
│   ├── training_pipeline.py   # Pipeline de entrenamiento
│   └── evaluation.py          # Métricas y visualización
├── notebooks/                 # Notebooks de Jupyter
│   └── MODELO_HIBRIDO_COMPLETO.ipynb
├── modelos/                   # Modelos entrenados
│   ├── hybrid_covid_model_completo_classifier.pkl
│   ├── hybrid_covid_model_completo_predictor_bajada.pth
│   ├── hybrid_covid_model_completo_predictor_estable.pth
│   ├── hybrid_covid_model_completo_predictor_subida.pth
│   └── *.json                 # Métricas y metadatos
└── resultados/                # Visualizaciones y gráficos
    ├── comparacion_4_paises.png
    ├── prediccion_Spain.png
    ├── predicciones_ejemplos.png
    ├── training_curves.png
    └── mapa_paises_*.png
```

## Descripción del Modelo

El modelo híbrido propuesto combina dos enfoques de aprendizaje automático en una arquitectura de dos etapas:

### Etapa 1: Clasificación de Régimen (XGBoost)
- **Objetivo**: Clasificar la serie temporal en tres regímenes epidemiológicos
  - Bajada (casos decrecientes)
  - Estable (casos constantes)
  - Subida (casos crecientes)
- **Entrada**: 26 características temporales extraídas de ventanas de 45 días
- **Salida**: Predicción de régimen futuro

### Etapa 2: Predicción de Magnitud (LSTM Especializadas)
- **Objetivo**: Predecir la magnitud exacta de casos para los próximos 7 días
- **Arquitectura**: Tres redes LSTM independientes, una por cada régimen
- **Especialización**: Cada LSTM se entrena exclusivamente con datos de su régimen

## Características Principales

- **Ingeniería de características avanzada**: 26 features temporales diseñadas específicamente para dinámica epidemiológica
- **Arquitectura bidireccional con atención**: Las LSTMs utilizan mecanismos de atención para enfocarse en los días más relevantes
- **Evaluación rigurosa**: Métricas diferenciadas para casos de continuación vs transición de régimen
- **Validación geográfica**: Evaluación en países completamente no vistos durante el entrenamiento

## Requisitos

```
Python 3.10+
pytorch >= 2.0
xgboost >= 2.0
scikit-learn >= 1.3
pandas >= 2.0
numpy >= 1.24
matplotlib >= 3.7
seaborn >= 0.12
```

## Uso del Modelo

### Cargar modelo entrenado

```python
from modelo.hybrid_model import HybridCOVIDModel
from modelo.config import HybridConfig

config = HybridConfig()
model = HybridCOVIDModel(config=config)

# Cargar pesos
model.load(
    path_prefix='./modelos/hybrid_covid_model_completo',
    input_size=4,
    output_size=7
)

# Realizar predicciones
predictions, regimes = model.predict(X_test)
```

### Entrenar desde cero

Consultar el notebook `notebooks/MODELO_HIBRIDO_COMPLETO.ipynb` para un ejemplo completo de entrenamiento.

## Resultados Principales

El modelo híbrido alcanza las siguientes métricas en el conjunto de test:

- **Accuracy de clasificación**: 85.2% global
  - Bajada: 87.3%
  - Estable: 91.2%
  - Subida: 68.7%

- **Predicción de magnitud**:
  - MAE: 0.0437
  - RMSE: 0.0812
  - Direction Accuracy: 76.8%

- **Análisis de transiciones**:
  - Accuracy en continuaciones: 94.6%
  - Accuracy en transiciones: 40.1%

## Dataset

Los datos utilizados provienen del repositorio **Our World in Data (OWID)**, que recopila datos epidemiológicos de COVID-19 de más de 200 países. Las variables principales son:

- `new_cases_smoothed_per_million` (variable objetivo)
- `new_deaths_smoothed_per_million`
- `reproduction_rate`
- `positive_rate`

## Estructura de la Memoria

La memoria del TFM se compone de 5 capítulos:

1. **Introducción**: Contexto, motivación y objetivos
2. **Estado del Arte**: Fundamentos teóricos y trabajos previos
3. **Datos y Análisis Exploratorio**: Descripción del dataset y selección de variables
4. **Metodología**: Arquitectura del modelo híbrido y técnicas empleadas
5. **Resultados**: Evaluación exhaustiva y análisis de rendimiento

## Autor

Trabajo Fin de Máster desarrollado como parte del Máster en Ciencia de Datos.

## Licencia

Este proyecto es de uso académico.
