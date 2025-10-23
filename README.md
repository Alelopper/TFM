# Modelo Híbrido para Predicción de COVID-19

Este repositorio contiene el código y los archivos del Trabajo Fin de Máster sobre predicción de casos de COVID-19 utilizando un modelo híbrido XGBoost-LSTM.

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

### Entrenar desde cero

Consultar el notebook `notebooks/MODELO_HIBRIDO_COMPLETO.ipynb` para un ejemplo completo de entrenamiento.


## Licencia

Este proyecto es de uso académico.
