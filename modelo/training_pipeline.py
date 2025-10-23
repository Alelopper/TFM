"""
Pipeline de Entrenamiento Completo
===================================

Pipeline end-to-end para entrenar el modelo híbrido desde datos raw hasta modelo entrenado.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import json
import os
from pathlib import Path

from config import HybridConfig
from hybrid_model import HybridCOVIDModel


def create_sequences(df: pd.DataFrame,
                     window_size: int,
                     output_size: int,
                     input_columns: List[str],
                     output_column: str,
                     stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crea secuencias X, Y para entrenamiento (MEJORADO: múltiples features de entrada).

    Args:
        df: DataFrame con datos del país
        window_size: Tamaño de ventana histórica
        output_size: Días a predecir
        input_columns: Lista de columnas a usar como features
        output_column: Columna objetivo a predecir
        stride: Paso entre secuencias

    Returns:
        X: [N, window_size, n_features]
        Y: [N, output_size]
    """
    # Verificar que columnas existen
    available_cols = [col for col in input_columns if col in df.columns]

    if len(available_cols) == 0:
        # Fallback: usar solo la primera columna de input_columns
        available_cols = [input_columns[0]]

    # Extraer datos de features
    X_data = df[available_cols].values  # [n_timesteps, n_features]

    # Target: solo la columna de salida
    Y_data = df[output_column].values  # [n_timesteps]

    X_list = []
    Y_list = []

    for i in range(0, len(X_data) - window_size - output_size + 1, stride):
        X_window = X_data[i:i + window_size]  # [window_size, n_features]
        Y_target = Y_data[i + window_size:i + window_size + output_size]  # [output_size]

        # Validar que no hay NaNs
        if not (np.isnan(X_window).any() or np.isnan(Y_target).any()):
            X_list.append(X_window)
            Y_list.append(Y_target)

    if len(X_list) == 0:
        # Retornar arrays vacíos con la forma correcta
        return np.array([]).reshape(0, window_size, len(available_cols)), np.array([]).reshape(0, output_size)

    return np.array(X_list), np.array(Y_list)


def prepare_data_from_dict(dic_paises: Dict[str, pd.DataFrame],
                           paises_seleccionados: List[str],
                           config: HybridConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepara datos de múltiples países (MEJORADO: soporta múltiples features).

    Args:
        dic_paises: Diccionario {pais: DataFrame}
        paises_seleccionados: Lista de países a usar
        config: Configuración

    Returns:
        X: [N, window_size, n_features]
        Y: [N, output_size]
    """
    X_all = []
    Y_all = []

    for pais in paises_seleccionados:
        if pais not in dic_paises:
            continue

        df = dic_paises[pais].copy()

        # Normalizar TODAS las columnas de entrada
        for col in config.INPUT_COLUMNS:
            if col not in df.columns:
                continue

            mean_val = df[col].mean()
            std_val = df[col].std()

            if std_val > 0:
                df[col] = (df[col] - mean_val) / std_val
            else:
                df[col] = 0.0  # Si std=0, poner a cero

        # Normalizar columna de salida también
        output_col = config.OUTPUT_COLUMN
        if output_col in df.columns:
            mean_val = df[output_col].mean()
            std_val = df[output_col].std()

            if std_val > 0:
                df[output_col] = (df[output_col] - mean_val) / std_val

        # Crear secuencias con múltiples features
        X_pais, Y_pais = create_sequences(
            df,
            window_size=config.WINDOW_SIZE,
            output_size=config.OUTPUT_SIZE,
            input_columns=config.INPUT_COLUMNS,
            output_column=config.OUTPUT_COLUMN,
            stride=config.STRIDE
        )

        if len(X_pais) > 0:
            X_all.append(X_pais)
            Y_all.append(Y_pais)

    if len(X_all) == 0:
        # Retornar arrays vacíos
        n_features = len(config.INPUT_COLUMNS)
        return np.array([]).reshape(0, config.WINDOW_SIZE, n_features), np.array([]).reshape(0, config.OUTPUT_SIZE)

    X_combined = np.concatenate(X_all, axis=0)
    Y_combined = np.concatenate(Y_all, axis=0)

    return X_combined, Y_combined


def split_data(X: np.ndarray, Y: np.ndarray,
              train_ratio: float = 0.7,
              val_ratio: float = 0.15,
              test_ratio: float = 0.15) -> Dict:
    """
    Divide datos en train/val/test.

    Args:
        X: [N, window_size, n_features]
        Y: [N, output_size]
        train_ratio: Proporción de entrenamiento
        val_ratio: Proporción de validación
        test_ratio: Proporción de test

    Returns:
        Dict con X_train, Y_train, X_val, Y_val, X_test, Y_test
    """
    n = len(X)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    X_train = X[:n_train]
    Y_train = Y[:n_train]

    X_val = X[n_train:n_train + n_val]
    Y_val = Y[n_train:n_train + n_val]

    X_test = X[n_train + n_val:]
    Y_test = Y[n_train + n_val:]

    return {
        'X_train': X_train,
        'Y_train': Y_train,
        'X_val': X_val,
        'Y_val': Y_val,
        'X_test': X_test,
        'Y_test': Y_test
    }


def train_hybrid_model(X_train: np.ndarray, Y_train: np.ndarray,
                      X_val: np.ndarray, Y_val: np.ndarray,
                      X_test: np.ndarray, Y_test: np.ndarray,
                      config: HybridConfig = None,
                      save_path: str = None,
                      verbose: bool = True) -> Tuple[HybridCOVIDModel, Dict]:
    """
    Entrena el modelo híbrido completo.

    Args:
        X_train, Y_train: Datos de entrenamiento
        X_val, Y_val: Datos de validación
        X_test, Y_test: Datos de test
        config: Configuración
        save_path: Ruta para guardar modelo (opcional)
        verbose: Imprimir progreso

    Returns:
        model: Modelo entrenado
        metrics: Métricas de evaluación
    """
    config = config or HybridConfig()

    if verbose:
        print("\n" + "=" * 70)
        print("PIPELINE DE ENTRENAMIENTO DEL MODELO HÍBRIDO")
        print("=" * 70)
        print(f"\nDataset sizes:")
        print(f"  Train: {len(X_train)} samples")
        print(f"  Val:   {len(X_val)} samples")
        print(f"  Test:  {len(X_test)} samples")

    # Crear modelo
    model = HybridCOVIDModel(config=config)

    # Entrenar
    model.train(X_train, Y_train, X_val, Y_val, verbose=verbose)

    # Evaluar en test
    metrics = model.evaluate(X_test, Y_test, verbose=verbose)

    # Guardar si se especifica
    if save_path:
        # Crear directorio si no existe
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        # Guardar modelo
        model.save(save_path)

        # Guardar métricas
        metrics_path = f"{save_path}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n✓ Métricas guardadas en {metrics_path}")

    return model, metrics


def train_from_dict(dic_paises: Dict[str, pd.DataFrame],
                   paises_train: List[str],
                   paises_val: List[str],
                   paises_test: List[str],
                   config: HybridConfig = None,
                   save_path: str = None,
                   verbose: bool = True) -> Tuple[HybridCOVIDModel, Dict]:
    """
    Pipeline completo desde diccionario de países.

    Args:
        dic_paises: {pais: DataFrame}
        paises_train: Lista de países para entrenamiento
        paises_val: Lista de países para validación
        paises_test: Lista de países para test
        config: Configuración
        save_path: Ruta para guardar modelo
        verbose: Imprimir progreso

    Returns:
        model: Modelo entrenado
        metrics: Métricas de evaluación
    """
    config = config or HybridConfig()

    if verbose:
        print("\n" + "=" * 70)
        print("PREPARANDO DATOS DESDE DICCIONARIO DE PAÍSES")
        print("=" * 70)

    # Preparar datos
    X_train, Y_train = prepare_data_from_dict(dic_paises, paises_train, config)
    X_val, Y_val = prepare_data_from_dict(dic_paises, paises_val, config)
    X_test, Y_test = prepare_data_from_dict(dic_paises, paises_test, config)

    if verbose:
        print(f"\n✓ Datos preparados:")
        print(f"  Train: {len(X_train)} secuencias de {len(paises_train)} países")
        print(f"  Val:   {len(X_val)} secuencias de {len(paises_val)} países")
        print(f"  Test:  {len(X_test)} secuencias de {len(paises_test)} países")

    # Entrenar
    return train_hybrid_model(
        X_train, Y_train,
        X_val, Y_val,
        X_test, Y_test,
        config=config,
        save_path=save_path,
        verbose=verbose
    )


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

