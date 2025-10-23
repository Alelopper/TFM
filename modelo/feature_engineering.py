"""
Feature Engineering para el Clasificador de Régimen
====================================================

Extrae features interpretables de series temporales para XGBoost.
"""

import numpy as np
from typing import List, Dict


def extract_features(window: np.ndarray, cluster_id: int = None) -> np.ndarray:
    """
    Extrae features de una ventana temporal.

    Args:
        window: Array de shape [window_size, n_features]
                Usamos solo la primera feature (casos)
        cluster_id: ID del cluster (opcional, para XGBoost)

    Returns:
        Array de features [n_features_extracted]
    """
    # Usar solo primera feature (casos normalizados)
    series = window[:, 0]

    features = []

    # ===== 1. VALOR ACTUAL =====
    features.append(series[-1])

    # ===== 2. TENDENCIAS (Pendientes) =====
    # Pendiente últimos 7 días
    if len(series) >= 7:
        slope_7 = (series[-1] - series[-7]) / 7
        features.append(slope_7)
    else:
        features.append(0)

    # Pendiente últimos 14 días
    if len(series) >= 14:
        slope_14 = (series[-1] - series[-14]) / 14
        features.append(slope_14)
    else:
        features.append(0)

    # Pendiente últimos 30 días
    if len(series) >= 30:
        slope_30 = (series[-1] - series[-30]) / 30
        features.append(slope_30)
    else:
        features.append(0)

    # ===== 3. ACELERACIÓN (Segunda derivada) =====
    if len(series) >= 14:
        slope_recent = (series[-1] - series[-7]) / 7
        slope_old = (series[-7] - series[-14]) / 7
        acceleration = slope_recent - slope_old
        features.append(acceleration)
    else:
        features.append(0)

    # NUEVO: Aceleración de 2º orden (cambio en la aceleración)
    if len(series) >= 21:
        accel_recent = (series[-1] - series[-7]) / 7 - (series[-7] - series[-14]) / 7
        accel_old = (series[-7] - series[-14]) / 7 - (series[-14] - series[-21]) / 7
        jerk = accel_recent - accel_old
        features.append(jerk)
    else:
        features.append(0)

    # ===== 4. ESTADÍSTICAS RECIENTES =====
    # Media últimos 7 días
    if len(series) >= 7:
        features.append(series[-7:].mean())
        features.append(series[-7:].std())
    else:
        features.append(series.mean())
        features.append(series.std())

    # Máximo y mínimo últimos 14 días
    if len(series) >= 14:
        features.append(series[-14:].max())
        features.append(series[-14:].min())
    else:
        features.append(series.max())
        features.append(series.min())

    # ===== 5. LAGS (Auto-correlación) =====
    for lag in [7, 14, 21, 28]:
        if len(series) > lag:
            features.append(series[-lag])
        else:
            features.append(series[0])

    # ===== 6. VOLATILIDAD =====
    # Cambios relativos recientes
    if len(series) >= 2:
        recent_changes = np.diff(series[-7:]) if len(series) >= 7 else np.diff(series)
        features.append(recent_changes.std())  # Volatilidad
        features.append(recent_changes.mean())  # Cambio medio
    else:
        features.append(0)
        features.append(0)

    # ===== 7. POSICIÓN RELATIVA =====
    # ¿Dónde está el valor actual respecto al rango?
    min_val = series.min()
    max_val = series.max()
    if max_val > min_val:
        relative_position = (series[-1] - min_val) / (max_val - min_val)
        features.append(relative_position)
    else:
        features.append(0.5)

    # NUEVO: ===== 8. CRUCES DE MEDIAS MÓVILES =====
    # Señal alcista/bajista basada en cruces
    if len(series) >= 21:
        ma_short = series[-7:].mean()  # Media corta (7 días)
        ma_long = series[-21:].mean()  # Media larga (21 días)
        ma_cross = ma_short - ma_long  # Positivo = alcista, Negativo = bajista
        features.append(ma_cross)

        # Tendencia de la media corta
        ma_short_slope = (ma_short - series[-14:-7].mean()) / 7
        features.append(ma_short_slope)
    else:
        features.append(0)
        features.append(0)

    # NUEVO: ===== 9. DETECCIÓN DE EXTREMOS LOCALES =====
    # ¿Estamos cerca de un mínimo local? (probable subida)
    # ¿Estamos cerca de un máximo local? (probable bajada)
    if len(series) >= 14:
        recent_window = series[-14:]
        current = series[-1]

        # Distancia al mínimo local
        distance_to_min = (current - recent_window.min()) / (recent_window.std() + 1e-6)
        features.append(distance_to_min)

        # Distancia al máximo local
        distance_to_max = (recent_window.max() - current) / (recent_window.std() + 1e-6)
        features.append(distance_to_max)
    else:
        features.append(0)
        features.append(0)

    # NUEVO: ===== 10. MOMENTUM =====
    # Cambio porcentual acumulado en diferentes ventanas
    if len(series) >= 7:
        momentum_7 = series[-1] - series[-7]
        features.append(momentum_7)
    else:
        features.append(0)

    if len(series) >= 14:
        momentum_14 = series[-1] - series[-14]
        features.append(momentum_14)
    else:
        features.append(0)

    # NUEVO: ===== 11. VOLATILIDAD COMPARATIVA =====
    # Comparar volatilidad reciente vs histórica
    if len(series) >= 28:
        vol_recent = series[-7:].std()
        vol_old = series[-28:-7].std()
        vol_ratio = vol_recent / (vol_old + 1e-6)
        features.append(vol_ratio)
    else:
        features.append(1.0)

    # NUEVO: ===== 12. DÍAS CONSECUTIVOS EN LA MISMA DIRECCIÓN =====
    # ¿Cuántos días seguidos subiendo/bajando?
    if len(series) >= 8:
        diffs = np.diff(series[-8:])

        # Contar días consecutivos en la dirección actual
        consecutive = 0
        direction = np.sign(diffs[-1])

        for d in reversed(diffs):
            if np.sign(d) == direction:
                consecutive += 1
            else:
                break

        features.append(consecutive * direction)  # Positivo si sube, negativo si baja
    else:
        features.append(0)

    # NUEVO: ===== 13. CLUSTER ID =====
    # Añadir cluster como feature categórica para XGBoost
    if cluster_id is not None:
        features.append(float(cluster_id))

    return np.array(features, dtype=np.float32)


def extract_features_batch(X_batch: np.ndarray, cluster_ids: np.ndarray = None) -> np.ndarray:
    """
    Extrae features de un batch.

    Args:
        X_batch: Array de shape [batch_size, window_size, n_features]
        cluster_ids: Array de cluster IDs [batch_size] (opcional)

    Returns:
        Array de shape [batch_size, n_features_extracted]
    """
    features_list = []

    for i in range(len(X_batch)):
        cluster_id = cluster_ids[i] if cluster_ids is not None else None
        features = extract_features(X_batch[i], cluster_id=cluster_id)
        features_list.append(features)

    return np.array(features_list, dtype=np.float32)


def get_feature_names(include_cluster: bool = False) -> List[str]:
    """
    Retorna nombres descriptivos de las features.

    Args:
        include_cluster: Si True, incluye 'cluster_id' en los nombres

    Útil para interpretabilidad del modelo.
    """
    names = [
        'current_value',
        'slope_7d',
        'slope_14d',
        'slope_30d',
        'acceleration',
        'jerk',  # NUEVO
        'mean_7d',
        'std_7d',
        'max_14d',
        'min_14d',
        'lag_7d',
        'lag_14d',
        'lag_21d',
        'lag_28d',
        'volatility',
        'mean_change',
        'relative_position',
        'ma_cross',  # NUEVO
        'ma_short_slope',  # NUEVO
        'distance_to_min',  # NUEVO
        'distance_to_max',  # NUEVO
        'momentum_7',  # NUEVO
        'momentum_14',  # NUEVO
        'vol_ratio',  # NUEVO
        'consecutive_days'  # NUEVO
    ]

    if include_cluster:
        names.append('cluster_id')  # NUEVO

    return names


def calculate_regime_label(Y_sequence: np.ndarray,
                          threshold_low: float = -0.05,
                          threshold_high: float = 0.05) -> int:
    """
    Calcula la etiqueta de régimen basado en la tendencia.

    Args:
        Y_sequence: Secuencia objetivo [output_size]
        threshold_low: Umbral para bajada
        threshold_high: Umbral para subida

    Returns:
        Etiqueta de régimen:
        - 0: Bajada
        - 1: Estable
        - 2: Subida
    """
    # Calcular pendiente general: último - primero
    slope = (Y_sequence[-1] - Y_sequence[0]) / len(Y_sequence)

    if slope < threshold_low:
        return 0  # Bajada
    elif slope > threshold_high:
        return 2  # Subida
    else:
        return 1  # Estable


def calculate_regime_labels_batch(Y_batch: np.ndarray,
                                  threshold_low: float = -0.05,
                                  threshold_high: float = 0.05) -> np.ndarray:
    """
    Calcula etiquetas de régimen para un batch.

    Args:
        Y_batch: Array de shape [batch_size, output_size]
        threshold_low: Umbral para bajada
        threshold_high: Umbral para subida

    Returns:
        Array de etiquetas [batch_size]
    """
    labels = []

    for i in range(len(Y_batch)):
        label = calculate_regime_label(Y_batch[i], threshold_low, threshold_high)
        labels.append(label)

    return np.array(labels, dtype=np.int32)


def calculate_regime_from_window(X_window: np.ndarray,
                                  threshold_low: float = -0.05,
                                  threshold_high: float = 0.05,
                                  lookback: int = 7) -> int:
    """
    Calcula el régimen basado en la VENTANA HISTÓRICA (últimos días de X).

    Esto permite detectar el régimen actual antes de la predicción.

    Args:
        X_window: Ventana histórica [window_size, n_features] o [window_size]
        threshold_low: Umbral para bajada
        threshold_high: Umbral para subida
        lookback: Últimos N días a considerar para calcular tendencia

    Returns:
        Etiqueta de régimen:
        - 0: Bajada
        - 1: Estable
        - 2: Subida
    """
    # Usar solo primera feature (casos)
    if X_window.ndim == 2:
        series = X_window[:, 0]
    else:
        series = X_window

    # Tomar últimos 'lookback' días
    lookback = min(lookback, len(series))
    recent_series = series[-lookback:]

    # Calcular pendiente
    slope = (recent_series[-1] - recent_series[0]) / lookback

    if slope < threshold_low:
        return 0  # Bajada
    elif slope > threshold_high:
        return 2  # Subida
    else:
        return 1  # Estable


def calculate_regime_from_window_batch(X_batch: np.ndarray,
                                       threshold_low: float = -0.05,
                                       threshold_high: float = 0.05,
                                       lookback: int = 7) -> np.ndarray:
    """
    Calcula régimen del histórico para un batch.

    Args:
        X_batch: Array de shape [batch_size, window_size, n_features]
        threshold_low: Umbral para bajada
        threshold_high: Umbral para subida
        lookback: Últimos N días a considerar

    Returns:
        Array de etiquetas [batch_size]
    """
    labels = []

    for i in range(len(X_batch)):
        label = calculate_regime_from_window(
            X_batch[i], threshold_low, threshold_high, lookback
        )
        labels.append(label)

    return np.array(labels, dtype=np.int32)


def detect_regime_transitions(historical_regimes: np.ndarray,
                              future_regimes: np.ndarray) -> np.ndarray:
    """
    Detecta CAMBIOS de régimen entre histórico y futuro.

    Args:
        historical_regimes: Régimen de la ventana histórica [N]
        future_regimes: Régimen del futuro (Y_target) [N]

    Returns:
        Máscara booleana [N] donde True = hubo cambio de régimen
    """
    transitions = historical_regimes != future_regimes
    return transitions


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

