"""
Evaluación y Visualización del Modelo Híbrido
==============================================

Herramientas para evaluar y visualizar resultados del modelo híbrido.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from config import HybridConfig
from feature_engineering import (
    calculate_regime_from_window_batch,
    detect_regime_transitions,
    calculate_regime_labels_batch
)


def calculate_comprehensive_metrics(predictions: np.ndarray,
                                    targets: np.ndarray,
                                    regime_labels_pred: np.ndarray = None,
                                    regime_labels_true: np.ndarray = None,
                                    X_historical: np.ndarray = None,
                                    config: HybridConfig = None) -> Dict:
    """
    Calcula métricas comprehensivas.

    Args:
        predictions: [N, output_size]
        targets: [N, output_size]
        regime_labels_pred: [N] (opcional)
        regime_labels_true: [N] (opcional)
        X_historical: [N, window_size, n_features] (opcional, para calcular transiciones)
        config: HybridConfig (opcional)

    Returns:
        Dict con todas las métricas
    """
    metrics = {}
    config = config or HybridConfig()

    # ===== MÉTRICAS DE MAGNITUD =====
    mae = np.abs(predictions - targets).mean()
    rmse = np.sqrt(((predictions - targets) ** 2).mean())
    mape = np.abs((predictions - targets) / (targets + 1e-8)).mean() * 100

    # Por horizonte temporal
    mae_by_horizon = []
    for t in range(predictions.shape[1]):
        mae_t = np.abs(predictions[:, t] - targets[:, t]).mean()
        mae_by_horizon.append(float(mae_t))

    metrics['mae'] = float(mae)
    metrics['rmse'] = float(rmse)
    metrics['mape'] = float(mape)
    metrics['mae_by_horizon'] = mae_by_horizon

    # ===== MÉTRICAS DE DIRECCIÓN =====
    direction_accuracy = calculate_direction_accuracy(predictions, targets)
    metrics['direction_accuracy'] = float(direction_accuracy)

    # Dirección por horizonte
    direction_by_horizon = []
    for t in range(1, predictions.shape[1]):
        pred_dir = predictions[:, t] - predictions[:, t-1]
        true_dir = targets[:, t] - targets[:, t-1]
        acc = ((pred_dir > 0) == (true_dir > 0)).mean() * 100
        direction_by_horizon.append(float(acc))

    metrics['direction_by_horizon'] = direction_by_horizon

    # ===== MÉTRICAS DE RÉGIMEN (si están disponibles) =====
    if regime_labels_pred is not None and regime_labels_true is not None:
        regime_accuracy = (regime_labels_pred == regime_labels_true).mean() * 100
        metrics['regime_accuracy'] = float(regime_accuracy)

        # Por clase
        regime_names = ['Bajada', 'Estable', 'Subida']
        regime_metrics = {}
        for label in [0, 1, 2]:
            mask = regime_labels_true == label
            if mask.sum() > 0:
                acc = (regime_labels_pred[mask] == regime_labels_true[mask]).mean() * 100
                regime_metrics[regime_names[label]] = float(acc)

        metrics['regime_accuracy_by_class'] = regime_metrics

        # NUEVO: ===== MÉTRICAS DE TRANSICIONES =====
        if X_historical is not None:
            # Calcular régimen del histórico
            historical_regimes = calculate_regime_from_window_batch(
                X_historical,
                threshold_low=config.REGIME_THRESHOLD_LOW,
                threshold_high=config.REGIME_THRESHOLD_HIGH,
                lookback=7
            )

            # Detectar transiciones
            transitions_mask = detect_regime_transitions(historical_regimes, regime_labels_true)
            n_transitions = transitions_mask.sum()
            n_continuations = (~transitions_mask).sum()

            metrics['n_transitions'] = int(n_transitions)
            metrics['n_continuations'] = int(n_continuations)
            metrics['transition_ratio'] = float(n_transitions / len(predictions) * 100) if len(predictions) > 0 else 0.0

            # Accuracy en transiciones
            if n_transitions > 0:
                transition_accuracy = (regime_labels_pred[transitions_mask] == regime_labels_true[transitions_mask]).mean() * 100
                metrics['transition_accuracy'] = float(transition_accuracy)

                # Por clase en transiciones
                transition_class_metrics = {}
                for label in [0, 1, 2]:
                    mask = (regime_labels_true == label) & transitions_mask
                    if mask.sum() > 0:
                        acc = (regime_labels_pred[mask] == regime_labels_true[mask]).mean() * 100
                        transition_class_metrics[regime_names[label]] = float(acc)
                metrics['transition_accuracy_by_class'] = transition_class_metrics

            # Accuracy en continuaciones
            if n_continuations > 0:
                continuation_accuracy = (regime_labels_pred[~transitions_mask] == regime_labels_true[~transitions_mask]).mean() * 100
                metrics['continuation_accuracy'] = float(continuation_accuracy)

    return metrics


def calculate_direction_accuracy(predictions: np.ndarray,
                                 targets: np.ndarray,
                                 threshold: float = 0.05) -> float:
    """
    Calcula direction accuracy con umbral.

    Args:
        predictions: [N, output_size]
        targets: [N, output_size]
        threshold: Umbral para clasificar estable

    Returns:
        Accuracy en porcentaje
    """
    # Tendencia: último - primero
    pred_trends = predictions[:, -1] - predictions[:, 0]
    true_trends = targets[:, -1] - targets[:, 0]

    # Clasificar
    def classify(trends):
        labels = np.zeros(len(trends), dtype=int)
        labels[trends < -threshold] = 0  # Bajada
        labels[trends > threshold] = 2   # Subida
        labels[(trends >= -threshold) & (trends <= threshold)] = 1  # Estable
        return labels

    pred_classes = classify(pred_trends)
    true_classes = classify(true_trends)

    accuracy = (pred_classes == true_classes).mean() * 100
    return accuracy


def plot_predictions_vs_actual(predictions: np.ndarray,
                               targets: np.ndarray,
                               regime_labels: np.ndarray = None,
                               n_samples: int = 6,
                               figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """
    Visualiza predicciones vs valores reales.

    Args:
        predictions: [N, output_size]
        targets: [N, output_size]
        regime_labels: [N] (opcional)
        n_samples: Número de ejemplos a mostrar
        figsize: Tamaño de figura

    Returns:
        Figura de matplotlib
    """
    regime_names = {0: 'Bajada', 1: 'Estable', 2: 'Subida'}
    colors = {0: 'red', 1: 'gray', 2: 'green'}

    n_samples = min(n_samples, len(predictions))
    indices = np.random.choice(len(predictions), n_samples, replace=False)

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        ax = axes[i]

        # Plotear
        ax.plot(predictions[idx], 'o-', label='Predicción', color='blue', linewidth=2)
        ax.plot(targets[idx], 's-', label='Real', color='orange', linewidth=2)

        # Título con régimen si está disponible
        title = f"Ejemplo {idx}"
        if regime_labels is not None:
            regime = regime_labels[idx]
            regime_name = regime_names.get(regime, 'Desconocido')
            color = colors.get(regime, 'black')
            title += f" - {regime_name}"
            ax.set_facecolor((0.95, 0.95, 0.95) if regime == 1 else
                            (1.0, 0.95, 0.95) if regime == 0 else
                            (0.95, 1.0, 0.95))

        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Día')
        ax.set_ylabel('Casos (normalizados)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_metrics_by_horizon(predictions: np.ndarray,
                            targets: np.ndarray,
                            figsize: Tuple[int, int] = (12, 5)) -> plt.Figure:
    """
    Visualiza métricas por horizonte temporal.

    Args:
        predictions: [N, output_size]
        targets: [N, output_size]
        figsize: Tamaño de figura

    Returns:
        Figura de matplotlib
    """
    output_size = predictions.shape[1]
    days = np.arange(1, output_size + 1)

    # Calcular métricas por día
    mae_by_day = []
    direction_by_day = []

    for t in range(output_size):
        mae = np.abs(predictions[:, t] - targets[:, t]).mean()
        mae_by_day.append(mae)

        if t > 0:
            pred_dir = predictions[:, t] - predictions[:, t-1]
            true_dir = targets[:, t] - targets[:, t-1]
            acc = ((pred_dir > 0) == (true_dir > 0)).mean() * 100
            direction_by_day.append(acc)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # MAE por día
    ax1.plot(days, mae_by_day, 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Día de Predicción', fontweight='bold')
    ax1.set_ylabel('MAE', fontweight='bold')
    ax1.set_title('Error Absoluto Medio por Horizonte', fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Direction accuracy por día
    if len(direction_by_day) > 0:
        ax2.plot(days[1:], direction_by_day, 'o-', linewidth=2, markersize=8, color='green')
        ax2.axhline(y=50, color='red', linestyle='--', label='Random (50%)')
        ax2.set_xlabel('Día de Predicción', fontweight='bold')
        ax2.set_ylabel('Direction Accuracy (%)', fontweight='bold')
        ax2.set_title('Precisión de Dirección por Horizonte', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_regime_confusion_matrix(regime_labels_pred: np.ndarray,
                                 regime_labels_true: np.ndarray,
                                 figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
    """
    Visualiza matriz de confusión de regímenes.

    Args:
        regime_labels_pred: [N]
        regime_labels_true: [N]
        figsize: Tamaño de figura

    Returns:
        Figura de matplotlib
    """
    regime_names = ['Bajada', 'Estable', 'Subida']

    cm = confusion_matrix(regime_labels_true, regime_labels_pred)

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=regime_names,
                yticklabels=regime_names,
                ax=ax)

    ax.set_xlabel('Predicho', fontweight='bold')
    ax.set_ylabel('Real', fontweight='bold')
    ax.set_title('Matriz de Confusión - Clasificación de Régimen', fontweight='bold')

    plt.tight_layout()
    return fig


def plot_transition_analysis(regime_labels_pred: np.ndarray,
                             regime_labels_true: np.ndarray,
                             X_historical: np.ndarray,
                             config: HybridConfig = None,
                             figsize: Tuple[int, int] = (16, 6)) -> plt.Figure:
    """
    NUEVO: Visualiza análisis de transiciones vs continuaciones.

    Args:
        regime_labels_pred: [N] - Predicciones de régimen
        regime_labels_true: [N] - Régimen verdadero
        X_historical: [N, window_size, n_features] - Datos históricos
        config: HybridConfig
        figsize: Tamaño de figura

    Returns:
        Figura de matplotlib con 2 matrices de confusión (transiciones y continuaciones)
    """
    config = config or HybridConfig()
    regime_names = ['Bajada', 'Estable', 'Subida']

    # Calcular régimen histórico
    historical_regimes = calculate_regime_from_window_batch(
        X_historical,
        threshold_low=config.REGIME_THRESHOLD_LOW,
        threshold_high=config.REGIME_THRESHOLD_HIGH,
        lookback=7
    )

    # Detectar transiciones
    transitions_mask = detect_regime_transitions(historical_regimes, regime_labels_true)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Matriz de confusión para TRANSICIONES
    if transitions_mask.sum() > 0:
        cm_transitions = confusion_matrix(
            regime_labels_true[transitions_mask],
            regime_labels_pred[transitions_mask]
        )

        sns.heatmap(cm_transitions, annot=True, fmt='d', cmap='Reds',
                    xticklabels=regime_names,
                    yticklabels=regime_names,
                    ax=ax1)

        ax1.set_xlabel('Predicho', fontweight='bold')
        ax1.set_ylabel('Real', fontweight='bold')
        acc_trans = (regime_labels_pred[transitions_mask] == regime_labels_true[transitions_mask]).mean() * 100
        ax1.set_title(f'TRANSICIONES (Cambio de Régimen)\nAcc: {acc_trans:.1f}% | N={transitions_mask.sum()}',
                     fontweight='bold')

    # Matriz de confusión para CONTINUACIONES
    continuations_mask = ~transitions_mask
    if continuations_mask.sum() > 0:
        cm_continuations = confusion_matrix(
            regime_labels_true[continuations_mask],
            regime_labels_pred[continuations_mask]
        )

        sns.heatmap(cm_continuations, annot=True, fmt='d', cmap='Greens',
                    xticklabels=regime_names,
                    yticklabels=regime_names,
                    ax=ax2)

        ax2.set_xlabel('Predicho', fontweight='bold')
        ax2.set_ylabel('Real', fontweight='bold')
        acc_cont = (regime_labels_pred[continuations_mask] == regime_labels_true[continuations_mask]).mean() * 100
        ax2.set_title(f'CONTINUACIONES (Mismo Régimen)\nAcc: {acc_cont:.1f}% | N={continuations_mask.sum()}',
                     fontweight='bold')

    plt.tight_layout()
    return fig


def generate_evaluation_report(predictions: np.ndarray,
                               targets: np.ndarray,
                               regime_labels_pred: np.ndarray = None,
                               regime_labels_true: np.ndarray = None,
                               X_historical: np.ndarray = None,
                               config: HybridConfig = None,
                               save_path: str = None) -> Dict:
    """
    Genera reporte completo de evaluación.

    Args:
        predictions: [N, output_size]
        targets: [N, output_size]
        regime_labels_pred: [N] (opcional)
        regime_labels_true: [N] (opcional)
        X_historical: [N, window_size, n_features] (opcional)
        config: HybridConfig (opcional)
        save_path: Ruta para guardar figuras (opcional)

    Returns:
        Dict con métricas
    """
    print("=" * 70)
    print("REPORTE DE EVALUACIÓN COMPLETO")
    print("=" * 70)

    # Calcular métricas
    metrics = calculate_comprehensive_metrics(
        predictions, targets,
        regime_labels_pred, regime_labels_true,
        X_historical, config
    )

    # Imprimir métricas
    print(f"\n📊 MÉTRICAS DE MAGNITUD:")
    print(f"  MAE:  {metrics['mae']:.6f}")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  MAPE: {metrics['mape']:.2f}%")

    print(f"\n📊 MÉTRICAS DE DIRECCIÓN:")
    print(f"  Direction Accuracy: {metrics['direction_accuracy']:.2f}%")

    if 'regime_accuracy' in metrics:
        print(f"\n📊 MÉTRICAS DE RÉGIMEN:")
        print(f"  Régimen Accuracy: {metrics['regime_accuracy']:.2f}%")
        print(f"\n  Por clase:")
        for name, acc in metrics['regime_accuracy_by_class'].items():
            print(f"    {name:8s}: {acc:.2f}%")

        # NUEVO: Mostrar métricas de transiciones
        if 'transition_accuracy' in metrics:
            print(f"\n🔄 MÉTRICAS DE TRANSICIÓN (casos difíciles):")
            print(f"  Total muestras: {len(predictions)}")
            print(f"  Transiciones (cambio régimen): {metrics['n_transitions']} ({metrics['transition_ratio']:.1f}%)")
            print(f"  Continuaciones (mismo régimen): {metrics['n_continuations']} ({100-metrics['transition_ratio']:.1f}%)")
            print(f"\n  ⚠️  Accuracy en TRANSICIONES: {metrics['transition_accuracy']:.2f}%")

            if 'transition_accuracy_by_class' in metrics:
                print(f"      Por clase en transiciones:")
                for name, acc in metrics['transition_accuracy_by_class'].items():
                    print(f"        {name:8s}: {acc:.2f}%")

            if 'continuation_accuracy' in metrics:
                print(f"\n  ✓  Accuracy en CONTINUACIONES: {metrics['continuation_accuracy']:.2f}%")

    # Generar visualizaciones
    print(f"\n📈 Generando visualizaciones...")

    fig1 = plot_predictions_vs_actual(predictions, targets, regime_labels_pred)
    if save_path:
        fig1.savefig(f"{save_path}_predictions.png", dpi=150, bbox_inches='tight')
        print(f"  ✓ Guardado: {save_path}_predictions.png")

    fig2 = plot_metrics_by_horizon(predictions, targets)
    if save_path:
        fig2.savefig(f"{save_path}_metrics_by_horizon.png", dpi=150, bbox_inches='tight')
        print(f"  ✓ Guardado: {save_path}_metrics_by_horizon.png")

    if regime_labels_pred is not None and regime_labels_true is not None:
        fig3 = plot_regime_confusion_matrix(regime_labels_pred, regime_labels_true)
        if save_path:
            fig3.savefig(f"{save_path}_confusion_matrix.png", dpi=150, bbox_inches='tight')
            print(f"  ✓ Guardado: {save_path}_confusion_matrix.png")

        # NUEVO: Visualización de transiciones
        if X_historical is not None:
            fig4 = plot_transition_analysis(regime_labels_pred, regime_labels_true, X_historical, config)
            if save_path:
                fig4.savefig(f"{save_path}_transition_analysis.png", dpi=150, bbox_inches='tight')
                print(f"  ✓ Guardado: {save_path}_transition_analysis.png")

    print(f"\n{'='*70}")
    print("REPORTE COMPLETADO")
    print(f"{'='*70}")

    return metrics


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

