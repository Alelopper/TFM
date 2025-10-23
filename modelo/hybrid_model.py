"""
Modelo Híbrido Completo
=======================

Combina RegimeClassifier (XGBoost) + RegimePredictors (LSTM) en una interfaz unificada.
"""

import numpy as np
import torch
from typing import Dict, Tuple

from regime_classifier import RegimeClassifier
from magnitude_predictor import RegimePredictors
from config import HybridConfig


class HybridCOVIDModel:
    """
    Modelo híbrido de 2 etapas:
    1. Clasificar régimen (XGBoost)
    2. Predecir magnitud con LSTM específico
    """
    def __init__(self, config: HybridConfig = None):
        self.config = config or HybridConfig()

        # Componentes del modelo
        self.classifier = RegimeClassifier(config=self.config)
        self.predictors = RegimePredictors(config=self.config)

        self.is_trained = False

    def train(self, X_train: np.ndarray, Y_train: np.ndarray,
             X_val: np.ndarray = None, Y_val: np.ndarray = None,
             cluster_ids_train: np.ndarray = None, cluster_ids_val: np.ndarray = None,
             verbose: bool = True):
        """
        Entrena el modelo híbrido completo.

        Args:
            X_train: [N_train, window_size, n_features]
            Y_train: [N_train, output_size]
            X_val: [N_val, window_size, n_features] (opcional)
            Y_val: [N_val, output_size] (opcional)
            cluster_ids_train: [N_train] Cluster IDs de train (opcional)
            cluster_ids_val: [N_val] Cluster IDs de val (opcional)
            verbose: Imprimir progreso
        """
        if verbose:
            print("\n" + "=" * 70)
            print("ENTRENAMIENTO DEL MODELO HÍBRIDO")
            print("=" * 70)
            self.config.print_config()

        # ===== ETAPA 1: ENTRENAR CLASIFICADOR =====
        if verbose:
            print("\n" + "=" * 70)
            print("ETAPA 1: ENTRENANDO CLASIFICADOR DE RÉGIMEN")
            print("=" * 70)

        self.classifier.train(
            X_train, Y_train,
            X_val, Y_val,
            cluster_ids_train, cluster_ids_val,
            verbose=verbose
        )

        # ===== ETAPA 2: ENTRENAR PREDICTORES =====
        # Necesitamos las etiquetas de régimen para entrenar los LSTMs
        if verbose:
            print("\n" + "=" * 70)
            print("ETAPA 2: ENTRENANDO PREDICTORES LSTM")
            print("=" * 70)

        # Obtener etiquetas de régimen
        regime_labels_train = self.classifier.predict(X_train, cluster_ids_train)

        regime_labels_val = None
        if X_val is not None:
            regime_labels_val = self.classifier.predict(X_val, cluster_ids_val)

        # Entrenar predictores
        self.predictors.train(
            X_train, Y_train, regime_labels_train,
            X_val, Y_val, regime_labels_val,
            verbose=verbose
        )

        self.is_trained = True

        if verbose:
            print("\n" + "=" * 70)
            print("✓ MODELO HÍBRIDO ENTRENADO COMPLETAMENTE")
            print("=" * 70)

    def predict(self, X: np.ndarray, cluster_ids: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predice usando el modelo híbrido.

        Args:
            X: [N, window_size, n_features]
            cluster_ids: [N] Cluster IDs (opcional)

        Returns:
            predictions: [N, output_size] - Predicciones de magnitud
            regime_labels: [N] - Régimen predicho (0, 1, 2)
        """
        if not self.is_trained:
            raise ValueError("Modelo no entrenado. Llama a train() primero.")

        # 1. Clasificar régimen
        regime_labels = self.classifier.predict(X, cluster_ids)

        # 2. Predecir con LSTM apropiado
        predictions = self.predictors.predict(X, regime_labels)

        return predictions, regime_labels

    def predict_with_probabilities(self, X: np.ndarray, cluster_ids: np.ndarray = None) -> Dict:
        """
        Predice con información adicional sobre probabilidades.

        Args:
            X: [N, window_size, n_features]
            cluster_ids: [N] Cluster IDs (opcional)

        Returns:
            Dict con predictions, regime_labels, regime_probas
        """
        if not self.is_trained:
            raise ValueError("Modelo no entrenado. Llama a train() primero.")

        # Clasificar régimen con probabilidades
        regime_labels = self.classifier.predict(X, cluster_ids)
        regime_probas = self.classifier.predict_proba(X, cluster_ids)

        # Predecir magnitud
        predictions = self.predictors.predict(X, regime_labels)

        return {
            'predictions': predictions,
            'regime_labels': regime_labels,
            'regime_probas': regime_probas
        }

    def evaluate(self, X_test: np.ndarray, Y_test: np.ndarray,
                cluster_ids_test: np.ndarray = None, verbose: bool = True) -> dict:
        """
        Evalúa el modelo completo.

        Args:
            X_test: [N, window_size, n_features]
            Y_test: [N, output_size]
            cluster_ids_test: [N] Cluster IDs de test (opcional)
            verbose: Imprimir resultados

        Returns:
            Dict con todas las métricas
        """
        if verbose:
            print("\n" + "=" * 70)
            print("EVALUACIÓN DEL MODELO HÍBRIDO")
            print("=" * 70)

        # Evaluar clasificador
        classifier_metrics = self.classifier.evaluate(X_test, Y_test, cluster_ids_test, verbose=verbose)

        # Evaluar predictores
        regime_labels_test = self.classifier.predict(X_test, cluster_ids_test)
        predictor_metrics = self.predictors.evaluate(
            X_test, Y_test, regime_labels_test, verbose=verbose
        )

        # Métricas combinadas
        predictions, _ = self.predict(X_test, cluster_ids_test)

        # Direction accuracy
        direction_accuracy = self._calculate_direction_accuracy(predictions, Y_test)

        # MAE global
        mae_global = np.abs(predictions - Y_test).mean()

        # RMSE
        rmse_global = np.sqrt(((predictions - Y_test) ** 2).mean())

        if verbose:
            print("\n" + "=" * 70)
            print("MÉTRICAS GLOBALES DEL MODELO HÍBRIDO")
            print("=" * 70)
            print(f"\n✓ Direction Accuracy: {direction_accuracy:.2f}%")
            print(f"✓ MAE Global: {mae_global:.6f}")
            print(f"✓ RMSE Global: {rmse_global:.6f}")

        return {
            'direction_accuracy': float(direction_accuracy),
            'mae_global': float(mae_global),
            'rmse_global': float(rmse_global),
            'classifier_metrics': classifier_metrics,
            'predictor_metrics': predictor_metrics
        }

    def _calculate_direction_accuracy(self, predictions: np.ndarray,
                                      targets: np.ndarray) -> float:
        """
        Calcula el accuracy de dirección.

        Args:
            predictions: [N, output_size]
            targets: [N, output_size]

        Returns:
            Direction accuracy en porcentaje
        """
        # Calcular tendencia: último - primero
        pred_trends = predictions[:, -1] - predictions[:, 0]
        true_trends = targets[:, -1] - targets[:, 0]

        # Clasificar en bajada/estable/subida
        def classify_trend(trends):
            labels = np.zeros_like(trends, dtype=int)
            labels[trends < -0.05] = 0  # Bajada
            labels[trends > 0.05] = 2   # Subida
            labels[(trends >= -0.05) & (trends <= 0.05)] = 1  # Estable
            return labels

        pred_classes = classify_trend(pred_trends)
        true_classes = classify_trend(true_trends)

        # Accuracy
        accuracy = (pred_classes == true_classes).mean() * 100
        return accuracy

    def save(self, path_prefix: str):
        """
        Guarda el modelo completo.

        Args:
            path_prefix: Prefijo para los archivos (ej: './models/hybrid')
        """
        # Guardar clasificador
        classifier_path = f"{path_prefix}_classifier.pkl"
        self.classifier.save(classifier_path)

        # Guardar predictores
        predictors_path_prefix = f"{path_prefix}_predictor"
        self.predictors.save(predictors_path_prefix)

        print(f"\n✓ Modelo híbrido completo guardado con prefijo: {path_prefix}")

    def load(self, path_prefix: str, input_size: int, output_size: int):
        """
        Carga el modelo completo.

        Args:
            path_prefix: Prefijo de los archivos
            input_size: Tamaño de entrada (n_features)
            output_size: Tamaño de salida (días a predecir)
        """
        # Cargar clasificador
        classifier_path = f"{path_prefix}_classifier.pkl"
        self.classifier.load(classifier_path)

        # Cargar predictores
        predictors_path_prefix = f"{path_prefix}_predictor"
        self.predictors.load(predictors_path_prefix, input_size, output_size)

        self.is_trained = True
        print(f"\n✓ Modelo híbrido completo cargado desde: {path_prefix}")


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

