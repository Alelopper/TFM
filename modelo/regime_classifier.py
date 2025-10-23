"""
Clasificador de Régimen usando XGBoost
=======================================

Clasifica series temporales en 3 regímenes:
- 0: Bajada (casos decrecientes)
- 1: Estable (casos constantes)
- 2: Subida (casos crecientes)
"""

import xgboost as xgb
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import pickle
from typing import Tuple

from feature_engineering import (
    extract_features_batch,
    calculate_regime_labels_batch,
    calculate_regime_from_window_batch,
    detect_regime_transitions
)
from config import HybridConfig


class RegimeClassifier:
    """
    Clasificador de régimen usando XGBoost.
    """
    def __init__(self, config: HybridConfig = None):
        self.config = config or HybridConfig()

        self.model = xgb.XGBClassifier(
            n_estimators=self.config.XGB_N_ESTIMATORS,
            max_depth=self.config.XGB_MAX_DEPTH,
            learning_rate=self.config.XGB_LEARNING_RATE,
            min_child_weight=self.config.XGB_MIN_CHILD_WEIGHT,
            subsample=self.config.XGB_SUBSAMPLE,
            colsample_bytree=self.config.XGB_COLSAMPLE_BYTREE,
            objective='multi:softmax',
            num_class=3,
            random_state=42,
            eval_metric='mlogloss',
            enable_categorical=True  # NUEVO: Soportar features categóricas (cluster)
        )

        self.is_trained = False
        self.use_cluster = self.config.USE_CLUSTER_FEATURE

    def prepare_data(self, X: np.ndarray, Y: np.ndarray, cluster_ids: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepara datos para entrenamiento.

        Args:
            X: [N, window_size, n_features]
            Y: [N, output_size]
            cluster_ids: [N] Cluster IDs (opcional)

        Returns:
            X_features: [N, n_features_extracted]
            y_labels: [N]
        """
        # Extraer features (incluyendo cluster si está disponible)
        if self.use_cluster and cluster_ids is not None:
            X_features = extract_features_batch(X, cluster_ids=cluster_ids)
        else:
            X_features = extract_features_batch(X)

        # Calcular labels de régimen
        y_labels = calculate_regime_labels_batch(
            Y,
            threshold_low=self.config.REGIME_THRESHOLD_LOW,
            threshold_high=self.config.REGIME_THRESHOLD_HIGH
        )

        return X_features, y_labels

    def train(self, X_train: np.ndarray, Y_train: np.ndarray,
             X_val: np.ndarray = None, Y_val: np.ndarray = None,
             cluster_ids_train: np.ndarray = None, cluster_ids_val: np.ndarray = None,
             verbose: bool = True):
        """
        Entrena el clasificador.

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
            print("ENTRENANDO CLASIFICADOR DE RÉGIMEN (XGBoost)")
            if self.use_cluster and cluster_ids_train is not None:
                print("🔵 USANDO CLUSTER COMO FEATURE")
            print("=" * 70)

        # Preparar datos
        X_train_features, y_train_labels = self.prepare_data(X_train, Y_train, cluster_ids_train)

        if verbose:
            print(f"\nDatos de entrenamiento:")
            print(f"  X_features shape: {X_train_features.shape}")
            print(f"  y_labels shape: {y_train_labels.shape}")

            # Distribución de clases
            unique, counts = np.unique(y_train_labels, return_counts=True)
            print(f"\n  Distribución de clases:")
            for label, count in zip(unique, counts):
                regime_name = ['Bajada', 'Estable', 'Subida'][label]
                print(f"    {label} ({regime_name:8s}): {count:6d} ({count/len(y_train_labels)*100:.1f}%)")

        # NUEVO: Calcular pesos para balancear clases
        # Dar más peso a las clases minoritarias (especialmente subidas)
        unique, counts = np.unique(y_train_labels, return_counts=True)
        class_weights = {}
        total_samples = len(y_train_labels)

        for label, count in zip(unique, counts):
            # Peso inversamente proporcional a la frecuencia
            class_weights[label] = total_samples / (len(unique) * count)

        # Crear array de pesos por muestra
        sample_weights = np.array([class_weights[label] for label in y_train_labels])

        if verbose:
            print(f"\n  Pesos de clase (para balanceo):")
            for label, weight in class_weights.items():
                regime_name = ['Bajada', 'Estable', 'Subida'][label]
                print(f"    {label} ({regime_name:8s}): {weight:.3f}")

        # Evaluar en validación si está disponible
        eval_set = None
        if X_val is not None and Y_val is not None:
            X_val_features, y_val_labels = self.prepare_data(X_val, Y_val, cluster_ids_val)
            eval_set = [(X_train_features, y_train_labels),
                       (X_val_features, y_val_labels)]

        # Entrenar con pesos de muestra (MEJORADO: Balanceo de clases)
        self.model.fit(
            X_train_features,
            y_train_labels,
            sample_weight=sample_weights,  # NUEVO: Balancear clases
            eval_set=eval_set,
            verbose=verbose
        )

        self.is_trained = True

        if verbose:
            print("\n✓ Clasificador entrenado")

            # Accuracy en train
            y_pred_train = self.model.predict(X_train_features)
            train_acc = (y_pred_train == y_train_labels).mean() * 100
            print(f"  Train Accuracy: {train_acc:.2f}%")

            # Accuracy en val si está disponible
            if X_val is not None:
                y_pred_val = self.model.predict(X_val_features)
                val_acc = (y_pred_val == y_val_labels).mean() * 100
                print(f"  Val Accuracy: {val_acc:.2f}%")

    def predict(self, X: np.ndarray, cluster_ids: np.ndarray = None) -> np.ndarray:
        """
        Predice régimen para nuevos datos.

        Args:
            X: [N, window_size, n_features]
            cluster_ids: [N] Cluster IDs (opcional)

        Returns:
            labels: [N] - Etiquetas de régimen (0, 1, 2)
        """
        if not self.is_trained:
            raise ValueError("Modelo no entrenado. Llama a train() primero.")

        if self.use_cluster and cluster_ids is not None:
            X_features = extract_features_batch(X, cluster_ids=cluster_ids)
        else:
            X_features = extract_features_batch(X)

        labels = self.model.predict(X_features)
        return labels

    def predict_proba(self, X: np.ndarray, cluster_ids: np.ndarray = None) -> np.ndarray:
        """
        Predice probabilidades de cada régimen.

        Args:
            X: [N, window_size, n_features]
            cluster_ids: [N] Cluster IDs (opcional)

        Returns:
            probas: [N, 3] - Probabilidades para cada clase
        """
        if not self.is_trained:
            raise ValueError("Modelo no entrenado. Llama a train() primero.")

        if self.use_cluster and cluster_ids is not None:
            X_features = extract_features_batch(X, cluster_ids=cluster_ids)
        else:
            X_features = extract_features_batch(X)

        probas = self.model.predict_proba(X_features)
        return probas

    def evaluate(self, X_test: np.ndarray, Y_test: np.ndarray, cluster_ids_test: np.ndarray = None, verbose: bool = True) -> dict:
        """
        Evalúa el clasificador en datos de test.

        Args:
            X_test: [N, window_size, n_features]
            Y_test: [N, output_size]
            cluster_ids_test: [N] Cluster IDs de test (opcional)
            verbose: Imprimir resultados

        Returns:
            Diccionario con métricas
        """
        # Preparar datos
        X_test_features, y_test_labels = self.prepare_data(X_test, Y_test, cluster_ids_test)

        # Predecir
        y_pred = self.model.predict(X_test_features)

        # Métricas
        accuracy = (y_pred == y_test_labels).mean() * 100

        # Por clase
        regime_names = ['Bajada', 'Estable', 'Subida']
        class_accuracies = {}

        for label in [0, 1, 2]:
            mask = y_test_labels == label
            if mask.sum() > 0:
                class_acc = (y_pred[mask] == y_test_labels[mask]).mean() * 100
                class_accuracies[regime_names[label]] = class_acc

        # NUEVO: Métricas de TRANSICIONES (casos difíciles)
        # Calcular régimen del histórico (últimos 7 días de X)
        historical_regimes = calculate_regime_from_window_batch(
            X_test,
            threshold_low=self.config.REGIME_THRESHOLD_LOW,
            threshold_high=self.config.REGIME_THRESHOLD_HIGH,
            lookback=7
        )

        # Detectar transiciones (cambios de régimen)
        transitions_mask = detect_regime_transitions(historical_regimes, y_test_labels)
        n_transitions = transitions_mask.sum()

        # Accuracy en transiciones (solo casos difíciles)
        transition_accuracy = None
        transition_class_accuracies = {}

        if n_transitions > 0:
            transition_accuracy = (y_pred[transitions_mask] == y_test_labels[transitions_mask]).mean() * 100

            # Accuracy por clase en transiciones
            for label in [0, 1, 2]:
                mask = (y_test_labels == label) & transitions_mask
                if mask.sum() > 0:
                    acc = (y_pred[mask] == y_test_labels[mask]).mean() * 100
                    transition_class_accuracies[regime_names[label]] = acc

        # Accuracy en continuaciones (casos fáciles)
        continuations_mask = ~transitions_mask
        n_continuations = continuations_mask.sum()
        continuation_accuracy = None

        if n_continuations > 0:
            continuation_accuracy = (y_pred[continuations_mask] == y_test_labels[continuations_mask]).mean() * 100

        if verbose:
            print("\n" + "=" * 70)
            print("EVALUACIÓN DEL CLASIFICADOR")
            print("=" * 70)
            print(f"\n✓ Overall Accuracy: {accuracy:.2f}%")

            print(f"\n📊 Accuracy por régimen:")
            for regime, acc in class_accuracies.items():
                print(f"  {regime:8s}: {acc:.2f}%")

            # NUEVO: Mostrar métricas de transiciones
            print(f"\n🔄 MÉTRICAS DE TRANSICIÓN (casos difíciles):")
            print(f"  Total muestras: {len(X_test)}")
            print(f"  Transiciones (cambio régimen): {n_transitions} ({n_transitions/len(X_test)*100:.1f}%)")
            print(f"  Continuaciones (mismo régimen): {n_continuations} ({n_continuations/len(X_test)*100:.1f}%)")

            if transition_accuracy is not None:
                print(f"\n  ⚠️  Accuracy en TRANSICIONES: {transition_accuracy:.2f}%")
                if transition_class_accuracies:
                    print(f"      Por clase en transiciones:")
                    for regime, acc in transition_class_accuracies.items():
                        print(f"        {regime:8s}: {acc:.2f}%")

            if continuation_accuracy is not None:
                print(f"\n  ✓  Accuracy en CONTINUACIONES: {continuation_accuracy:.2f}%")

            print(f"\n📊 Confusion Matrix:")
            cm = confusion_matrix(y_test_labels, y_pred)
            print(cm)

            print(f"\n📊 Classification Report:")
            print(classification_report(y_test_labels, y_pred,
                                      target_names=regime_names))

        return {
            'accuracy': accuracy,
            'class_accuracies': class_accuracies,
            'confusion_matrix': cm.tolist(),
            # Nuevas métricas
            'n_transitions': int(n_transitions),
            'n_continuations': int(n_continuations),
            'transition_accuracy': float(transition_accuracy) if transition_accuracy is not None else None,
            'continuation_accuracy': float(continuation_accuracy) if continuation_accuracy is not None else None,
            'transition_class_accuracies': transition_class_accuracies
        }

    def save(self, path: str):
        """Guarda el modelo"""
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"✓ Clasificador guardado en {path}")

    def load(self, path: str):
        """Carga el modelo"""
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        self.is_trained = True
        print(f"✓ Clasificador cargado desde {path}")


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

