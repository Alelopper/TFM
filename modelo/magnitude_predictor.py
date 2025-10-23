"""
Predictores LSTM Especializados por Régimen
============================================

3 modelos LSTM, cada uno entrenado solo con datos de su régimen:
- lstm_bajada: Especializado en predecir durante bajadas
- lstm_estable: Especializado en predecir durante estabilidad
- lstm_subida: Especializado en predecir durante subidas
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from typing import Dict, Tuple

from config import HybridConfig


class AdaptiveLoss(nn.Module):
    """
    NUEVO: Loss personalizada que penaliza más los errores cuando hay cambios grandes.
    Esto ayuda a predecir mejor las subidas y bajadas explosivas.
    """
    def __init__(self, base_weight=1.0, change_weight=2.0):
        super().__init__()
        self.base_weight = base_weight
        self.change_weight = change_weight

    def forward(self, pred, target):
        # MSE básico
        mse = (pred - target) ** 2

        # Calcular magnitud del cambio en el target
        # Mayor cambio = mayor peso
        abs_change = torch.abs(target[:, -1] - target[:, 0])
        weight = self.base_weight + self.change_weight * abs_change

        # Aplicar peso
        weighted_mse = (mse * weight.unsqueeze(1)).mean()

        return weighted_mse


class LSTMWithAttention(nn.Module):
    """
    NUEVO: LSTM con attention mechanism.
    Aprende qué días del histórico son más importantes para la predicción.
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 output_size: int, dropout: float = 0.0):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Attention layer
        self.attention = nn.Linear(hidden_size, 1)

        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # LSTM output
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden)

        # Attention weights
        attn_scores = self.attention(lstm_out)  # (batch, seq_len, 1)
        attn_weights = F.softmax(attn_scores, dim=1)  # (batch, seq_len, 1)

        # Weighted sum
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden)

        # Dropout
        context = self.dropout(context)

        # Output
        output = self.fc(context)  # (batch, output_size)

        return output


class SimpleLSTMPredictor(nn.Module):
    """LSTM simple para predicción de magnitud (MEJORADO: multicapa con dropout)"""
    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 output_size: int, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        output = self.fc(last_hidden)
        return output


class RegimePredictors:
    """
    Conjunto de 3 predictores LSTM, uno por régimen.
    """
    def __init__(self, config: HybridConfig = None):
        self.config = config or HybridConfig()
        self.device = torch.device(self.config.DEVICE if torch.cuda.is_available() else 'cpu')

        # Crear 3 modelos
        self.models = {
            'bajada': None,
            'estable': None,
            'subida': None
        }

        self.is_trained = {
            'bajada': False,
            'estable': False,
            'subida': False
        }

    def _create_model(self, input_size: int, output_size: int):
        """Crea un modelo LSTM (MEJORADO: con attention si está configurado)"""
        if self.config.LSTM_USE_ATTENTION:
            model = LSTMWithAttention(
                input_size=input_size,
                hidden_size=self.config.LSTM_HIDDEN_SIZE,
                num_layers=self.config.LSTM_NUM_LAYERS,
                output_size=output_size,
                dropout=self.config.LSTM_DROPOUT
            )
        else:
            model = SimpleLSTMPredictor(
                input_size=input_size,
                hidden_size=self.config.LSTM_HIDDEN_SIZE,
                num_layers=self.config.LSTM_NUM_LAYERS,
                output_size=output_size,
                dropout=self.config.LSTM_DROPOUT
            )
        return model.to(self.device)

    def _split_by_regime(self, X: np.ndarray, Y: np.ndarray,
                        regime_labels: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Divide datos por régimen.

        Args:
            X: [N, window_size, n_features]
            Y: [N, output_size]
            regime_labels: [N] - Etiquetas 0, 1, 2

        Returns:
            Dict con datos por régimen
        """
        regime_names = {0: 'bajada', 1: 'estable', 2: 'subida'}
        data_by_regime = {}

        for label, name in regime_names.items():
            mask = regime_labels == label
            if mask.sum() > 0:
                X_regime = X[mask]
                Y_regime = Y[mask]
                data_by_regime[name] = (X_regime, Y_regime)
            else:
                data_by_regime[name] = None

        return data_by_regime

    def train(self, X_train: np.ndarray, Y_train: np.ndarray,
             regime_labels_train: np.ndarray,
             X_val: np.ndarray = None, Y_val: np.ndarray = None,
             regime_labels_val: np.ndarray = None,
             verbose: bool = True):
        """
        Entrena los 3 predictores.

        Args:
            X_train: [N_train, window_size, n_features]
            Y_train: [N_train, output_size]
            regime_labels_train: [N_train] - Etiquetas de régimen
            X_val, Y_val, regime_labels_val: Datos de validación (opcionales)
            verbose: Imprimir progreso
        """
        if verbose:
            print("\n" + "=" * 70)
            print("ENTRENANDO PREDICTORES LSTM POR RÉGIMEN")
            print("=" * 70)

        # Dividir datos por régimen
        train_data = self._split_by_regime(X_train, Y_train, regime_labels_train)

        val_data = None
        if X_val is not None:
            val_data = self._split_by_regime(X_val, Y_val, regime_labels_val)

        # Entrenar cada modelo
        for regime_name in ['bajada', 'estable', 'subida']:
            if train_data[regime_name] is None:
                if verbose:
                    print(f"\n⚠️  No hay datos para régimen '{regime_name}', omitiendo...")
                continue

            X_regime_train, Y_regime_train = train_data[regime_name]

            X_regime_val, Y_regime_val = None, None
            if val_data and val_data[regime_name] is not None:
                X_regime_val, Y_regime_val = val_data[regime_name]

            if verbose:
                print(f"\n{'='*70}")
                print(f"Entrenando predictor para régimen: {regime_name.upper()}")
                print(f"{'='*70}")
                print(f"  Train samples: {len(X_regime_train)}")
                if X_regime_val is not None:
                    print(f"  Val samples: {len(X_regime_val)}")

            # Entrenar modelo específico
            self._train_single_model(
                regime_name,
                X_regime_train, Y_regime_train,
                X_regime_val, Y_regime_val,
                verbose=verbose
            )

    def _train_single_model(self, regime_name: str,
                           X_train: np.ndarray, Y_train: np.ndarray,
                           X_val: np.ndarray = None, Y_val: np.ndarray = None,
                           verbose: bool = True):
        """Entrena un modelo específico"""
        input_size = X_train.shape[2]
        output_size = Y_train.shape[1]

        # Crear modelo
        model = self._create_model(input_size, output_size)
        self.models[regime_name] = model

        # Preparar datos
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(Y_train)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True
        )

        val_loader = None
        if X_val is not None:
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val),
                torch.FloatTensor(Y_val)
            )
            val_loader = DataLoader(val_dataset, batch_size=self.config.BATCH_SIZE)

        # Optimizer y criterion (MEJORADO: Loss adaptativa si está configurada)
        if self.config.USE_ADAPTIVE_LOSS:
            criterion = AdaptiveLoss(base_weight=1.0, change_weight=2.0)
        else:
            criterion = nn.MSELoss()

        optimizer = optim.Adam(model.parameters(), lr=self.config.LEARNING_RATE)

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.config.NUM_EPOCHS):
            # Train
            model.train()
            train_loss = 0.0

            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            val_loss = None
            if val_loader:
                model.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for data, target in val_loader:
                        data, target = data.to(self.device), target.to(self.device)
                        output = model(data)
                        loss = criterion(output, target)
                        val_loss += loss.item()

                val_loss /= len(val_loader)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict()
                else:
                    patience_counter += 1

                if patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                    if verbose:
                        print(f"    Early stopping en epoch {epoch+1}")
                    break

            # Logging
            if verbose and (epoch % 10 == 0 or epoch < 5):
                msg = f"    Epoch {epoch+1:3d}: train_loss={train_loss:.6f}"
                if val_loss:
                    msg += f", val_loss={val_loss:.6f}"
                print(msg)

        # Restaurar mejor modelo
        if val_loader and best_model_state:
            model.load_state_dict(best_model_state)

        self.is_trained[regime_name] = True

        if verbose:
            print(f"  ✓ Predictor '{regime_name}' entrenado")

    def predict(self, X: np.ndarray, regime_labels: np.ndarray) -> np.ndarray:
        """
        Predice usando el modelo apropiado según el régimen.

        Args:
            X: [N, window_size, n_features]
            regime_labels: [N] - Etiquetas de régimen (0, 1, 2)

        Returns:
            predictions: [N, output_size]
        """
        predictions = np.zeros((len(X), self.config.OUTPUT_SIZE), dtype=np.float32)

        regime_names = {0: 'bajada', 1: 'estable', 2: 'subida'}

        for label, name in regime_names.items():
            mask = regime_labels == label

            if mask.sum() == 0 or not self.is_trained[name]:
                continue

            # Predecir con modelo específico
            model = self.models[name]
            model.eval()

            X_regime = X[mask]
            X_tensor = torch.FloatTensor(X_regime).to(self.device)

            with torch.no_grad():
                output = model(X_tensor)
                predictions[mask] = output.cpu().numpy()

        return predictions

    def evaluate(self, X_test: np.ndarray, Y_test: np.ndarray,
                regime_labels_test: np.ndarray, verbose: bool = True) -> dict:
        """Evalúa los predictores"""
        predictions = self.predict(X_test, regime_labels_test)

        # MAE global
        mae_global = np.abs(predictions - Y_test).mean()

        # MAE por régimen
        regime_names = {0: 'bajada', 1: 'estable', 2: 'subida'}
        mae_by_regime = {}

        for label, name in regime_names.items():
            mask = regime_labels_test == label
            if mask.sum() > 0:
                mae = np.abs(predictions[mask] - Y_test[mask]).mean()
                mae_by_regime[name] = mae

        if verbose:
            print("\n" + "=" * 70)
            print("EVALUACIÓN DE PREDICTORES")
            print("=" * 70)
            print(f"\n✓ MAE Global: {mae_global:.6f}")

            print(f"\n📊 MAE por régimen:")
            for name, mae in mae_by_regime.items():
                print(f"  {name:8s}: {mae:.6f}")

        return {
            'mae_global': float(mae_global),
            'mae_by_regime': mae_by_regime
        }

    def save(self, path_prefix: str):
        """Guarda los modelos"""
        for name, model in self.models.items():
            if model is not None:
                path = f"{path_prefix}_{name}.pth"
                torch.save(model.state_dict(), path)
                print(f"  ✓ Predictor '{name}' guardado en {path}")

    def load(self, path_prefix: str, input_size: int, output_size: int):
        """Carga los modelos"""
        for name in ['bajada', 'estable', 'subida']:
            path = f"{path_prefix}_{name}.pth"
            try:
                model = self._create_model(input_size, output_size)
                model.load_state_dict(torch.load(path, map_location=self.device))
                self.models[name] = model
                self.is_trained[name] = True
                print(f"  ✓ Predictor '{name}' cargado desde {path}")
            except FileNotFoundError:
                print(f"  ⚠️  No se encontró {path}")
