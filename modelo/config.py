"""
Configuración Centralizada del Modelo Híbrido
==============================================
"""

class HybridConfig:
    """Configuración global del modelo híbrido"""

    # ===== DATOS =====
    WINDOW_SIZE = 30  # Días de histórico para predicción
    OUTPUT_SIZE = 7   # Días a predecir
    STRIDE = 1        # Para crear secuencias

    # Features de entrada (MEJORADO: Features exógenas)
    INPUT_COLUMNS = [
        'new_cases_smoothed_per_million',
        'new_deaths_smoothed_per_million',
        'reproduction_rate',
        'positive_rate'
    ]
    OUTPUT_COLUMN = 'new_cases_smoothed_per_million'

    # NUEVO: Cluster (para XGBoost)
    USE_CLUSTER_FEATURE = True  # Activar/desactivar cluster como feature
    CLUSTER_COLUMN = 'Cluster'  # Nombre de la columna de cluster en el DataFrame

    # ===== RÉGIMEN =====
    # Umbrales para clasificar régimen (MEJORADO: Asimétricos para detectar mejor subidas)
    REGIME_THRESHOLD_LOW = -0.005      # < -0.05 = Bajada
    REGIME_THRESHOLD_HIGH = 0.007      # > 0.08 = Subida (más alto para subidas explosivas)
    # Entre -0.05 y 0.08 = Estable

    # ===== XGBOOST (CLASIFICADOR) =====
    XGB_N_ESTIMATORS = 150  # MEJORADO: Más árboles
    XGB_MAX_DEPTH = 6       # MEJORADO: Más profundo
    XGB_LEARNING_RATE = 0.05  # MEJORADO: Más conservador
    XGB_MIN_CHILD_WEIGHT = 1
    XGB_SUBSAMPLE = 0.8
    XGB_COLSAMPLE_BYTREE = 0.8
    XGB_SCALE_POS_WEIGHT = None  # Se calculará automáticamente para balancear clases

    # ===== LSTM (PREDICTOR) =====
    LSTM_HIDDEN_SIZE = 64       # MEJORADO: Más capacidad (era 32)
    LSTM_NUM_LAYERS = 2         # MEJORADO: Más profundo (era 1)
    LSTM_DROPOUT = 0.2          # MEJORADO: Regularización (era 0.0)
    LSTM_BIDIRECTIONAL = False
    LSTM_USE_ATTENTION = True   # NUEVO: Usar attention mechanism

    # ===== TRAINING =====
    BATCH_SIZE = 256
    LEARNING_RATE = 0.005       # MEJORADO: Más conservador (era 0.01)
    NUM_EPOCHS = 100            # MEJORADO: Más épocas (era 50)
    EARLY_STOPPING_PATIENCE = 15  # MEJORADO: Más paciencia (era 10)
    USE_ADAPTIVE_LOSS = True    # NUEVO: Loss personalizada que penaliza más errores en cambios grandes

    # ===== DEVICE =====
    DEVICE = 'cuda'  # o 'cpu'

    # ===== PATHS =====
    MODELS_DIR = './modeloHibrido/models'
    RESULTS_DIR = './modeloHibrido/results'

    @classmethod
    def print_config(cls):
        """Imprime la configuración actual"""
        print("=" * 70)
        print("CONFIGURACIÓN DEL MODELO HÍBRIDO")
        print("=" * 70)
        print(f"\n📊 DATOS:")
        print(f"  Window size: {cls.WINDOW_SIZE} días")
        print(f"  Output size: {cls.OUTPUT_SIZE} días")
        print(f"  Features: {cls.INPUT_COLUMNS}")

        print(f"\n🎯 RÉGIMEN:")
        print(f"  Bajada: pendiente < {cls.REGIME_THRESHOLD_LOW}")
        print(f"  Estable: {cls.REGIME_THRESHOLD_LOW} ≤ pendiente ≤ {cls.REGIME_THRESHOLD_HIGH}")
        print(f"  Subida: pendiente > {cls.REGIME_THRESHOLD_HIGH}")

        print(f"\n🌳 XGBOOST:")
        print(f"  N estimators: {cls.XGB_N_ESTIMATORS}")
        print(f"  Max depth: {cls.XGB_MAX_DEPTH}")
        print(f"  Learning rate: {cls.XGB_LEARNING_RATE}")

        print(f"\n🧠 LSTM:")
        print(f"  Hidden size: {cls.LSTM_HIDDEN_SIZE}")
        print(f"  Num layers: {cls.LSTM_NUM_LAYERS}")
        print(f"  Bidirectional: {cls.LSTM_BIDIRECTIONAL}")

        print(f"\n🎓 TRAINING:")
        print(f"  Batch size: {cls.BATCH_SIZE}")
        print(f"  Learning rate: {cls.LEARNING_RATE}")
        print(f"  Epochs: {cls.NUM_EPOCHS}")
        print("=" * 70)
