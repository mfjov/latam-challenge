import logging
from datetime import datetime
from typing import Tuple, Union, List

import numpy as np
import pandas as pd
import xgboost as xgb

logger = logging.getLogger(__name__)


class DelayModel:

    # Sacadas del feature_importance del XGBoost entrenado en el notebook (cell 58-59).
    # Con estas 10 el modelo mantiene la misma performance que con las ~30 dummies originales.
    TOP_10_FEATURES = [
        "OPERA_Latin American Wings",
        "MES_7",
        "MES_10",
        "OPERA_Grupo LATAM",
        "MES_12",
        "TIPOVUELO_I",
        "MES_4",
        "MES_11",
        "OPERA_Sky Airline",
        "OPERA_Copa Air",
    ]

    THRESHOLD_IN_MINUTES = 15

    def __init__(self):
        self._model = None

    def preprocess(
        self, data: pd.DataFrame, target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Genera features one-hot a partir de OPERA, TIPOVUELO y MES,
        filtrando a las top 10 por importancia.
        """
        features = pd.concat([
            pd.get_dummies(data["OPERA"], prefix="OPERA"),
            pd.get_dummies(data["TIPOVUELO"], prefix="TIPOVUELO"),
            pd.get_dummies(data["MES"], prefix="MES"),
        ], axis=1)

        # en serving llegan pocas filas y no todas las dummies se generan
        for col in self.TOP_10_FEATURES:
            if col not in features.columns:
                features[col] = 0

        features = features[self.TOP_10_FEATURES]

        if target_column is not None:
            data = data.copy()  # para no mutar el df del caller
            data["min_diff"] = data.apply(self._get_min_diff, axis=1)
            data[target_column] = np.where(data["min_diff"] > self.THRESHOLD_IN_MINUTES, 1, 0)
            target = pd.DataFrame(data[target_column])
            logger.info("Preprocess: %d filas, delay rate %.1f%%", len(data), target[target_column].mean() * 100)
            return features, target

        return features

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """Entrena XGBoost con scale_pos_weight para compensar desbalanceo."""
        y = target.values.ravel()
        n_y0 = int((y == 0).sum())
        n_y1 = int((y == 1).sum())
        scale = n_y0 / n_y1

        self._model = xgb.XGBClassifier(
            random_state=1, learning_rate=0.01, scale_pos_weight=scale
        )
        self._model.fit(features, y)
        logger.info("Modelo entrenado: %d muestras, ratio %.1f:1", len(y), scale)

    def predict(self, features: pd.DataFrame) -> List[int]:
        """Retorna predicciones de delay (0 o 1)."""
        if self._model is None:
            logger.warning("Predict sin modelo entrenado, retornando 0s")
            return [0] * features.shape[0]
        return [int(p) for p in self._model.predict(features)]

    @staticmethod
    def _get_min_diff(row):
        fecha_o = datetime.strptime(row["Fecha-O"], "%Y-%m-%d %H:%M:%S")
        fecha_i = datetime.strptime(row["Fecha-I"], "%Y-%m-%d %H:%M:%S")
        return ((fecha_o - fecha_i).total_seconds()) / 60
