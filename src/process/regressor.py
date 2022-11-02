"""
This module houses the regression step.
"""
from typing import Any, Dict, Type, Union
import numpy as np
import pandas as pd
from src.process.preprocessor import Preprocessor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib


class Regressor:
    def __init__(self) -> None:
        """
        Initializes the regressor.
        """
        self.model = LinearRegression()

        return

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
    ) -> None:
        """
        Fits self.model on given data.
        Args:
          - X (pd.DataFrame): training features.
          - y (pd.DataFrame): training targets.
        Returns:
          - (None).
        """
        self.model.fit(X, y)

        return

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        preprocessor: Union[Type[Preprocessor], None] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates the self.model on given data. The model needs to be fitted on data beforehand.
        Args:
          - X (pd.DataFrame): evaluation features.
          - y (pd.DataFrame): evaluation targets.
          - preprocessor (Union[Type[Preprocessor], None]): processor object used for normalizing the data.
            Needed in order to undo the normalization, if no preprocessor is given, we assume
            that no normalization was done on the data beforehand.
        Returns:
          - (Dict[str, Any]): A dictionnary whose keys are metrics names and values are scores.
        """
        y_pred = self.predict(X, preprocessor)

        # undo normalization
        _, y_true = (
            preprocessor.undo_normalization(None, y) if preprocessor else (None, y)
        )

        results = {}
        results["coefficient_of_determination"] = r2_score(y_true, y_pred)
        results["mean_absolute_error"] = mean_absolute_error(y_true, y_pred)

        return results

    def predict(
        self,
        X: pd.DataFrame,
        preprocessor: Union[Type[Preprocessor], None] = None,
    ) -> np.ndarray:
        """
        Predicts [woz of a list of houses] according to given features or the area.
        self.model needs to be fitted on data beforehand.
        Args:
          - X (pd.DataFrame): area features.
          - preprocessor (Union[Type[Preprocessor], None]): processor object used for normalizing the data.
            Needed in order to undo the normalization, if no preprocessor is given, we assume
            that no normalization was done on the data beforehand.
        Returns:
          - (npt.NDArray): Predicted woz.
        """
        prediction = self.model.predict(X)

        # if no normalization was applied
        if not preprocessor:
            return prediction

        # if normalization was applied
        _, unscaled_prediction = preprocessor.undo_normalization(None, prediction)

        return unscaled_prediction

    def save_model(
        self,
        path: str,
    ) -> None:
        """
        Saves self.model to a .joblib file.
        Args:
          - path (str): path to which the file will be saved.
        Returns:
          - (None).
        """
        joblib.dump(self.model, path)

        return

    def load_model(
        self,
        path: str,
    ) -> None:
        """
        Loads model saved to a .joblib file into self.model.
        Args:
          - path (str): path from which the file will be loaded.
        Returns:
          - (None).
        """
        self.model = joblib.load(path)

        return
