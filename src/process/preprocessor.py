"""
This module houses the preprocessing step.
"""
from __future__ import annotations
from sklearn.linear_model import LinearRegression
from typing import Any, Dict, Tuple, Type, Union
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from src.utils.constants import (
    DEFAULT_INDEX_COLUMN,
    DEFAULT_TARGET_COLUMN,
    LIST_OF_OUTLIERS,
)
import joblib


class Preprocessor:
    def __init__(
        self,
        normalize_data: bool = True,
        drop_na: bool = True,
        drop_outliers: bool = True,
    ) -> None:
        """
        Initializes the Preprocessor.
        Args:
          - nomalize_data (bool): Choose whether to min-max normalize features to be in [0, 1], and apply log(x) to targets. Defaults to True.
          - drop_na (bool): Choose whether to drop null values. Defaults to true.
          - drop_outliers (bool): Choose whether to drop areas present in constants.LIST_OF_OUTLIERS
        Returns:
          - (None).
        """
        self._normalize_data = normalize_data
        self._drop_na = drop_na
        self._drop_outliers = drop_outliers

        # setup a data min and data max for normalization
        ## give them values what would nullify undo_normalization if apply_normalization wasn't used
        self._features_min = 0
        self._features_max = 1
        # Switched to log
        # self._targets_min = 0
        # self._targets_max = 1

        return

    def load_data(
        self,
        family_composition_file_path: str,
        woz_file_path: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads data from given paths and returns the features and target variables.
        Args:
          - family_composition_file_path (str): path to the family composition excel file.
          - woz_file_path (str): path to the woz excel file.
        Returns:
          - (Tuple[pd.DataFrame, pd.DataFrame]): features (X) and targets (y).
        """
        # load family composition data
        fam_com = pd.read_excel(
            family_composition_file_path,
            skiprows=2,
            index_col=DEFAULT_INDEX_COLUMN,
            na_values=["NA"],
        )
        # converting to numeric values
        fam_com = fam_com.apply(lambda x: pd.to_numeric(x, errors="coerce"))
        # replacing empty values with 0
        fam_com = fam_com.fillna(value=0)
        # removing empty areas
        fam_com = fam_com[fam_com.total != 0]

        # load woz data
        ## I only considered average woz values for 2021
        woz = pd.read_excel(
            woz_file_path,
            skiprows=2,
            nrows=431,
            index_col=DEFAULT_INDEX_COLUMN,
            usecols="A,B",
            na_values=["NA"],
        )

        ## merging woz and fam_com by index (area)
        dataset = fam_com.join(woz)

        # keeping only float values
        dataset = dataset.apply(lambda x: pd.to_numeric(x, errors="coerce"))

        # dropping null values
        if self._drop_na:
            dataset = dataset.dropna()

        # dropping outliers
        if self._drop_outliers:
            dataset = dataset.drop(LIST_OF_OUTLIERS, axis=0)

        # separating the data into features and target.
        X = dataset.drop(DEFAULT_TARGET_COLUMN, axis=1)
        y = dataset[DEFAULT_TARGET_COLUMN]

        # save the used Min and Max to undo normalization and to apply it when predicting.
        self._features_min = X.min()
        self._features_max = X.max()
        # Switched to log
        # self._targets_min = y.min()
        # self._targets_max = y.max()

        # apply normalization
        X, y = self.apply_normalization(X, y)

        return X, y

    def apply_normalization(
        self,
        X: Union[pd.DataFrame, np.ndarray, None] = None,
        y: Union[pd.DataFrame, np.ndarray, None] = None,
    ) -> Tuple[
        Union[pd.DataFrame, np.ndarray, None], Union[pd.DataFrame, np.ndarray, None]
    ]:
        """
        Min-max normalizes the features. Calculates the log of the targets.
        Args:
          - scaled_X (Union[pd.DataFrame, np.ndarray, None]): non-normalized features.
          - scaled_y (Union[pd.DataFrame, np.ndarray, None]): non-normalized targets.
        Returns:
          - (Tuple[Union[pd.DataFrame, np.ndarray, None], Union[pd.DataFrame, np.ndarray, None]]): normalized features (X) and targets (y).
        """  # if no normalization was applied
        if not self._normalize_data:
            return X, y

        # apply min max normalization
        scaled_X = (
            None
            if X is None
            else (X - self._features_min) / (self._features_max - self._features_min)
        )

        # Switched to log
        # scaled_y = None if y is None else (y - self._targets_min) / (self._targets_max - self._targets_min)
        # calculate log
        scaled_y = None if y is None else np.log(y)

        return scaled_X, scaled_y

    def undo_normalization(
        self,
        scaled_X: Union[pd.DataFrame, np.ndarray, None] = None,
        scaled_y: Union[pd.DataFrame, np.ndarray, None] = None,
    ) -> Tuple[
        Union[pd.DataFrame, np.ndarray, None], Union[pd.DataFrame, np.ndarray, None]
    ]:
        """
        Undo the min-max normalization of the features and log of the targets. Useful for woz prediction.
        Args:
          - scaled_X (Union[pd.DataFrame, np.ndarray, None]): normalized features.
          - scaled_y (Union[pd.DataFrame, np.ndarray, None]): normalized targets.
        Returns:
          - (Tuple[Union[pd.DataFrame, np.ndarray, None], Union[pd.DataFrame, np.ndarray, None]]): un-normalized features (X) and targets (y).
        """
        # if no normalization was applied
        if not self._normalize_data:
            return scaled_X, scaled_y

        # undo normalization
        X = (
            None
            if scaled_X is None
            else scaled_X * (self._features_max - self._features_min)
            + self._features_min
        )

        # Switched to log
        # y = None if scaled_y is None else scaled_y * (self._targets_max - self._targets_min) + self._targets_min
        # undo log
        y = None if scaled_y is None else np.exp(scaled_y)

        return X, y

    def train_test_split(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        test_size: float,
        shuffle: bool = True,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits the data into training and testing sets.
        Args:
          - X (pd.DataFrame): data features.
          - y (pd.DataFrame): data targets.
          - test_size (float): ratio from 0 to 1 of the data to be allocated to the test set.
          - shuffle (bool): whether to shuffle data or not. Defaults toTtrue.
          - random_state (int): Random seed to be used. Defaults to 42.
        Returns:
          - (Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]):
            respectively: training features, testing features, training targets, testing targets.

        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=shuffle
        )

        return X_train, X_test, y_train, y_test

    def get_report(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Generates a report about the data. Report includes:
          - the predictive power per feature.
          - the correlation between feature and target per feature.
          - the variance inflation factor.
          - the collinearity between features.
        Args:
          - X (pd.DataFrame): data features.
          - y (pd.DataFrame): data targets.
        Returns:
          - (Dict[str, Any]): A dictionnary whose keys are metrics names and values are scores.
        """
        # combine features and targets
        dataset = X.copy()
        dataset[DEFAULT_TARGET_COLUMN] = y
        report = {}

        # Predictive Power per Feature
        features = X.columns.values.tolist()
        pps = {}
        for feature in features:
            X_feature = X[feature].values.reshape(-1, 1)
            mvp_model = LinearRegression()
            mvp_model.fit(X_feature, y)
            y_pred_feature = mvp_model.predict(X_feature)
            pps[feature] = r2_score(y, y_pred_feature)

        report["pps"] = pps

        # Add correlation between feature and target per feature
        report["corr_coefs"] = dataset.corr().loc[:, DEFAULT_TARGET_COLUMN].to_dict()

        # Variance Inflation Factor
        report["vif"] = {
            col: variance_inflation_factor(X, index)
            for index, col in enumerate(X.columns)
        }

        # Collinearity between Features (to json so that it's json serializable)
        report["collinearity"] = X.corr().to_json()

        return report

    def save_object(
        self,
        path: str,
    ) -> None:
        """
        Saves current object to a .joblib file.
        Args:
          - path (str): path to which the file will be saved.
        Returns:
          - (None).
        """
        joblib.dump(self, path)

        return

    @staticmethod
    def load_object(
        path: str,
    ) -> Type[Preprocessor]:
        """
        Loads model object to a .joblib file into self.model.
        Args:
          - path (str): path from which the file will be loaded.
        Returns:
          - (Type[Preprocessor]: loaded object
        """

        return joblib.load(path)
