from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler


class PassThroughTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def get_feature_names_out(self, input_features=None):
        return input_features


class ThresholdApplier(BaseEstimator, TransformerMixin):
    """
    Apply a threshold to the data.
    """

    def __init__(self, threshold=0.99, threshold_type="hard"):
        """
        Initializes the transformer with the threshold and threshold type.

        Parameters:
            threshold (float): The threshold value.
            threshold_type (str): 'hard' or 'soft'.
                'hard' will set values below the threshold to 0.
                'soft' will subtract the threshold from all values.
        """
        self.threshold = threshold
        self.threshold_type = threshold_type

    def fit(self, X, y=None):
        """
        This transformer doesn't learn anything from the data
        """
        return self

    def transform(self, X):
        """
        Apply the threshold to the data.
        """
        X_transformed = X.copy()

        # Check if threshold_type is valid
        if self.threshold_type not in ["hard", "soft"]:
            raise ValueError("threshold_type must be 'hard' or 'soft'")

        # Apply the threshold
        if self.threshold_type == "hard":
            X_transformed = X_transformed.where(X_transformed >= self.threshold, 0)
        elif self.threshold_type == "soft":
            X_transformed = X_transformed - self.threshold
            X_transformed = X_transformed.clip(lower=0)

        return X_transformed

    def get_feature_names_out(self, input_features=None):
        return input_features


class LassoFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, alpha=1.0, threshold=0.01):
        self.alpha = alpha
        self.threshold = threshold
        self.lasso = Lasso(alpha=self.alpha, max_iter=1000000)
        self.support_mask: Optional[np.ndarray] = None
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        X_scaled = self.scaler.fit_transform(X)
        self.lasso.fit(X_scaled, y)
        self.support_mask = np.abs(self.lasso.coef_) > self.threshold

        # If all coefficients < threshold, keep them all
        if not np.any(self.support_mask):
            self.support_mask = np.ones(X.shape[1], dtype=bool)

        return self

    def transform(self, X):
        if self.support_mask is None:
            raise ValueError("Must call fit() before transform()")

        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.support_mask]
        return X[:, self.support_mask]

    def get_feature_names_out(self, input_features=None):
        if input_features:
            return input_features[self.support_mask]
        return None


class RandomForestFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Select features using Random Forest.
    """

    def __init__(self, n_estimators=100, threshold=0.001):
        """
        Initializes the transformer for feature importances.

        Parameters:
            n_estimators (int): The number of trees in the forest.
            threshold (float): Feature importances below this threshold will be discarded.
        """
        self.n_estimators = n_estimators
        self.threshold = threshold
        self.model = RandomForestClassifier(n_estimators=self.n_estimators)

    def fit(self, X, y=None):
        if X is None or y is None:
            raise ValueError("X and y cannot be None")

        # Fit RandomForest
        self.model.fit(X, y)
        self.feature_importances_ = self.model.feature_importances_
        self.support_mask = self.feature_importances_ >= self.threshold

        if not np.any(self.support_mask):
            self.support_mask = np.ones(X.shape[1], dtype=bool)

        return self

    def transform(self, X):
        # Check if X has the same number of features as the data used in fit
        if X.shape[1] != len(self.support_mask):
            raise ValueError(
                "X has a different number of features than the data used in fit"
            )

        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.support_mask]
        return X[:, self.support_mask]

    def get_feature_names_out(self, input_features=None):
        if input_features:
            return input_features[self.support_mask]
        return None
