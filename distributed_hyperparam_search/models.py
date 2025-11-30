import copy
import hashlib
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import pandas as pd
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class SplitConfig:
    """
    Contains data to train a model on.
    
    Attributes:
        split_id: Identifier for the data split
        X_train: Training features (None indicates skip and look for CV)
        y_train: Training labels
        X_val: Validation features (same columns as X_train)
        y_val: Validation labels
        X_cv: CV features (same columns as X_train)
        y_cv: CV labels
        cv_indices: Splits for cross validation (may contain indices from train and validation)
        X_test: Test features for final evaluation (same columns as X_train)
        y_test: Test labels (optional, may be None for blind test sets)
    """
    split_id: str
    X_train: Optional[pd.DataFrame]
    y_train: Optional[pd.Series]
    X_val: Optional[pd.DataFrame]
    y_val: Optional[pd.Series]
    X_cv: Optional[pd.DataFrame]
    y_cv: Optional[pd.Series]
    cv_indices: Optional[List[Tuple[List[Any], List[Any]]]]
    X_test: Optional[pd.DataFrame] = None
    y_test: Optional[pd.Series] = None


@dataclass
class ModelConfig:
    """
    Contains configuration required for training and evaluating a model.
    
    Attributes:
        split_id: Identifier for the data split
        pipeline_name: Name of the training pipeline (e.g., 'enet', 'rf')
        _unfit_pipe: The unfit pipeline
        model: Fitted model (preferably store elsewhere)
        cv_scores: Cross-validation AUC scores
        validate_score: Optional validation AUC score
        config_hash: Unique identifier based on configuration
    """
    # Set by manager
    split_id: str
    pipeline_name: str
    _unfit_pipe: Pipeline
    
    # Set by worker after fitting
    model: Optional[Pipeline] = field(default=None, repr=False)
    cv_scores: List[float] = field(default_factory=list, repr=False)
    validate_score: Optional[float] = None
    
    _config_hash: Optional[str] = None
    
    @property
    def config_hash(self) -> str:
        """Returns the config hash"""
        if self._config_hash is None:
            self._config_hash = self.create_hash()
        return self._config_hash
    
    def create_hash(self) -> str:
        """Uses pipeline hyperparameters and split name"""
        hyperparams = self._unfit_pipe.get_params()
        hyperparams_str = str(sorted(hyperparams.items()))
        unique_str = f"{hyperparams_str}|split_data_id:{self.split_id}"
        return hashlib.sha256(unique_str.encode()).hexdigest()
    
    def get_empty_pipe(self) -> Pipeline:
        """Returns a fresh copy of the pipeline with hyperparameters set"""
        return copy.deepcopy(self._unfit_pipe)
    
    def __str__(self):
        return (
            f"ModelConfig(\n"
            f"  Config Hash: {self.config_hash}\n"
            f"  Split Name: {self.split_id}\n"
            f"  Pipeline Name: {self.pipeline_name}\n"
            f"  Hyperparameters: {self._unfit_pipe.get_params()}\n"
            f"  Val Score: {self.validate_score}\n"
            f")"
        )