import logging
import os
import pickle
from importlib import import_module
from itertools import product, zip_longest
from pathlib import Path
from typing import Dict, List, Optional

import redis
import yaml
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.pipeline import Pipeline

from .data_interface import DataGetter
from .models import ModelConfig, SplitConfig


class HyperparamSearchManager:
    """Manager for distributed hyperparameter search"""

    def __init__(
        self,
        config_path: str,
        data_getter: DataGetter,
        split_dir: Optional[str] = None,
        redis_host: str = "localhost",
        redis_port: int = 6379,
    ):
        """
        Initialize manager with configuration.

        Args:
            config_path: Path to YAML configuration file
            data_getter: Implementation of DataGetter interface
            split_dir: Directory for split data (relative to caller's cwd)
            redis_host: Redis server host
            redis_port: Redis server port
        """
        self.config_path = Path(config_path)
        self.data_getter = data_getter
        self.split_dir = Path(split_dir or "split_data")

        # Load configuration
        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Setup Redis connection
        self.redis = redis.Redis(
            host=redis_host, port=redis_port, decode_responses=False
        )

        # Setup cache directory if enabled
        self.cache_dir = None
        if self.config.get("model_caching", {}).get("enabled"):
            self.cache_dir = Path(self.config["model_caching"]["dir"])
            self.cache_dir.mkdir(exist_ok=True)

    def get_split_config(
        self,
        split_id: str,
        train_cohorts: List[str],
        validate_cohorts: List[str],
    ) -> SplitConfig:
        """
        Prepare the input data for the models

        Args:
            split_id: Unique identifier
            train_cohorts: Cohorts for training data
            validate_cohorts: Cohorts for validation data
        """
        logging.info(f"Setting up split {split_id}")

        seed = self.config.get("seed")
        train_eval = self.config.get("train_eval", True)
        num_cv_splits = self.config.get("num_cv_splits", 5)

        if train_eval:
            X_train, y_train = self.data_getter.get_data(train_cohorts)
            logging.info(f"Got X_train.shape={X_train.shape} for id {split_id}")

            X_val, y_val = self.data_getter.get_data(
                validate_cohorts, schema=list(X_train.columns)
            )
            logging.info(f"Got X_val.shape={X_val.shape} for id {split_id}")
        else:
            logging.info("Skipping train/eval data setup")
            X_train, y_train = None, None
            X_val, y_val = None, None

        if num_cv_splits:
            skf = StratifiedKFold(
                n_splits=num_cv_splits, shuffle=True, random_state=seed
            )
            # Use both train and validate data for cv
            cv_cohorts = list(set(train_cohorts) | set(validate_cohorts))
            X_cv, y_cv = self.data_getter.get_data(cv_cohorts)
            run_indices = y_cv.index
            cv_indices = []
            for train_idx, test_idx in skf.split(X_cv, y_cv):
                # Convert indices to run ids
                train_runs = [run_indices[i] for i in train_idx]
                test_runs = [run_indices[i] for i in test_idx]
                cv_indices.append((train_runs, test_runs))
            logging.info(
                f"Added cv args of {X_cv.shape=} to split config. "
                f"Included cohorts {cv_cohorts=}."
            )
        else:
            logging.info("Skipping cv data setup")
            X_cv = None
            y_cv = None
            cv_indices = None

        return SplitConfig(
            split_id=split_id,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_cv=X_cv,
            y_cv=y_cv,
            cv_indices=cv_indices,
        )

    def get_model_configs(
        self,
        steps: Dict,
    ) -> List[Pipeline]:
        """Creates a list of pipelines from the yaml config"""

        def import_class(module_path: str):
            module_name, class_name = module_path.rsplit(".", 1)
            module = import_module(module_name)
            return getattr(module, class_name)

        # Create a param_grid sklearn style for each step permutation
        pipeline_search = []
        step_names = [list(step.keys())[0] for step in steps]
        step_configs_lists = [list(step.values())[0] for step in steps]

        for step_configs in product(*step_configs_lists):
            model_steps = []
            hyperparams = {}

            for step_name, step_config in zip(step_names, step_configs):
                func = import_class(step_config["function"])
                args = step_config.get("args", {})
                model_steps.append((step_name, func(**args)))

                # Collect hyperparameters
                for hp_name, hp_values in step_config.get("hyperparams", {}).items():
                    hyperparams[f"{step_name}__{hp_name}"] = hp_values

            cache_dir_str = str(self.cache_dir) if self.cache_dir else None
            pipeline = Pipeline(model_steps, memory=cache_dir_str)
            pipeline_search.append((pipeline, hyperparams))

        # Expand out each param grid
        all_pipelines = []
        for base_pipeline, param_grid in pipeline_search:
            if param_grid:
                # Generate all parameter combinations
                for params in ParameterGrid(param_grid):
                    pipeline_copy = clone(base_pipeline)
                    pipeline_copy.set_params(**params)
                    all_pipelines.append(pipeline_copy)
            else:
                all_pipelines.append(clone(base_pipeline))

        return all_pipelines

    def setup_splits_and_queue(self):
        """Set up split data and populate Redis queue"""
        logging.info("Setting up splits and data")
        self.split_dir.mkdir(exist_ok=True)

        splits = self.config.get("splits", {})
        pipe_configs = self.config.get("pipe_configs", {})

        for i, split_id in enumerate(splits.keys()):
            split_config = self.get_split_config(
                split_id=str(split_id),
                train_cohorts=splits[split_id]["train"],
                validate_cohorts=splits[split_id]["validate"],
            )

            # Write as pickle for workers
            savepath = self.split_dir / f"split_{split_id}.pkl"
            logging.info(f"Writing split_config to {savepath}")
            with open(savepath, "wb") as f:
                pickle.dump(split_config, f)

            # Make model_configs a list of lists for interlacing
            logging.info(f"Setting up pipeline config for {split_id}")
            nested_model_configs = []

            for pipe_name, pipe_config in pipe_configs.items():
                pipelines = self.get_model_configs(steps=pipe_config["steps"])
                nested_model_configs.append(
                    [
                        ModelConfig(
                            split_id=str(split_id),
                            pipeline_name=pipe_name,
                            _unfit_pipe=pipe,
                        )
                        for pipe in pipelines
                    ]
                )

            # Interlace pipelines for fair training
            interlaced_model_configs = [
                item
                for sublist in zip_longest(*nested_model_configs)
                for item in sublist
                if item
            ]

            logging.info(
                f"Adding {len(interlaced_model_configs)} to Redis queue "
                f"for split {split_id}..."
            )

            for model_config in interlaced_model_configs:
                self.redis.lpush("model_queue", pickle.dumps(model_config))

            logging.info(f"Done adding args for split {i + 1} out of {len(splits)}")

        logging.info("Args setup complete")

    def clear_queue(self):
        """Clear the Redis queue"""
        self.redis.delete("model_queue")
        logging.info("Cleared Redis queue")

    def get_queue_size(self) -> int:
        """Get current queue size"""
        return self.redis.llen("model_queue")

    def get_results_count(self) -> int:
        """Get count of completed results"""
        return self.redis.scard("results")

