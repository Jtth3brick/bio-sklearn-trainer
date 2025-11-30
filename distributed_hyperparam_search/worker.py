import argparse
import logging
import pickle
import random
import signal
import time
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import redis
from sklearn.metrics import roc_auc_score

from .models import ModelConfig, SplitConfig

# SIGALRM won't work on non-unix systems
MAX_FIT_TIME = 60 * 45  # 45 minutes
MAX_UNCAUGHT_FAILURES = 3


def fitting_handler(signum, frame):
    raise TimeoutError("Model failed to train in time.")


class Worker:
    """Worker for distributed hyperparameter search"""
    
    def __init__(
        self,
        worker_id: int,
        split_dir: str = "split_data",
        redis_connection_file: str = "redis_connection.txt",
        skip_completed: bool = True,
        train_only: bool = False,
        save_dir: str = "data",
    ):
        """
        Initialize worker.
        
        Args:
            worker_id: Unique worker identifier
            split_dir: Directory containing split data (relative to cwd)
            redis_connection_file: File containing redis connection info
            skip_completed: Whether to skip already completed models
            train_only: Skip CV/eval and train on full data
            save_dir: Directory to save models and eval results
        """
        self.worker_id = worker_id
        self.split_dir = Path(split_dir)
        self.skip_completed = skip_completed
        self.train_only = train_only
        self.save_dir = Path(save_dir)
        
        # Setup logging
        logging.basicConfig(
            level=logging.DEBUG,
            format=f"%(asctime)s - worker-{worker_id} - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
        )
        logging.info("Worker active")
        
        # Connect to Redis
        with open(redis_connection_file, "r") as f:
            host, port = f.read().strip().split(":")
        
        self.redis = redis.Redis(
            host=host, 
            port=int(port), 
            decode_responses=False
        )
        logging.info("Worker successfully connected to Redis")
        
        # Cache for split configs
        self.split_cache: Dict[str, SplitConfig] = {}
    
    def get_arg(self, num_attempts: int = 3) -> Optional[ModelConfig]:
        """
        Pop a ModelConfig from Redis queue.
        
        Args:
            num_attempts: Number of retry attempts on Redis failures
            
        Returns:
            ModelConfig or None if queue is empty
        """
        def pop_item():
            item = self.redis.brpop("model_queue", timeout=1)
            if not item:
                logging.info("No items in queue")
                return None
            return pickle.loads(item[1])
        
        # Retry loop for Redis connection failures
        for attempt in range(num_attempts):
            try:
                # Keep popping until we find an untrained model or queue is empty
                while model_config := pop_item():
                    # Return immediately if we don't skip completed, 
                    # or if no result exists yet
                    if not self.skip_completed or \
                       not self.redis.exists(f"result:{model_config.config_hash}"):
                        return model_config
                    
                    logging.info(
                        "Skipping model from queue because result is already in Redis."
                    )
                
                # Queue is empty
                return None
                
            except Exception as e:
                logging.warning(f"Attempt {attempt + 1}/{num_attempts} failed: {e}")
                if attempt < num_attempts - 1:
                    sleep_time = random.uniform(0.5, 2.0)
                    logging.info(f"Sleeping {sleep_time:.2f} seconds before retry")
                    time.sleep(sleep_time)
                else:
                    logging.error(f"All {num_attempts} attempts failed")
        return None
    
    def put_result(self, result: ModelConfig):
        """Store a completed result in Redis"""
        pickled_result = pickle.dumps(result)
        
        # Store the result with hash as key
        self.redis.set(f"result:{result.config_hash}", pickled_result)
        
        # Add to set of all results
        self.redis.sadd("results", result.config_hash)
        
        # Log progress
        queue_remaining = self.redis.llen("model_queue")
        completed_count = self.redis.scard("results")
        
        logging.info(f"Stored result with hash: {result.config_hash}")
        logging.info(
            f"Progress: {completed_count} completed, "
            f"{queue_remaining} remaining in queue"
        )
    
    def ensure_split(
        self, 
        split_id: str, 
        max_size: Optional[int] = None
    ):
        """Load the split config if we don't already have it"""
        if split_id in self.split_cache:
            return
        
        split_path = self.split_dir / f"split_{split_id}.pkl"
        with open(split_path, "rb") as f:
            split_config: SplitConfig = pickle.load(f)
        self.split_cache[split_id] = split_config
        
        while max_size and len(self.split_cache) > max_size:
            # Remove first key that isn't our current split_id
            for key in self.split_cache:
                if key != split_id:
                    del self.split_cache[key]
                    break
    
    def fit_train_only(self, model_config: ModelConfig, split_config: SplitConfig):
        """
        Train model on full training data and generate predictions on test data.
        Save model and predictions to disk.
        """
        assert model_config.split_id == split_config.split_id, \
            "Data does not match. Exiting..."
        
        # Determine training data - use X_train/y_train if available, otherwise X_cv/y_cv
        if split_config.X_train is not None and split_config.y_train is not None:
            X_train_data = split_config.X_train.copy()
            y_train_data = split_config.y_train.copy()
            logging.info("Training model on X_train/y_train data")
        elif split_config.X_cv is not None and split_config.y_cv is not None:
            X_train_data = split_config.X_cv.copy()
            y_train_data = split_config.y_cv.copy()
            logging.info("Training model on X_cv/y_cv data (X_train not available)")
        else:
            raise AssertionError("No training data available (neither X_train nor X_cv)")
        
        logging.info(f"Training data shape: {X_train_data.shape}")
        pipe = model_config.get_empty_pipe()
        pipe.fit(X_train_data, y_train_data)
        
        # Store the fitted model
        model_config.model = pipe
        
        # Generate predictions on test data
        assert split_config.X_test is not None, \
            "Test features (X_test) must be available in split for train_only mode"
        
        logging.info("Generating predictions on test data")
        pred_proba = pipe.predict_proba(split_config.X_test)[:, 1]
        
        # Create save directory if it doesn't exist
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the fitted model
        model_path = self.save_dir / f"{model_config.config_hash}.model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(pipe, f)
        logging.info(f"Saved model to {model_path}")
        
        # Save predictions as dataframe
        pred_df = pd.DataFrame({
            "index": split_config.X_test.index,
            "pred_proba": pred_proba
        })
        eval_path = self.save_dir / f"{model_config.config_hash}_{split_config.split_id}.eval.pkl"
        with open(eval_path, "wb") as f:
            pickle.dump(pred_df, f)
        logging.info(f"Saved predictions to {eval_path}")
    
    def fit(self, model_config: ModelConfig, split_config: SplitConfig):
        """
        Process ModelConfig including CV scores if requested.
        Scoring metric is AUC.
        """
        assert model_config.split_id == split_config.split_id, \
            "Data does not match. Exiting..."
        
        # Get cv scores if requested
        if split_config.cv_indices:
            for i, (train_idx, val_idx) in enumerate(split_config.cv_indices):
                logging.info(f"Training cv {i} of {len(split_config.cv_indices)}")
                
                assert split_config.X_cv is not None and \
                       isinstance(split_config.X_cv, pd.DataFrame)
                assert split_config.y_cv is not None and \
                       isinstance(split_config.y_cv, pd.Series)
                
                X_train_fold = split_config.X_cv.loc[train_idx].copy()
                y_train_fold = split_config.y_cv.loc[train_idx].copy()
                X_val_fold = split_config.X_cv.loc[val_idx]
                y_val_fold = split_config.y_cv.loc[val_idx]
                
                # Fit the model
                pipe = model_config.get_empty_pipe()
                pipe.fit(X_train_fold, y_train_fold)
                
                # Add cv score to result
                pred_proba = pipe.predict_proba(X_val_fold)[:, 1]
                model_config.cv_scores.append(
                    float(roc_auc_score(y_val_fold, pred_proba))
                )
        else:
            logging.info("Skipping CV fitting")
        
        # Fit traditional train/validate if requested
        if split_config.X_train is not None:
            assert split_config.y_train is not None
            logging.info("Fitting full model")
            pipe = model_config.get_empty_pipe()
            pipe.fit(split_config.X_train.copy(), split_config.y_train.copy())
            pred_proba = pipe.predict_proba(split_config.X_val)[:, 1]
            model_config.validate_score = float(
                roc_auc_score(split_config.y_val, pred_proba)
            )
        else:
            logging.info("Skipping train/val fitting")
    
    def run(self):
        """Main worker loop"""
        trained_model_count = 0
        uncaught_failures = 0
        
        while True:
            try:
                model_config = self.get_arg(num_attempts=3)
                if model_config is None:
                    logging.info("Could not get arg. Queue is likely empty.")
                    break
                
                # Load split and only keep one in memory at a time
                self.ensure_split(
                    split_id=model_config.split_id, 
                    max_size=1
                )
                
                try:
                    logging.info(f"Fitting for {model_config} starting")
                    signal.signal(signal.SIGALRM, fitting_handler)
                    signal.alarm(MAX_FIT_TIME)
                    
                    if self.train_only:
                        # In train_only mode, use fit_train_only
                        self.fit_train_only(
                            model_config=model_config,
                            split_config=self.split_cache[model_config.split_id],
                        )
                    else:
                        # Normal mode with CV/eval
                        self.fit(
                            model_config=model_config,
                            split_config=self.split_cache[model_config.split_id],
                        )
                    
                    signal.alarm(0)
                    trained_model_count += 1
                    logging.info(f"Successfully trained {trained_model_count} models")
                except TimeoutError:
                    logging.warning(
                        f"Model fitting failed due to timeout. "
                        f"Current timeout setting: {MAX_FIT_TIME / 60:.2f} minutes. "
                        f"model_config = {model_config}"
                    )
                
                self.put_result(model_config)
                logging.debug("Successfully saved model.")
                
            except Exception as e:
                uncaught_failures += 1
                logging.exception(
                    f"Model loop failed. "
                    f"{MAX_UNCAUGHT_FAILURES - uncaught_failures} remaining: {e}"
                )
                if uncaught_failures >= MAX_UNCAUGHT_FAILURES:
                    raise
        
        logging.info("Worker is complete. Exiting...")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker_id", type=int, default=-1)
    parser.add_argument("--split_dir", type=str, default="split_data")
    parser.add_argument("--redis_connection", type=str, default="redis_connection.txt")
    parser.add_argument("--no_skip_completed", action="store_true",
                        help="Process all models even if results exist")
    parser.add_argument("--train_only", action="store_true",
                        help="Skip CV/eval, train on full data and save model")
    parser.add_argument("--save_dir", type=str, default="data",
                        help="Directory to save models and eval results")
    args = parser.parse_args()
    
    worker = Worker(
        worker_id=args.worker_id,
        split_dir=args.split_dir,
        redis_connection_file=args.redis_connection,
        skip_completed=not args.no_skip_completed,
        train_only=args.train_only,
        save_dir=args.save_dir,
    )
    worker.run()


if __name__ == "__main__":
    main()