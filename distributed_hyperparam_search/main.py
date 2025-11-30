import argparse
import logging
from pathlib import Path

from .manager import HyperparamSearchManager


def main():
    """Main entry point for distributed hyperparameter search manager"""
    parser = argparse.ArgumentParser(
        description="Distributed Hyperparameter Search Manager"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--data-getter",
        type=str,
        required=True,
        help="Import path to DataGetter implementation (e.g., 'mymodule.MyDataGetter')"
    )
    parser.add_argument(
        "--split-dir",
        type=str,
        default="split_data",
        help="Directory for split data (relative to current directory)"
    )
    parser.add_argument(
        "--redis-host",
        type=str,
        default="localhost",
        help="Redis server host"
    )
    parser.add_argument(
        "--redis-port",
        type=int,
        default=6379,
        help="Redis server port"
    )
    parser.add_argument(
        "--clear-queue",
        action="store_true",
        help="Clear Redis queue before starting"
    )
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    
    # Import data getter implementation
    module_path, class_name = args.data_getter.rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    DataGetterClass = getattr(module, class_name)
    data_getter = DataGetterClass()
    
    # Initialize manager
    manager = HyperparamSearchManager(
        config_path=args.config,
        data_getter=data_getter,
        split_dir=args.split_dir,
        redis_host=args.redis_host,
        redis_port=args.redis_port,
    )
    
    # Clear queue if requested
    if args.clear_queue:
        manager.clear_queue()
    
    # Setup splits and populate queue
    manager.setup_splits_and_queue()
    
    # Report status
    logging.info(f"Queue size: {manager.get_queue_size()}")
    logging.info(f"Results count: {manager.get_results_count()}")


if __name__ == "__main__":
    main()
