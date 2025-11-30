# Distributed Hyperparameter Search

A distributed hyperparameter search package for machine learning pipelines using Redis for task distribution.

## Features

- Distributed task queue using Redis
- Support for multiple ML models (Lasso, Ridge, ElasticNet, SVM, XGBoost, Random Forest, Neural Networks)
- Cross-validation and train/validation splits
- Configurable pipeline with custom transformers
- Dynamic data loading through abstract interface
- Optimized for use with `uv` package manager

## Installation

Install using uv in editable mode:

```bash
uv pip install -e /path/to/distributed_hyperparam_search
```

Or with optional ML dependencies:

```bash
uv pip install -e "/path/to/distributed_hyperparam_search[ml]"
```

## Usage

### 1. Implement Data Getter

Create a class implementing the `DataGetter` interface:

```python
from distributed_hyperparam_search import DataGetter

class MyDataGetter(DataGetter):
    def get_data(self, cohorts, schema=None):
        # Your data loading logic here
        return X, y
```

### 2. Configure Search

Create a YAML configuration file with splits and pipeline configurations.

### 3. Run Manager

```python
from distributed_hyperparam_search import HyperparamSearchManager
from my_module import MyDataGetter

data_getter = MyDataGetter()
manager = HyperparamSearchManager(
    config_path="config.yaml",
    data_getter=data_getter,
    split_dir="split_data"
)
manager.setup_splits_and_queue()
```

### 4. Run Workers

```python
from distributed_hyperparam_search import Worker

worker = Worker(
    worker_id=1,
    split_dir="split_data"
)
worker.run()
```

## CLI Commands

After installation, you can use the CLI commands:

- `dhs-manager` - Run the manager
- `dhs-worker` - Run a worker

## Requirements

- Python >= 3.8
- Redis server
- See pyproject.toml for full dependency list

## License

MIT