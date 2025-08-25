# Auto-Tune vLLM

[![PyPI version](https://badge.fury.io/py/auto-tune-vllm.svg)](https://badge.fury.io/py/auto-tune-vllm)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A distributed hyperparameter optimization framework for vLLM serving, built with Ray and Optuna.

## Features

- 🚀 **Distributed Optimization**: Scale across multiple GPUs and nodes using Ray
- 🎯 **Flexible Backends**: Run locally or on Ray clusters  
- 📊 **Rich Benchmarking**: Built-in GuideLLM support + custom benchmark providers
- 🗄️ **Centralized Storage**: PostgreSQL for trials, metrics, and logs
- ⚙️ **Easy Configuration**: YAML-based study and parameter configuration
- 📈 **Multi-Objective**: Support for throughput vs latency trade-offs
- 🔧 **Extensible**: Plugin system for custom benchmarks

## Quick Start

### Installation

```bash
pip install auto-tune-vllm
```

### Basic Usage

#### CLI Interface
```bash
# Run optimization study
auto-tune-vllm optimize --config study_config.yaml

# Stream live logs  
auto-tune-vllm logs --study-id 42 --trial-number 15

# Resume interrupted study
auto-tune-vllm resume --study-id 42
```

#### Python API
```python
import asyncio
from auto_tune_vllm import StudyController, StudyConfig, RayExecutionBackend

async def main():
    config = StudyConfig.from_file("study_config.yaml") 
    backend = RayExecutionBackend({"GPU": 1, "CPU": 4})
    
    controller = StudyController(backend, study, config)
    await controller.run_optimization(n_trials=100)

asyncio.run(main())
```

## Documentation

- [Ray Cluster Setup](docs/ray_cluster_setup.md) - **Important for distributed optimization**
- [Configuration Reference](docs/configuration.md) 
- [API Documentation](docs/api.md)
- [Custom Benchmarks](docs/custom_benchmarks.md)

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support
- PostgreSQL database

All ML dependencies (vLLM, Ray, GuideLLM, BoTorch) are included automatically.

## License

Apache License 2.0 - see [LICENSE](LICENSE) file for details.