# Auto-tuning vLLM with MLPerf
**Build Status:** working  
**Python Version:** 3.10+

This project aims to find optimal vLLM server configurations for large language models using MLPerf benchmarking. The system maximizes throughput (tokens per second) while maintaining latency constraints, using Optuna for hyperparameter optimization with support for parallel GPU trials. MLPerf provides standardized benchmarking for LLM inference performance evaluation.

## Why is auto-tuning useful?
Auto-tuning is important because it helps us make LLMs work better and faster. By finding the best settings, we can make sure our models respond quickly, which improves the user experience. It also allows us to see which configurations/models can handle more configurations showing us the most efficient way to use LLMs.

## Requirements for Understanding

- Understanding of [vLLM](https://docs.vllm.ai/en/latest/)
- Understanding of [MLPerf Inference](https://mlcommons.org/benchmarks/inference/)
- Understanding of [Optuna](https://optuna.readthedocs.io/en/stable/)

## Installation

### Prerequisites
- Python 3.10+ (**Tested with python 3.12**)
- NVIDIA GPU with CUDA support (**System tested on dual L40S**)
- HuggingFace Hub access (**Model validation**)
- MLPerf Inference benchmark suite (**Performance evaluation**)

### Setup Instructions

1. **Clone the repository:**
```bash
git clone https://github.com/Rehan164/auto-tuning-vllm.git
cd auto-tuning-vllm
```

2. **Create and activate virtual environment:**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Dependencies
- **vllm**: LLM serving framework
- **optuna**: Hyperparameter optimization framework
- **torch**: PyTorch for ML operations
- **huggingface_hub**: Model downloading and validation
- **mlperf-inference**: MLPerf benchmarking suite for standardized performance evaluation
- **pyyaml**: Configuration file parsing
- **requests**: HTTP client for API calls

## How does the System Run
The project operates by serving a Large Language Model with vLLM and evaluating their performance using MLPerf benchmarking. This entire process is repeated for **N** number of trials which are managed through an Optuna study, with all the data getting saved into a SQLite database located at `src/studies/study_n/optuna.db`. The system supports both single-GPU and parallel multi-GPU optimization. vLLM parameters are defined in the `vllm_config.yaml` file, which allows users to easily add new parameters or change existing values without modifying the code.

![System Diagram](docs/assets/system-diagram.jpg)

## How to Run

To run the program, use the `main.py` script with specific arguments to define the model, dataset, and optimization parameters. Example commands:

### Single GPU optimization:
```bash
python -m src.serving.main --model "meta-llama/Llama-3.1-8B-Instruct" --n-trials 50
```

### Parallel GPU optimization:
```bash
python -m src.serving.main --parallel --gpus "0,1" --model "meta-llama/Llama-3.1-8B-Instruct" --n-trials 20
```

### Command Line Arguments

- `--model`: HuggingFace model name (default: `meta-llama/Llama-3.1-8B-Instruct`)
- `--dataset`: Dataset for MLPerf: local path to dataset file (default: `datasets/cnn_eval.json`)
- `--n-trials`: Number of optimization trials (overrides config file setting)
- `--parallel`: Enable parallel optimization across multiple GPUs (works with BoTorch)
- `--gpus`: Comma-separated list of GPU IDs for parallel optimization (default: "0,1")
- `--baseline-gpu`: GPU ID to use for baseline test (default: 0)

The optimization approach is configured in `vllm_config.yaml`:
- `single_objective`: Maximize throughput only (current default)

## Search Space

Currently we are exploring the following parameters, which are defined in the `vllm_config.yaml` file, to help us find the best performance:

### vLLM Server Parameters
- **max_num_batched_tokens**: Maximum tokens processed in a single batch (512, 1024, 2048, 4096, 8192, 16384)
- **block_size**: Memory block size for attention (1, 8, 16, 32, 64, 128)
- **max_num_seqs**: Maximum number of sequences per batch (64, 128, 192, 256, 384, 512, 1024, 2048, 4096, 8192, 16384)
- **gpu_memory_utilization**: GPU memory usage ratio (0.90 to 0.95 in 0.01 steps)
- **cuda_graph_sizes**: CUDA graph capture sizes (8 to 16328 in steps of 64)
- **long_prefill_token_threshold**: Threshold for long prefill tokens (0, 256, 512, 1024, 2048)
- **max_num_partial_prefills**: Maximum partial prefill operations (1, 2, 4, 8)
- **max_seq_len_to_capture**: Maximum sequence length for CUDA graphs (256, 512, 1024, 2048, 4096, 8192, 16384)

### MLPerf Benchmarking Parameters
- **qps**: Target queries per second for server scenario (5.0 to 8.0 in steps of 0.1)

## How to Add Unlisted Parameters

You can add new parameters to the `vllm_config.yaml` file by defining them with a specific structure. Each parameter block should include a name, an enabled flag, and either a `range` or `options` field to specify its possible values.

### Parameter Definition Examples

**Range-based parameters** (for continuous numerical values):
```yaml
cuda_graph_sizes:
  name: "cuda-graph-sizes"
  enabled: true
  range:
    start: 8
    end: 16384
    step: 64
```

**Options-based parameters** (for discrete predefined values):
```yaml
long_prefill_token_threshold:
  name: "long-prefill-token-threshold"
  enabled: true
  options: [0, 256, 512, 1024, 2048, 4096, 8192]
```

### Field Descriptions
- **name**: The actual command-line argument name for vLLM (e.g., `"--cuda-graph-sizes"`)
- **enabled**: Set to `true` to include the parameter in optimization, `false` to disable it
- **range**: Used for numerical parameters with continuous values
  - **start**: Minimum value for the parameter
  - **end**: Maximum value for the parameter
  - **step**: Increment between values within the range
- **options**: Used for parameters with discrete predefined values (list format)

## Parallel GPU Optimization

The system supports parallel optimization across multiple GPUs for improved efficiency and faster hyperparameter search.

### How Parallel GPU Trials Work
- **Dynamic Scheduling**: Trials are automatically distributed across available GPUs using round-robin assignment
- **Independent Execution**: Each GPU runs trials independently with separate vLLM servers and MLPerf instances
- **Port Management**: Automatic port assignment prevents conflicts between parallel trials
- **Memory Management**: Built-in cleanup between trials prevents GPU memory leaks
- **Error Handling**: OOM and other errors on one GPU don't affect trials on other GPUs

### Benefits of Parallel Trials
- **Speed**: Reduce optimization time by running trials simultaneously
- **Resource Utilization**: Make full use of multi-GPU systems
- **Robustness**: Failed trials on one GPU don't block progress on others
- **Scalability**: Easily scale from 2 to 8+ GPUs

### Usage
```bash
# Use 2 GPUs (default)
python -m src.serving.main --parallel --gpus "0,1" --n-trials 40

# Use 4 GPUs for faster optimization
python -m src.serving.main --parallel --gpus "0,1,2,3" --n-trials 80
```

## Objective Functions

The system supports different optimization approaches configured in `vllm_config.yaml`:

### 1. Single Objective (`single_objective`) - Current Default
- **Goal**: Maximize throughput (tokens per second) using MLPerf benchmarking
- **Use Case**: When you want the highest possible throughput
- **Metrics**: Returns only throughput value from MLPerf results
- **Result**: Single best configuration for maximum throughput
- **Latency Constraints**: Trials are automatically pruned if they exceed MLPerf latency thresholds


### Optimization Configuration

The objective function approach is configured in `vllm_config.yaml`:

```yaml
optimization:
  approach: "multi_objective"  # or "single_objective"
  n_trials: 200
  sampler: "botorch"  # ["botorch", "nsga2", "tpe", "random", "grid"]
```

The system uses Optuna for optimization with support for various samplers including Bayesian optimization (BoTorch), evolutionary algorithms (NSGA-II), and traditional methods (TPE, Random, Grid Search).

## Samplers

The system supports multiple optimization samplers, each with different strengths and use cases. The sampler is configured in `vllm_config.yaml` under the `optimization.sampler` field.

### Available Samplers

#### 1. **BoTorch** (`"botorch"`)
- **Algorithm**: Bayesian Optimization using Gaussian Processes
- **Recommended For**: Multi-objective optimization (throughput vs latency)

#### 2. **NSGA-II** (`"nsga2"`)
- **Algorithm**: Non-dominated Sorting Genetic Algorithm II
- **Best For**: Multi-objective optimization problems

#### 3. **TPE** (`"tpe"`) - *Default*
- **Algorithm**: Tree-structured Parzen Estimator
- **Recommended For**: single objective

#### 4. **Random** (`"random"`)
- **Algorithm**: Uniform random sampling
- **Best For**: Baseline comparisons and simple exploration
- **Characteristics**:
  - No learning from previous trials
  - Uniform coverage of search space
  - Minimal computational overhead
  - Useful for establishing baselines

#### 5. **Grid** (`"grid"`)
- **Algorithm**: Exhaustive grid search
- **Best For**: complete coverage

## Project Structure

```
auto-tuning-vllm/
├── src/
│   ├── serving/           # Core optimization logic
│   │   ├── main.py       # Entry point and CLI handling
│   │   ├── optimization.py  # Objective functions and trial execution
│   │   ├── vllm_server.py   # vLLM server management
│   │   ├── benchmarking.py  # GuideLLM integration
│   │   ├── run_baseline.py  # Baseline testing
│   │   └── utils.py      # Utility functions
│   ├── visualization/     # Results visualization
│   │   └── main_visualization.py  # Plot generation and analysis
│   ├── studies/          # Optimization results (auto-generated)
│   │   └── study_N/      # Individual study directories
│   └── vllm_config.yaml  # Parameter configuration
├── docs/                 # Documentation and assets
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Results and Output

### Study Directory Structure
Each optimization run creates a unique study directory under `src/studies/study_N/`:

```
study_N/
├── optuna.db                    # SQLite database with all trial data
├── baseline/                   # Baseline test results
│   ├── vllm_server_gpu_0.log  # vLLM server logs for baseline
│   ├── mlperf_console.log      # MLPerf console output
│   └── mlperf_log_summary.txt  # MLPerf summary results
├── trial_1/                    # Individual trial directories
│   ├── vllm_server_gpu_0.log   # vLLM server logs (single GPU)
│   ├── vllm_server_gpu_1.log   # vLLM server logs (parallel GPU)
│   ├── mlperf_console.log      # MLPerf console output
│   └── mlperf_log_summary.txt  # MLPerf summary results
├── trial_2/
│   └── ...
└── visualizations/             # Generated plots (after running visualization)
    ├── optimization_history_N.html
    ├── pareto_front_N.html
    └── parameter_importance_N.html
```

### Accessing Results
Results are automatically saved and can be accessed via:
1. **Console Output**: Real-time trial progress and summary
2. **Optuna Database**: Complete trial history and parameters
3. **MLPerf Summary Files**: Individual trial benchmark results with throughput metrics
4. **Visualizations**: Interactive plots and analysis

## Visualization

The system includes comprehensive visualization capabilities through the `src/visualization/main_visualization.py` script.

### Generating Visualizations

After completing an optimization run, generate visualizations:

```bash
python3 src/visualization/main_visualization.py
```

### Available Visualizations

#### Single Objective Optimization
- **Optimization History**: Trial progress over time with baseline comparison
- **Parameter Importance**: Which parameters most affect performance
- **Parameter Relationships**: How parameter combinations impact results

#### Multi Objective Optimization
- **throughput and latency**: 2D Graph with throughput and latency
- **Optimization History**: Trial progress over time with baseline comparison

### Visualization Features
- **Interactive Plots**: Hover for detailed trial information
- **Baseline Comparisons**: Visual comparison with baseline performance
- **Export Options**: Save plots as HTML

## Examples

### Example 1: Single GPU Optimization
```bash
# Optimize for maximum throughput with standard Llama model
python -m src.serving.main \
    --model "meta-llama/Llama-3.1-8B-Instruct" \
    --dataset "datasets/cnn_eval.json" \
    --n-trials 50
```

### Example 2: Parallel Multi-GPU Optimization
```bash
# Use 4 GPUs for faster optimization
python -m src.serving.main \
    --parallel \
    --gpus "0,1,2,3" \
    --model "meta-llama/Llama-3.1-8B-Instruct" \
    --dataset "datasets/cnn_eval.json" \
    --n-trials 100
```

### Example 3: Custom Dataset with Parallel GPUs
```bash
# Use your own dataset for realistic workload optimization
python -m src.serving.main \
    --parallel \
    --gpus "0,1" \
    --model "meta-llama/Llama-3.1-8B-Instruct" \
    --dataset "./my_dataset.json" \
    --n-trials 80
```

### Example 4: Large-Scale Optimization
```bash
# Extensive search with multiple GPUs for production deployment
python -m src.serving.main \
    --parallel \
    --gpus "0,1,2,3,4,5,6,7" \
    --model "meta-llama/Llama-3.1-8B-Instruct" \
    --dataset "datasets/cnn_eval.json" \
    --n-trials 200
```