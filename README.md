# Auto-tuning vLLM 
**Build Status:** working  
**Python Version:** 3.10+

This project aims to find the best settings for running large language models using vLLM. We want to maximize the number of output tokens/sercond (throughput). At the same time, we will be minimizing the latency under a certain threshold. Specifically we would like to minimize the p95 latency under the default parameters (baseline results). This involves testing different parameters configurations for supported model on huggingface.

## Why is auto-tuning useful?
Auto-tuning is important because it helps us make LLMs work better and faster. By finding the best settings, we can make sure our models respond quickly, which improves the user experience. It also allows us to see which configurations/models can handle more configurations showing us the most efficient way to use LLMs.

## Requirements for Understanding

- Understanding of [vLLM](https://docs.vllm.ai/en/latest/)
- Understanding of [GuideLLM](https://github.com/vllm-project/guidellm)
- Understanding of [Optuna](https://optuna.readthedocs.io/en/stable/)

## Installation

### Prerequisites
- Python 3.10+ (**Tested with python 3.12**)
- NVIDIA GPU with CUDA support (**System tested on dual L40S**)
- HuggingFace Hub access (**Model validation**)

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
- **guidellm**: Benchmarking and load testing suite
- **pyyaml**: Configuration file parsing
- **requests**: HTTP client for API calls

## How does the System Run
The project operates by serving a Large Language Model with vLLM and evaluating their performance using GuideLLM. This entire process is repeated for **N** number of trials which is manage through an Optuna study, with all the data getting saved into a SQLite database located at `src/studies/study_n/optuna.db`. The parameters for vLLM are defined in the `vllm_config.yaml` file, which allows users to easily add new parameters or change existing values without neeting to modify the code.

![System Diagram](docs/assets/system-diagram.jpg)

## How to Run

To run the program, use the `main.py` script with specific arguments to define optimization mode, LLM and other benchmarking details you would like to use. Example command:

```bash
python -m src.serving.main --mode config --model "Qwen/Qwen3-30B-A3B-FP8" --max-seconds 600 --prompt-tokens 8000 --output-tokens 2000 --n-trials 300
```

### Command Line Arguments

- `--mode`: This argument specifies the optimization approach:
  - `p95_latency`: Focuses just on minimizing P95 latency
  - `config`: Uses the configuration file settings (recommended)
- `--model`: Specifies the large language model to be used for testing (default: `Qwen/Qwen3-32B-FP8`)
- `--dataset`: Dataset for GuideLLM: HuggingFace dataset ID, local path to dataset file (CSV, JSONL, etc.), or leave empty to use synthetic data
- `--max-seconds`: Duration for each trial in seconds (default: 240)
- `--prompt-tokens`: Number of prompt tokens for synthetic data (default: 1000)
- `--output-tokens`: Number of output tokens for synthetic data (default: 1000)
- `--n-trials`: Number of optimization trials (overrides config file setting)

When using `--mode config`, you can choose between:
- `single_objective`: Maximize throughput only
- `multi_objective`: Find Pareto-optimal throughput vs latency trade-offs

## Search Space

Currently we are exploring the following parameters, which are defined in the `vllm_config.yaml` file, to help us find the best performance:

### vLLM Server Parameters
- **max_num_batched_tokens**: Maximum tokens processed in a single batch (8192, 16384)
- **compilation_config**: Compilation optimization level (0, 3)
- **block_size**: Memory block size for attention (8, 16, 32, 64, 128)
- **kv_cache_dtype**: Key-value cache data type ("auto", "fp8", "fp8_e5m2", "fp8_e4m3")
- **gpu_memory_utilization**: GPU memory usage ratio (0.90 to 0.95 in 0.01 steps)
- **cuda_graph_sizes**: CUDA graph capture sizes (8 to 16384 in steps of 64)
- **long_prefill_token_threshold**: Threshold for long prefill tokens (0, 256, 512, 1024, 2048, 4096, 8192)
- **max_seq_len_to_capture**: Maximum sequence length for CUDA graphs (4096, 8192, 16384)
- **max_num_partial_prefills**: Maximum partial prefill operations (1, 2, 4, 8)

### GuideLLM Benchmarking Parameters
- **guidellm_concurrency**: Number of concurrent requests (10 to 240 in steps of 10)

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

## Objective Functions

The system supports three different optimization approaches, each with its own objective function:

### 1. Single Objective (`single_objective`)
- **Goal**: Maximize throughput (output tokens per second)
- **Use Case**: When you want the highest possible throughput regardless of latency
- **Metrics**: Returns only throughput value
- **Result**: Single best configuration for maximum throughput

### 2. Multi Objective (`multi_objective`)
- **Goal**: Find Pareto-optimal trade-offs between throughput and latency
- **Use Case**: When you need to balance performance and responsiveness
- **Metrics**: Returns both throughput (maximize) and latency (minimize)
- **Result**: Set of Pareto-optimal solutions showing different throughput/latency trade-offs

### 3. P95 Latency (`p95_latency`)
- **Goal**: Minimize P95 latency while staying under baseline threshold
- **Use Case**: When latency consistency is critical (e.g., real-time applications)
- **Metrics**: Returns P95 latency value
- **Result**: Configuration with the lowest P95 latency under acceptable limits

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
├── vllm_logs/                  # vLLM server logs for each trial
│   └── vllm_server_logs_N.{trial_id}.log
├── guidellm_logs/              # GuideLLM benchmark logs
│   └── guidellm_logs_N.{trial_id}.log
├── benchmarks_N.{trial_id}.json   # Individual trial results
├── benchmarks_N.baseline.concurrency_X.json  # Baseline results
└── visualizations/             # Generated plots (after running visualization)
    ├── optimization_history_N.html
    ├── pareto_front_N.html
    └── parameter_importance_N.html
```

### Accessing Results
Results are automatically saved and can be accessed via:
1. **Console Output**: Real-time trial progress and summary
2. **Optuna Database**: Complete trial history and parameters
3. **JSON Files**: Individual trial benchmark results
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

### Example 1: Quick Single-Objective Optimization
```bash
# Optimize for maximum throughput with a small model
python -m src.serving.main \
    --mode config \
    --model "Qwen/Qwen1.5-7B" \
    --max-seconds 120 \
    --n-trials 50
```

### Example 2: Multi-Objective Production Optimization
```bash
# Find optimal throughput/latency trade-offs for production use
python -m src.serving.main \
    --mode config \
    --model "Qwen/Qwen3-30B-A3B-FP8" \
    --max-seconds 600 \
    --prompt-tokens 8000 \
    --output-tokens 2000 \
    --n-trials 300
```

### Example 3: Custom Dataset Optimization
```bash
# Use your own dataset for realistic workload optimization
python -m src.serving.main \
    --mode config \
    --model "meta-llama/Llama-3.1-8B" \
    --dataset "./my_dataset.jsonl" \
    --max-seconds 300 \
    --n-trials 200
```

### Example 4: P95 Latency Optimization
```bash
# Minimize P95 latency for latency-critical applications
python -m src.serving.main \
    --mode p95_latency \
    --model "Qwen/Qwen3-30B-A3B-FP8" \
    --max-seconds 400 \
    --n-trials 150
```