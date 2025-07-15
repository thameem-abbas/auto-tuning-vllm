# vLLM Auto-tuning Redhat

Program finds a configuration (or multiple) that maximises output tokens/second (throughput) for a 32B MoE LLM served by vLLM.

## Features

- **P95 Latency Optimization**: Minimize 95th percentile end-to-end latency for optimal user experience
- **Multi-Objective Optimization**: Find optimal trade-offs between throughput and latency
- **Single-Objective Optimization**: Maximize throughput for high-volume applications
- **HuggingFace Model Validation**: Automatic validation of model names
- **Detailed Metrics**: Extract p50, p90, p95, p99 percentiles from guidellm benchmarks

```bash
python src/serving/main.py --mode p95_latency --model "Qwen/Qwen3-32B-FP8"
# GPU memory will be properly cleaned up on shutdown
```

#### Manual GPU Cleanup
If your vLLM server crashed or didn't shutdown properly:
```bash
# Basic cleanup
python src/serving/gpu_cleanup.py

# Verbose output to see detailed cleanup process
python src/serving/gpu_cleanup.py --verbose

# Force kill any remaining vLLM processes and cleanup
python src/serving/gpu_cleanup.py --kill-processes --force --verbose
```

#### Check GPU Status
```bash
# Check current GPU memory usage
nvidia-smi

# Check for active GPU processes
python src/serving/gpu_cleanup.py --verbose
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start: P95 Latency Optimization

To optimize for the lowest P95 end-to-end latency (recommended for user-facing applications):

```bash
# Basic P95 latency optimization
python src/serving/main.py \
  --mode p95_latency \
  --model "Qwen/Qwen3-32B-FP8" \
  --max-seconds 120 \
  --n-trials 50

# Custom workload parameters
python src/serving/main.py \
  --mode p95_latency \
  --model "microsoft/DialoGPT-medium" \
  --max-seconds 60 \
  --prompt-tokens 256 \
  --output-tokens 128 \
  --n-trials 100
```

## Universal Parameters

All optimization modes now support the same configurable parameters:

- `--model`: HuggingFace model name (default: Qwen/Qwen3-32B-FP8)
- `--max-seconds`: Duration for each trial (default: 240 seconds)
- `--prompt-tokens`: Prompt tokens for synthetic data (default: 1000)
- `--output-tokens`: Output tokens for synthetic data (default: 1000)
- `--n-trials`: Number of optimization trials (overrides config file)

## What is P95 End-to-End Latency?

**P95 latency** is the 95th percentile of request latency, meaning **95% of requests complete faster than this time**. This metric is crucial for user experience because:

- **Median (P50)**: Shows average performance, but ignores outliers
- **P95**: Shows worst-case experience for most users
- **P99**: Shows extreme outliers

P95 latency is commonly used in **Service Level Agreements (SLAs)** because it ensures consistent performance under load.

## Example Output

```
=== P95 LATENCY OPTIMIZATION MODE ===
Model: Qwen/Qwen3-32B-FP8
Trial duration: 120 seconds
Prompt tokens: 512
Output tokens: 256
Number of trials: 50
Objective: Minimize P95 end-to-end latency
==================================================

✓ Validated HuggingFace model: Qwen/Qwen3-32B-FP8

Trial 0: P95 latency = 245.67 ms (score = -245.67)
Trial 1: P95 latency = 198.43 ms (score = -198.43)
...
Trial 49: P95 latency = 156.78 ms (score = -156.78)

================================================================================
P95 LATENCY OPTIMIZATION RESULTS
================================================================================
Best P95 latency achieved: 142.35 ms
Best trial parameters:
  --max-num-batched-tokens: 2048
  --block-size: 32
  --kv-cache-dtype: fp8
  --gpu-memory-utilization: 0.92

Additional metrics from best trial:
  request_latency_p50: 89.12
  request_latency_p90: 128.45
  request_latency_p95: 142.35
  request_latency_p99: 189.67

To use this configuration:
vllm serve Qwen/Qwen3-32B-FP8 \
  --max-num-batched-tokens 2048 \
  --block-size 32 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.92 \
  --port 8000

Running post-shutdown GPU memory cleanup...
✓ Cleaned GPU 0: 32.1GB freed
✓ GPU memory cleanup completed
```

## Config-Based Optimization

You can use the traditional config-based optimization with the same parameter flexibility:

```bash
# Single-objective optimization with custom parameters
python src/serving/main.py \
  --mode config \
  --model "microsoft/DialoGPT-medium" \
  --max-seconds 180 \
  --prompt-tokens 512 \
  --output-tokens 256 \
  --n-trials 150

# Multi-objective optimization (set approach in vllm_config.yaml)
python src/serving/main.py \
  --mode config \
  --model "Qwen/Qwen3-32B-FP8" \
  --max-seconds 300 \
  --prompt-tokens 1024 \
  --output-tokens 512
```

## GPU Memory Management

### Environment Variables
The framework automatically sets optimal environment variables for GPU memory management:

```bash
# PyTorch CUDA allocator settings for better memory management
PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:512

# Disable tokenizer parallelism to avoid deadlocks
TOKENIZERS_PARALLELISM=false

# Enhanced logging for debugging
VLLM_LOGGING_LEVEL=INFO
```

### Memory Cleanup Process
The enhanced shutdown procedure includes:

1. **Graceful Shutdown**: SIGTERM → wait → SIGINT → wait → SIGKILL
2. **vLLM Parallel State Cleanup**: `destroy_model_parallel()`
3. **PyTorch CUDA Cleanup**: `torch.cuda.empty_cache()` + `torch.cuda.synchronize()`
4. **Ray Cleanup**: `ray.shutdown()` if initialized
5. **Multiple Garbage Collection Cycles**: Force Python GC multiple times
6. **Memory Verification**: Check final GPU memory usage

### Troubleshooting GPU Memory Issues

If you encounter "CUDA out of memory" errors or the GPU memory isn't released:

1. **Run the cleanup utility**:
   ```bash
   python src/serving/gpu_cleanup.py --verbose
   ```

2. **Force kill processes and cleanup**:
   ```bash
   python src/serving/gpu_cleanup.py --kill-processes --force
   ```

3. **Check environment variables**:
   ```bash
   echo $PYTORCH_CUDA_ALLOC_CONF
   echo $TOKENIZERS_PARALLELISM
   ```

4. **Monitor GPU usage**:
   ```bash
   watch -n 1 nvidia-smi
   ```

## Configuration Files

### vllm_config.yaml
Controls optimization parameters and ranges:

```yaml
parameters:
  max_num_batched_tokens:
    enabled: true
    options: [512, 1024, 2048, 4096, 8192, 16384]
  
  block_size:
    enabled: true
    options: [16, 32, 64, 128]
  
  kv_cache_dtype:
    enabled: true
    options: ["auto", "fp8", "fp8_e5m2", "fp8_e4m3"]
  
  gpu_memory_utilization:
    enabled: true
    range:
      start: 0.90
      end: 0.96
      step: 0.01

optimization:
  approach: "single_objective"  # or "multi_objective" 
  n_trials: 100
```

## Dynamic Parameter System

The framework now automatically reads all optimization parameters from the YAML config using a dynamic loader:

**Supported Parameter Types:**
- **`options`**: Categorical choices (e.g., `[16, 32, 64, 128]`)
- **`level`**: Categorical levels (e.g., `[0, 3]` for compilation)
- **`range`**: Numeric ranges with start/end/step (supports int and float)

**To add a new parameter:** Simply add it to `vllm_config.yaml` - no Python code changes needed!

```yaml
# Example: Adding a new parameter
parameters:
  your_new_parameter:
    name: "your-new-flag"
    enabled: true
    options: ["option1", "option2", "option3"]
```

## Optimization Approaches

### 1. P95 Latency Optimization (Command Line)
**Best for**: User-facing applications requiring consistent response times
```bash
python src/serving/main.py --mode p95_latency --model "Qwen/Qwen3-32B-FP8"
```
- **Objective**: Minimize 95th percentile latency
- **Result**: Single configuration with lowest P95 latency
- **Use case**: Chat applications, real-time AI assistants

### 2. Multi-Objective Optimization (Config)
**Best for**: Understanding throughput vs latency trade-offs
```yaml
# vllm_config.yaml
optimization:
  approach: "multi_objective"
```
- **Objective**: Find Pareto-optimal solutions
- **Result**: Multiple solutions showing different trade-offs
- **Use case**: When you need to choose between high throughput vs low latency

### 3. Single-Objective Optimization (Config)
**Best for**: Maximum throughput applications
```yaml
# vllm_config.yaml  
optimization:
  approach: "single_objective"
```
- **Objective**: Maximize requests per second
- **Result**: Highest throughput configuration
- **Use case**: Batch processing, high-volume inference

## Use Cases

### Gaming/Chat Applications (Low Latency Priority)
```bash
python src/serving/main.py \
  --mode p95_latency \
  --model "microsoft/DialoGPT-medium" \
  --max-seconds 30 \
  --prompt-tokens 128 \
  --output-tokens 64
```

### Document Processing (Balanced Performance)
```bash
python src/serving/main.py \
  --mode p95_latency \
  --model "Qwen/Qwen3-32B-FP8" \
  --max-seconds 120 \
  --prompt-tokens 1024 \
  --output-tokens 512
```

### High-Throughput Batch Processing
```bash
python src/serving/main.py \
  --mode config \
  --model "Qwen/Qwen3-32B-FP8" \
  --max-seconds 300 \
  --prompt-tokens 2048 \
  --output-tokens 1024 \
  --n-trials 200
# Set approach: "single_objective" in vllm_config.yaml
```

### Exploring Trade-offs
```bash
python src/serving/main.py \
  --mode config \
  --model "Qwen/Qwen3-32B-FP8" \
  --max-seconds 240 \
  --prompt-tokens 512 \
  --output-tokens 256
# Set approach: "multi_objective" in vllm_config.yaml
```

## Advanced Features

### Automatic Parameter Handling
- **max_num_seqs**: Automatically set to concurrency (50) for optimal performance
- **Parameter Validation**: All YAML parameters are validated before optimization
- **Model Validation**: HuggingFace model names are validated automatically

### Metrics Extracted

The framework extracts comprehensive metrics from guidellm:

- **Latency**: `request_latency` (median), `request_latency_p50`, `request_latency_p90`, `request_latency_p95`, `request_latency_p99`
- **Throughput**: `requests_per_second` (median and percentiles)
- **Token Metrics**: `time_to_first_token_ms`, `inter_token_latency_ms`, `output_tokens_per_second`

## Requirements

- Python 3.8+
- vLLM server
- guidellm
- HuggingFace Hub access (for model validation)
- NVIDIA GPU with CUDA support

## Study Results

Results are stored in SQLite database at `src/studies/study_N/optuna.db` and can be analyzed using Optuna's visualization tools. 