"""
vLLM Optimization Module

This module provides both single-GPU and parallel multi-GPU optimization capabilities:

1. Single-GPU optimization (default):
   - Uses one GPU (typically GPU 0) for sequential trials
   - Compatible with all Optuna samplers (TPE, Random, Grid, BoTorch)

2. Parallel multi-GPU optimization (--parallel flag):
   - Runs trials simultaneously across multiple GPUs
   - Uses round-robin GPU assignment
   - Particularly effective with BoTorch sampler
   - Supports 2-8 GPUs depending on system configuration

Usage:
  # Single GPU
  python src/serving/main.py --n-trials 50
  
  # Parallel on GPU 0,1
  python src/serving/main.py --parallel --gpus "0,1" --n-trials 20
"""

import optuna
import sys
import time
import os
import re
import concurrent.futures
import threading
from src.serving.vllm_server import build_vllm_command, start_vllm_server, stop_vllm_server, cleanup_zombie_vllm_processes_on_ports
from src.serving.benchmarking import run_mlperf, parse_benchmarks

def generate_vllm_parameters(trial, config):
    """Generate vLLM parameters for MLPerf optimization (no concurrency parameters)"""
    candidate_flags = []
    
    parameters_config = config.get("parameters", {})
    
    for param_name, param_config in parameters_config.items():
        if not param_config.get("enabled", False):
            continue
            
        # Skip concurrency-related parameters as MLPerf doesn't use them
        if param_name in ["guidellm_concurrency", "concurrency"]:
            continue
            
        flag_name = param_config.get("name", param_name.replace("_", "-"))
        
        if "options" in param_config:
            value = trial.suggest_categorical(param_name, param_config["options"])
            candidate_flags.extend([f"--{flag_name}", str(value)])
            
        elif "level" in param_config:
            value = trial.suggest_categorical(param_name, param_config["level"])
            candidate_flags.extend([f"--{flag_name}", str(value)])
            
        elif "range" in param_config:
            range_config = param_config["range"]
            start = range_config["start"]
            end = range_config["end"]
            step = range_config.get("step", 1)
            
            if isinstance(step, float) or isinstance(start, float) or isinstance(end, float):
                value = trial.suggest_float(param_name, start, end, step=step)
            else:
                value = trial.suggest_int(param_name, start, end, step=step)
                
            candidate_flags.extend([f"--{flag_name}", str(value)])
    
    return candidate_flags

def run_single_trial(trial, model=None, dataset=None, vllm_config=None, 
                    study_dir=None, study_id=None):
    """Run a single MLPerf optimization trial"""
    port = 8000
    
    if model is None:
        model = "meta-llama/Llama-3.1-8B-Instruct"  # Default for MLPerf

    candidate_flags = generate_vllm_parameters(trial, vllm_config)

    trial_id = trial.number
    
    # Create trial directory
    trial_dir = os.path.join(study_dir, f"trial_{trial_id}")
    os.makedirs(trial_dir, exist_ok=True)
    
    vllm_log_file = os.path.join(trial_dir, "vllm_server.log")
    mlperf_log_file = os.path.join(trial_dir, "mlperf_console.log")
    
    print(f"\nStarting MLPerf trial {trial_id}")
    print(f"Model: {model}")
    print(f"Dataset: {dataset}")
    print(f"vLLM Parameters: {' '.join(candidate_flags)}")
    print(f"Trial directory: {trial_dir}")

    vllm_cmd = build_vllm_command(model_name=model, port=port, candidate_flags=candidate_flags)
    vllm_proc = start_vllm_server(vllm_cmd, log_file=vllm_log_file)

    try:
        print("Starting MLPerf benchmark...")
        summary_file = run_mlperf(model, dataset, trial_dir, mlperf_log_file)
        print("MLPerf benchmark completed successfully")

        metrics = parse_benchmarks(summary_file)
        
        # Store all metrics as trial attributes
        for metric_name, value in metrics.items():
            if value is not None:
                trial.set_user_attr(metric_name, value)

        return metrics

    finally:
        stop_vllm_server(vllm_proc)
        print("vLLM server stopped.")
        interval = vllm_config["settings"].get("trial_interval", 30)
        print(f"Waiting {interval} seconds before next trial...")
        time.sleep(interval)

def objective(trial, model=None, dataset=None, vllm_config=None, 
             study_dir=None, study_id=None):
    """Single-objective function for MLPerf optimization (maximize tokens per second)"""
    try:
        metrics = run_single_trial(trial, model, dataset, vllm_config,
                                 study_dir, study_id)
        
        tokens_per_second = float(metrics["tokens_per_second"])
        print(f"Trial {trial.number}: Tokens per second = {tokens_per_second:.2f}")
        
        return tokens_per_second

    except Exception as e:
        print(f"Error during MLPerf trial {trial.number}:", str(e), file=sys.stderr)
        import traceback
        traceback.print_exc()
        trial.set_user_attr("error", str(e))
        trial.set_user_attr("traceback", traceback.format_exc())
        raise optuna.TrialPruned(f"Trial failed: {str(e)}")

def analyze_trial_results(study, baseline_metrics=None):
    """Analyze MLPerf optimization results"""
    print("\n" + "="*80)
    print("DETAILED MLPERF OPTIMIZATION ANALYSIS")
    print("="*80)
    
    if len(study.trials) == 0:
        print("No completed trials found.")
        return
    
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    if not completed_trials:
        print("No successfully completed trials found.")
        return
    
    print(f"Completed trials: {len(completed_trials)}")
    print(f"Failed trials: {len(study.trials) - len(completed_trials)}")
    
    # Collect throughput metrics
    throughputs = []
    
    for trial in completed_trials:
        if hasattr(trial, 'user_attrs'):
            throughput = trial.user_attrs.get('tokens_per_second')
            if throughput is not None:
                throughputs.append(throughput)
    
    if throughputs:
        print(f"\nTOKENS PER SECOND STATISTICS:")
        print(f"  Min: {min(throughputs):.2f} tokens/s")
        print(f"  Max: {max(throughputs):.2f} tokens/s")
        print(f"  Mean: {sum(throughputs)/len(throughputs):.2f} tokens/s")
        print(f"  Median: {sorted(throughputs)[len(throughputs)//2]:.2f} tokens/s")
        
    if baseline_metrics and throughputs:
        baseline_throughput = baseline_metrics.get('tokens_per_second')
        if baseline_throughput:
            best_throughput = max(throughputs)
            improvement = ((best_throughput - baseline_throughput) / baseline_throughput) * 100
            
            print(f"\nIMPROVEMENT OVER BASELINE:")
            print(f"  Baseline throughput: {baseline_throughput:.2f} tokens/s")
            print(f"  Best throughput: {best_throughput:.2f} tokens/s")
            print(f"  Improvement: {improvement:.2f}%")

def get_optimization_recommendations(study):
    """Get optimization recommendations for MLPerf results"""
    print("\n" + "="*80)
    print("MLPERF OPTIMIZATION RECOMMENDATIONS")
    print("="*80)
    
    if not study.best_trial:
        print("No best trial found")
        return
    
    best_trial = study.best_trial
    best_throughput = best_trial.value
    
    print(f"Best throughput achieved: {best_throughput:.2f} tokens/s")
    print("\nOptimal vLLM parameters:")
    for param, value in best_trial.params.items():
        print(f"  --{param.replace('_', '-')}: {value}")
    
    print(f"\nTo use this configuration:")
    print(f"vllm serve meta-llama/Llama-3.1-8B \\")
    for param, value in best_trial.params.items():
        print(f"  --{param.replace('_', '-')} {value} \\")
    print(f"  --port 8000")


def run_parallel_trial(trial, model, dataset, vllm_config, study_dir, study_id, gpu_id, port):
    """Run a single trial on a specific GPU with a specific port"""
    candidate_flags = generate_vllm_parameters(trial, vllm_config)
    
    trial_id = trial.number
    
    # Create trial directory
    trial_dir = os.path.join(study_dir, f"trial_{trial_id}")
    os.makedirs(trial_dir, exist_ok=True)
    
    vllm_log_file = os.path.join(trial_dir, f"vllm_server_gpu_{gpu_id}.log")
    mlperf_log_file = os.path.join(trial_dir, "mlperf_console.log")
    
    print(f"\nStarting MLPerf trial {trial_id} on GPU {gpu_id}, port {port}")
    print(f"Model: {model}")
    print(f"Dataset: {dataset}")
    print(f"vLLM Parameters: {' '.join(candidate_flags)}")
    print(f"Trial directory: {trial_dir}")

    vllm_cmd = build_vllm_command(model_name=model, port=port, candidate_flags=candidate_flags, gpu_id=gpu_id)
    vllm_proc = start_vllm_server(vllm_cmd, log_file=vllm_log_file, gpu_id=gpu_id)

    try:
        print(f"Starting MLPerf benchmark on GPU {gpu_id}...")
        summary_file = run_mlperf(model, dataset, trial_dir, mlperf_log_file, port=port)
        print(f"MLPerf benchmark completed successfully on GPU {gpu_id}")

        metrics = parse_benchmarks(summary_file)
        
        # Store all metrics as trial attributes
        for metric_name, value in metrics.items():
            if value is not None:
                trial.set_user_attr(metric_name, value)

        return metrics

    finally:
        stop_vllm_server(vllm_proc)
        print(f"vLLM server stopped on GPU {gpu_id}.")
        # Wait a bit for GPU memory to clear
        time.sleep(5)


def parallel_objective(trial, model, dataset, vllm_config, study_dir, study_id, gpu_ids):
    """Parallel objective function that assigns trials to available GPUs"""
    # Assign GPU based on trial number (round-robin)
    gpu_id = gpu_ids[trial.number % len(gpu_ids)]
    port = 8000 + trial.number % 100  # Use different ports to avoid conflicts
    
    try:
        metrics = run_parallel_trial(trial, model, dataset, vllm_config,
                                   study_dir, study_id, gpu_id, port)
        
        tokens_per_second = float(metrics["tokens_per_second"])
        print(f"Trial {trial.number}: Tokens per second = {tokens_per_second:.2f} on GPU {gpu_id}")
        
        return tokens_per_second

    except Exception as e:
        print(f"Error during MLPerf trial {trial.number} on GPU {gpu_id}:", str(e), file=sys.stderr)
        import traceback
        traceback.print_exc()
        trial.set_user_attr("error", str(e))
        trial.set_user_attr("traceback", traceback.format_exc())
        raise optuna.TrialPruned(f"Trial failed: {str(e)}")


def run_parallel_trials(study, model, dataset, vllm_config, study_dir, study_id, gpu_ids, n_trials):
    """
    Run optimization trials in parallel across multiple GPUs
    
    Args:
        study: Optuna study object
        model: HuggingFace model name
        dataset: Dataset path
        vllm_config: vLLM configuration dict
        study_dir: Study directory path
        study_id: Study ID
        gpu_ids: List of GPU IDs to use for parallel trials
        n_trials: Total number of trials to run
        
    Returns:
        Updated study object
    """
    print(f"\n=== PARALLEL OPTIMIZATION ===")
    print(f"GPUs: {gpu_ids}")
    print(f"Total trials: {n_trials}")
    print(f"Parallel workers: {len(gpu_ids)}")
    print("=" * 50)
    
    # Clean up any existing processes on ports we might use
    ports_to_clean = [8000 + i for i in range(len(gpu_ids) * 10)]
    print("Cleaning up any existing vLLM processes...")
    cleanup_zombie_vllm_processes_on_ports(ports_to_clean)
    
    def objective_wrapper(trial):
        return parallel_objective(trial, model, dataset, vllm_config, 
                                study_dir, study_id, gpu_ids)
    
    # For better parallelism with BoTorch, run trials in batches
    batch_size = min(len(gpu_ids), 2)  # Maximum 2 trials at once for stability
    
    completed_trials = 0
    while completed_trials < n_trials:
        remaining_trials = n_trials - completed_trials
        current_batch_size = min(batch_size, remaining_trials)
        
        print(f"\nRunning batch of {current_batch_size} trials...")
        
        # Use ThreadPoolExecutor for parallel trial execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=current_batch_size) as executor:
            futures = []
            for i in range(current_batch_size):
                future = executor.submit(study.optimize, objective_wrapper, n_trials=1)
                futures.append(future)
            
            # Wait for all trials in this batch to complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                    completed_trials += 1
                    print(f"Completed {completed_trials}/{n_trials} trials")
                except Exception as e:
                    print(f"Batch trial failed: {str(e)}")
                    completed_trials += 1  # Still count as completed to avoid infinite loop
        
        # Small delay between batches to let GPUs cool down
        if completed_trials < n_trials:
            print("Waiting between batches...")
            time.sleep(3)
    
    print(f"\nParallel optimization complete!")
    if study.best_trial:
        print(f"Best trial: {study.best_trial.value:.2f} tokens/s")
        print(f"Best parameters: {study.best_trial.params}")
    
    return study 