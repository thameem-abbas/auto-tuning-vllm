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
import logging
from src.serving.vllm_server import build_vllm_command, start_vllm_server, stop_vllm_server, cleanup_zombie_vllm_processes_on_ports
from src.serving.benchmarking import run_mlperf, parse_benchmarks

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
                
            if flag_name != "qps": #qps goes to mlperf, not vllm
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
    
    logger.info(f"\nStarting MLPerf trial {trial_id}")
    logger.info(f"Model: {model}")
    logger.info(f"Dataset: {dataset}")
    logger.debug(f"vLLM Parameters: {' '.join(candidate_flags)}")
    logger.info(f"Trial directory: {trial_dir}")

    vllm_cmd = build_vllm_command(model_name=model, port=port, candidate_flags=candidate_flags)
    vllm_proc = start_vllm_server(vllm_cmd, log_file=vllm_log_file)

    try:
        logger.info("Starting MLPerf benchmark...")
        summary_file = run_mlperf(model, dataset, trial_dir, mlperf_log_file)
        logger.info("MLPerf benchmark completed successfully")

        metrics = parse_benchmarks(summary_file)
        
        # Store all metrics as trial attributes
        for metric_name, value in metrics.items():
            if value is not None:
                trial.set_user_attr(metric_name, value)

        return metrics

    finally:
        stop_vllm_server(vllm_proc)
        logger.info("vLLM server stopped.")
        interval = vllm_config["settings"].get("trial_interval", 30)
        logger.info(f"Waiting {interval} seconds before next trial...")
        time.sleep(interval)

def objective(trial, model=None, dataset=None, vllm_config=None, 
             study_dir=None, study_id=None):
    """Single-objective function for MLPerf optimization (maximize tokens per second)"""
    try:
        metrics = run_single_trial(trial, model, dataset, vllm_config,
                                 study_dir, study_id)
        
        tokens_per_second = float(metrics["tokens_per_second"])
        logger.info(f"Trial {trial.number}: Tokens per second = {tokens_per_second:.2f}")
        
        return tokens_per_second

    except Exception as e:
        logger.error(f"Error during MLPerf trial {trial.number}: {str(e)}")
        import traceback
        traceback.print_exc()
        trial.set_user_attr("error", str(e))
        trial.set_user_attr("traceback", traceback.format_exc())
        raise optuna.TrialPruned(f"Trial failed: {str(e)}")

def run_parallel_trial(trial, model, dataset, vllm_config, study_dir, study_id, gpu_id, port):
    """Run a single trial on a specific GPU with a specific port"""
    candidate_flags = generate_vllm_parameters(trial, vllm_config)
    
    trial_id = trial.number
    
    # Create trial directory
    trial_dir = os.path.join(study_dir, f"trial_{trial_id}")
    os.makedirs(trial_dir, exist_ok=True)
    
    vllm_log_file = os.path.join(trial_dir, f"vllm_server_gpu_{gpu_id}.log")
    mlperf_log_file = os.path.join(trial_dir, "mlperf_console.log")
    
    logger.info(f"\nStarting MLPerf trial {trial_id} on GPU {gpu_id}, port {port}")
    logger.info(f"Model: {model}")
    logger.info(f"Dataset: {dataset}")
    logger.debug(f"vLLM Parameters: {' '.join(candidate_flags)}")
    logger.info(f"Trial directory: {trial_dir}")
    if trial.params["max_num_batched_tokens"] < trial.params["max_num_seqs"]:
       raise optuna.TrialPruned("Invalid config: batched tokens < num_seqs")
    vllm_cmd = build_vllm_command(model_name=model, port=port, candidate_flags=candidate_flags, gpu_id=gpu_id)
    vllm_proc = start_vllm_server(vllm_cmd, log_file=vllm_log_file, gpu_id=gpu_id)

    try:
        logger.info(f"Starting MLPerf benchmark on GPU {gpu_id}...")
        summary_file = run_mlperf(model, dataset, trial_dir, mlperf_log_file, port=port, qps=trial.params.get("qps", 0))
        logger.info(f"MLPerf benchmark completed successfully on GPU {gpu_id}")

        metrics = parse_benchmarks(summary_file)
        
        # Store all metrics as trial attributes
        for metric_name, value in metrics.items():
            if value is not None:
                trial.set_user_attr(metric_name, value)

        return metrics

    finally:
        stop_vllm_server(vllm_proc)
        logger.info(f"vLLM server stopped on GPU {gpu_id}.")
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
        logger.info(f"Trial {trial.number}: Tokens per second = {tokens_per_second:.2f} on GPU {gpu_id}")
        
        return tokens_per_second

    except Exception as e:
        logger.error(f"Error during MLPerf trial {trial.number} on GPU {gpu_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        trial.set_user_attr("error", str(e))
        trial.set_user_attr("traceback", traceback.format_exc())
        raise optuna.TrialPruned(f"Trial failed: {str(e)}")


def run_parallel_trials(study, model, dataset, vllm_config, study_dir, study_id, gpu_ids, n_trials):
    """
    Run optimization trials in parallel across multiple GPUs with dynamic scheduling.
    When any GPU completes (success or failure), immediately start the next trial.
    
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
    logger.info(f"\nParallel optimization")
    logger.info(f"GPUs: {gpu_ids}")
    logger.info(f"Total trials: {n_trials}")
    logger.info(f"Dynamic workers: {len(gpu_ids)}")
    logger.info("Note: Next trial starts immediately when any GPU becomes available")
    logger.info("=" * 60)
    
    # Clean up any existing processes
    ports_to_clean = [8000 + i for i in range(len(gpu_ids) * 10)]
    logger.info("Cleaning up any existing vLLM processes...")
    cleanup_zombie_vllm_processes_on_ports(ports_to_clean)
    
    # GPU state tracking
    gpu_states = {gpu_id: {'status': 'idle', 'trial_id': None, 'trial_obj': None, 'port': 8000 + gpu_id, 'future': None} 
                  for gpu_id in gpu_ids}
    
    completed_trials = 0
    failed_trials = 0
    
    # Thread-safe counters
    stats_lock = threading.Lock()
    
    def run_trial_on_gpu(gpu_id, trial):
        """Run a single trial on a specific GPU"""
        port = gpu_states[gpu_id]['port']
        
        logger.info(f"\n[GPU {gpu_id}] Starting trial {trial.number}")
        
        try:
            # Use existing parallel_trial logic but with explicit GPU assignment
            candidate_flags = generate_vllm_parameters(trial, vllm_config)
            
            # Create trial directory
            trial_dir = os.path.join(study_dir, f"trial_{trial.number}")
            os.makedirs(trial_dir, exist_ok=True)
            
            vllm_log_file = os.path.join(trial_dir, f"vllm_server_gpu_{gpu_id}.log")
            mlperf_log_file = os.path.join(trial_dir, "mlperf_console.log")
            
            logger.debug(f"[GPU {gpu_id}] vLLM Parameters: {' '.join(candidate_flags)}")
            
            # Check for invalid configs early
            if trial.params["max_num_batched_tokens"] < trial.params["max_num_seqs"]:
                logger.warning(f"[GPU {gpu_id}] Invalid config - batched tokens < num_seqs")
                raise optuna.TrialPruned("Invalid config: batched tokens < num_seqs")
            
            # Start vLLM server
            vllm_cmd = build_vllm_command(model_name=model, port=port, 
                                        candidate_flags=candidate_flags, gpu_id=gpu_id)
            vllm_proc = start_vllm_server(vllm_cmd, log_file=vllm_log_file, gpu_id=gpu_id)

            try:
                # Run MLPerf benchmark
                summary_file = run_mlperf(model, dataset, trial_dir, mlperf_log_file, 
                                        port=port, qps=trial.params.get("qps", 0))
                logger.info(f"[GPU {gpu_id}] MLPerf completed successfully for trial {trial.number}")

                # Parse results
                metrics = parse_benchmarks(summary_file)
                
                # Store all metrics as trial attributes
                for metric_name, value in metrics.items():
                    if value is not None:
                        trial.set_user_attr(metric_name, value)
                
                tokens_per_second = float(metrics["tokens_per_second"])
                logger.info(f"[GPU {gpu_id}] Trial {trial.number}: {tokens_per_second:.2f} tokens/s")
                
                with stats_lock:
                    nonlocal completed_trials
                    completed_trials += 1
                    
                return tokens_per_second
                
            finally:
                stop_vllm_server(vllm_proc)
                # Brief pause to let GPU memory clear
                time.sleep(2)
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"[GPU {gpu_id}] Trial {trial.number} failed: {error_msg}")
            
            # Check if it's an OOM error
            if "CUDA out of memory" in error_msg or "OutOfMemoryError" in error_msg:
                logger.warning(f"[GPU {gpu_id}] OOM detected - GPU memory exhausted, moving to next trial")
            
            trial.set_user_attr("error", str(error_msg))
            
            with stats_lock:
                nonlocal failed_trials
                failed_trials += 1
                
            # Return None to indicate failure - the main loop will handle telling Optuna
            raise Exception(f"Trial failed: {str(e)}")
    
    # Use ThreadPoolExecutor for dynamic scheduling
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        
        while completed_trials + failed_trials < n_trials:
            # Check for completed GPU tasks and start new ones
            for gpu_id in gpu_ids:
                gpu_state = gpu_states[gpu_id]
                
                # If GPU is running a trial, check if it's done
                if gpu_state['status'] == 'running' and gpu_state['future']:
                    if gpu_state['future'].done():
                        try:
                            result = gpu_state['future'].result()
                            # Report the result back to Optuna
                            trial = gpu_state['trial_obj']
                            study.tell(trial, result)
                            logger.info(f"[GPU {gpu_id}] Trial {gpu_state['trial_id']} completed successfully, value: {result:.2f}")
                        except Exception as e:
                            logger.error(f"[GPU {gpu_id}] Trial {gpu_state['trial_id']} failed: {e}")
                            # Mark trial as failed in Optuna
                            if 'trial_obj' in gpu_state and gpu_state['trial_obj']:
                                try:
                                    study.tell(gpu_state['trial_obj'], state=optuna.trial.TrialState.FAIL)
                                except:
                                    pass  # Don't fail if we can't mark it as failed
                        
                        gpu_state['status'] = 'idle'
                        gpu_state['future'] = None
                        gpu_state['trial_id'] = None
                        gpu_state['trial_obj'] = None
                
                # If GPU is idle and we have more trials to run, start the next one
                if (gpu_state['status'] == 'idle' and 
                    completed_trials + failed_trials < n_trials):
                    
                    try:
                        trial = study.ask()
                        
                        # Submit trial to this GPU
                        future = executor.submit(run_trial_on_gpu, gpu_id, trial)
                        gpu_state['future'] = future
                        gpu_state['status'] = 'running'
                        gpu_state['trial_id'] = trial.number
                        gpu_state['trial_obj'] = trial
                        
                        logger.info(f"[GPU {gpu_id}] Assigned trial {trial.number} (Progress: {completed_trials + failed_trials}/{n_trials})")
                        
                    except Exception as e:
                        logger.error(f"Error getting next trial: {e}")
                        break
            
            # Brief sleep to avoid busy waiting
            time.sleep(0.5)
        
        # Wait for any remaining trials to complete and report results
        logger.info("Waiting for remaining trials to complete...")
        for gpu_id in gpu_ids:
            if gpu_states[gpu_id]['future']:
                try:
                    result = gpu_states[gpu_id]['future'].result()
                    # Report the result back to Optuna
                    if gpu_states[gpu_id]['trial_obj']:
                        study.tell(gpu_states[gpu_id]['trial_obj'], result)
                        logger.info(f"[GPU {gpu_id}] Final trial {gpu_states[gpu_id]['trial_id']} completed, value: {result:.2f}")
                except Exception as e:
                    logger.error(f"Final trial on GPU {gpu_id} failed: {e}")
                    # Mark trial as failed in Optuna
                    if gpu_states[gpu_id]['trial_obj']:
                        try:
                            study.tell(gpu_states[gpu_id]['trial_obj'], state=optuna.trial.TrialState.FAIL)
                        except:
                            pass
    
    logger.info(f"\nParallel optimization complete!")
    logger.info(f"Completed trials: {completed_trials}")
    logger.info(f"Failed trials: {failed_trials}")
    logger.info(f"Total trials: {completed_trials + failed_trials}")
    
    if study.best_trial:
        logger.info(f"Best trial: {study.best_trial.value:.2f} tokens/s")
        logger.info(f"Best parameters: {study.best_trial.params}")
    
    return study 