import optuna
import sys
import time
import os
import re
from src.serving.vllm_server import build_vllm_command, start_vllm_server, stop_vllm_server
from src.serving.benchmarking import run_guidellm, parse_benchmarks

def generate_vllm_parameters(trial, config):
    candidate_flags = []
    
    parameters_config = config.get("parameters", {})
    
    for param_name, param_config in parameters_config.items():
        if not param_config.get("enabled", False):
            continue
            
        # Skip guidellm_concurrency as it's not a vLLM server parameter
        if param_name == "guidellm_concurrency":
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

def run_single_trial(trial, model=None, max_seconds=None, prompt_tokens=None, output_tokens=None, dataset=None, 
                    vllm_config=None, study_dir=None, vllm_logs_dir=None, guidellm_logs_dir=None, study_id=None):
    port = 8000
    
    if model is None:
        model = "Qwen/Qwen3-32B-FP8"
    if max_seconds is None:
        max_seconds = 240
    if prompt_tokens is None:
        prompt_tokens = 1000
    if output_tokens is None:
        output_tokens = 1000

    candidate_flags = generate_vllm_parameters(trial, vllm_config)
    
    # Get guidellm concurrency from config or use default
    parameters_config = vllm_config.get("parameters", {})
    concurrency_config = parameters_config.get("guidellm_concurrency", {})
    if concurrency_config.get("enabled", False):
        concurrency = trial.suggest_int("guidellm_concurrency", 
                                       concurrency_config["range"]["start"],
                                       concurrency_config["range"]["end"],
                                       step=concurrency_config["range"]["step"])
    else:
        concurrency = 50  # Default fallback
    
    candidate_flags.extend(["--max-num-seqs", str(concurrency)])

    trial_id = trial.number
    vllm_log_file = os.path.join(vllm_logs_dir, f"vllm_server_logs_{study_id}.{trial_id}.log")
    guidellm_log_file = os.path.join(guidellm_logs_dir, f"guidellm_logs_{study_id}.{trial_id}.log")
    
    print(f"\nStarting trial {trial_id}")
    print(f"Model: {model}")
    print(f"Duration: {max_seconds} seconds")
    print(f"Prompt tokens: {prompt_tokens}, Output tokens: {output_tokens}")
    print(f"Concurrency: {concurrency}")
    print(f"Parameters: {' '.join(candidate_flags)}")
    print(f"vLLM log file: {vllm_log_file}")
    print(f"guidellm log file: {guidellm_log_file}")

    vllm_cmd = build_vllm_command(model_name=model, port=port, candidate_flags=candidate_flags)
    vllm_proc = start_vllm_server(vllm_cmd, log_file=vllm_log_file)

    bench_file = os.path.join(
        study_dir,
        f"benchmarks_{study_id}.{trial_id}.json"
    )

    try:
        print("Starting guidellm benchmark...")
        guidellm_args = [
            "benchmark",
            "--target",      "http://localhost:8000",
            "--model",       model,
            "--processor",   model,
        ]
        
        if dataset:
            print(f"Using custom dataset: {dataset}")
            guidellm_args.extend(["--data", dataset])
        else:
            print(f"Using synthetic data: {prompt_tokens} prompt tokens, {output_tokens} output tokens")
            guidellm_args.extend(["--data", f'{{"prompt_tokens":{prompt_tokens},"output_tokens":{output_tokens}}}'])
        
        guidellm_args.extend([
            "--rate-type",   "concurrent",
            "--rate",        str(concurrency),
            # "--warmup-percent", "0.1", # Add warmup
            "--max-seconds", str(max_seconds),
            "--output-path", bench_file
        ])

        run_guidellm(guidellm_args, guidellm_log_file)
        print("guidellm benchmark completed successfully")

        metrics = parse_benchmarks(bench_file)
        
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

def multi_objective_function(trial, model=None, max_seconds=None, prompt_tokens=None, output_tokens=None, dataset=None,
                            vllm_config=None, study_dir=None, vllm_logs_dir=None, guidellm_logs_dir=None, study_id=None):
    try:
        metrics = run_single_trial(trial, model, max_seconds, prompt_tokens, output_tokens, dataset,
                                 vllm_config, study_dir, vllm_logs_dir, guidellm_logs_dir, study_id)
        
        throughput = float(metrics["output_tokens_per_second"])
        latency = float(metrics["request_latency"])
        
        return throughput, latency
        
    except Exception as e:
        print(f"Error during multi-objective trial {trial.number}:", str(e), file=sys.stderr)
        import traceback
        traceback.print_exc()
        trial.set_user_attr("error", str(e))
        trial.set_user_attr("traceback", traceback.format_exc())
        raise optuna.TrialPruned(f"Trial failed: {str(e)}")

def objective(trial, model=None, max_seconds=None, prompt_tokens=None, output_tokens=None, dataset=None,
             vllm_config=None, study_dir=None, vllm_logs_dir=None, guidellm_logs_dir=None, study_id=None):
    try:
        metrics = run_single_trial(trial, model, max_seconds, prompt_tokens, output_tokens, dataset,
                                 vllm_config, study_dir, vllm_logs_dir, guidellm_logs_dir, study_id)
        return float(metrics["output_tokens_per_second"])

    except Exception as e:
        print(f"Error during trial {trial.number}:", str(e), file=sys.stderr)
        import traceback
        traceback.print_exc()
        trial.set_user_attr("error", str(e))
        trial.set_user_attr("traceback", traceback.format_exc())
        raise optuna.TrialPruned(f"Trial failed: {str(e)}")

def p95_latency_objective_function(trial, model, max_seconds, prompt_tokens, output_tokens, dataset=None,
                                  vllm_config=None, study_dir=None, vllm_logs_dir=None, guidellm_logs_dir=None, study_id=None):
    try:
        metrics = run_single_trial(trial, model, max_seconds, prompt_tokens, output_tokens, dataset,
                                 vllm_config, study_dir, vllm_logs_dir, guidellm_logs_dir, study_id)
        
        p95_latency = metrics.get("request_latency_p95")
        
        if p95_latency is None:
            p95_latency = metrics["request_latency"]
            print(f"Warning: p95 latency not available, using median: {p95_latency} ms")
        
        for metric_name, value in metrics.items():
            if value is not None:
                trial.set_user_attr(metric_name, value)
        
        latency_value = float(p95_latency)
        
        print(f"Trial {trial.number}: P95 latency = {latency_value:.2f} ms")
        
        return latency_value
        
    except Exception as e:
        print(f"Error during p95 latency trial {trial.number}:", str(e), file=sys.stderr)
        import traceback
        traceback.print_exc()
        trial.set_user_attr("error", str(e))
        trial.set_user_attr("traceback", traceback.format_exc())
        raise optuna.TrialPruned(f"Trial failed: {str(e)}")

def analyze_trial_results(study, baseline_metrics=None):
    print("\n" + "="*80)
    print("DETAILED OPTIMIZATION ANALYSIS")
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
    
    throughputs = []
    latencies = []
    for trial in completed_trials:
        if hasattr(trial, 'user_attrs'):
            throughput = trial.user_attrs.get('output_tokens_per_second')
            latency = trial.user_attrs.get('request_latency')
            if throughput is not None and latency is not None:
                throughputs.append(throughput)
                latencies.append(latency)
    
    if throughputs and latencies:
        print(f"\nTHROUGHPUT STATISTICS:")
        print(f"  Min: {min(throughputs):.2f} tokens/s")
        print(f"  Max: {max(throughputs):.2f} tokens/s")
        print(f"  Mean: {sum(throughputs)/len(throughputs):.2f} tokens/s")
        print(f"  Median: {sorted(throughputs)[len(throughputs)//2]:.2f} tokens/s")
        
        print(f"\nLATENCY STATISTICS:")
        print(f"  Min: {min(latencies):.2f} ms")
        print(f"  Max: {max(latencies):.2f} ms")
        print(f"  Mean: {sum(latencies)/len(latencies):.2f} ms")
        print(f"  Median: {sorted(latencies)[len(latencies)//2]:.2f} ms")
        
        if baseline_metrics:
            baseline_throughput = baseline_metrics['output_tokens_per_second']
            baseline_latency = baseline_metrics['request_latency']
            
            best_throughput = max(throughputs)
            best_latency = min(latencies)
            
            throughput_improvement = ((best_throughput - baseline_throughput) / baseline_throughput) * 100
            latency_improvement = ((baseline_latency - best_latency) / baseline_latency) * 100
            
            print(f"\nIMPROVEMENT OVER BASELINE:")
            print(f"  Best throughput improvement: {throughput_improvement:.2f}%")
            print(f"  Best latency improvement: {latency_improvement:.2f}%")

def get_optimization_recommendations(study, optimization_config):
    print("\n" + "="*80)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("="*80)
    
    approach = optimization_config.get("approach", "single_objective")
    
    if approach == "multi_objective":
        pareto_solutions = study.best_trials
        if len(pareto_solutions) > 0:
            print(f"Found {len(pareto_solutions)} Pareto-optimal solutions")
            print("\nTop 3 configurations:")
            
            for i, trial in enumerate(pareto_solutions[:3]):
                throughput, latency = trial.values
                print(f"\n  Option {i+1}: {throughput:.2f} tokens/s, {latency:.2f} ms")
                print(f"    Use case: {'High throughput' if throughput > latency/10 else 'Low latency'}")
                print("    Parameters:")
                for param, value in trial.params.items():
                    print(f"      --{param.replace('_', '-')}: {value}")
        else:
            print("No Pareto-optimal solutions found")
    
    else:
        best_trial = study.best_trial
        print(f"Best throughput achieved: {best_trial.value:.2f} tokens/s")
        print("Parameters:")
        for param, value in best_trial.params.items():
            print(f"  --{param.replace('_', '-')}: {value}")

def generate_command_line(trial_params):
    args = []
    for param, value in trial_params.items():
        arg_name = "--" + param.replace("_", "-")
        args.append(f"{arg_name} {value}")
    
    return " ".join(args) 