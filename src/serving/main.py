import os
import sys
import glob
import re
import optuna
import yaml
import argparse
from src.serving.utils import validate_huggingface_model
from src.serving.optimization import (
    multi_objective_function, objective, p95_latency_objective_function,
    analyze_trial_results, get_optimization_recommendations
)
from src.serving.run_baseline import run_baseline_test

SRC_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.abspath(os.path.join(SRC_DIR, ".."))
STUDIES_ROOT = os.path.join(PROJECT_DIR, "studies")
os.makedirs(STUDIES_ROOT, exist_ok=True)

# Load vLLM configuration
VLLM_CONFIG_PATH = os.path.join(PROJECT_DIR, "vllm_config.yaml")

try:
    with open(VLLM_CONFIG_PATH, 'r') as f:
        vllm_config = yaml.safe_load(f)
except FileNotFoundError:
    sys.exit(f"Config not found at {VLLM_CONFIG_PATH}.")
except yaml.YAMLError as e:
    sys.exit(f"Config syntax error in {VLLM_CONFIG_PATH}: {e}")

study_dirs = glob.glob(os.path.join(STUDIES_ROOT, "study_*"))
study_ids = []
for d in study_dirs:
    m = re.match(r".*study_(\d+)$", d)
    if m:
        study_ids.append(int(m.group(1)))

STUDY_ID = max(study_ids) + 1 if study_ids else 1
STUDY_DIR = os.path.join(STUDIES_ROOT, f"study_{STUDY_ID}")
VLLM_LOGS_DIR = os.path.join(STUDY_DIR, "vllm_logs")
GUIDELLM_LOGS_DIR = os.path.join(STUDY_DIR, "guidellm_logs")

os.makedirs(STUDY_DIR, exist_ok=True)
os.makedirs(VLLM_LOGS_DIR, exist_ok=True)
os.makedirs(GUIDELLM_LOGS_DIR, exist_ok=True)

print(f"Logging this study's data to: {STUDY_DIR}")
print(f"Logging vLLM server logs to: {VLLM_LOGS_DIR}")
print(f"Logging guidellm logs to: {GUIDELLM_LOGS_DIR}")



def main():
    parser = argparse.ArgumentParser(description='vLLM Performance Optimization')
    parser.add_argument('--mode', choices=['config', 'p95_latency'], default='config',
                        help='Optimization mode: config (use vllm_config.yaml) or p95_latency (minimize p95 latency)')
    parser.add_argument('--model', type=str, 
                        help='HuggingFace model name (default: Qwen/Qwen3-32B-FP8)')
    parser.add_argument('--max-seconds', type=int, default=240,
                        help='Duration for each trial in seconds (default: 240)')
    parser.add_argument('--prompt-tokens', type=int, default=1000,
                        help='Number of prompt tokens for synthetic data (default: 1000)')
    parser.add_argument('--output-tokens', type=int, default=1000,
                        help='Number of output tokens for synthetic data (default: 1000)')
    parser.add_argument('--n-trials', type=int, 
                        help='Number of optimization trials (overrides config)')
    parser.add_argument('--dataset', type=str,
                        help='Dataset for guidellm: HuggingFace dataset ID, local path to dataset file (CSV, JSONL, etc.), or leave empty to use synthetic data')
    
    args = parser.parse_args()
    
    model = args.model if args.model else "Qwen/Qwen3-32B-FP8"
    
    if not validate_huggingface_model(model):
        print(f"Error: Invalid HuggingFace model: {model}")
        sys.exit(1)
    
    print("=" * 80)
    print("RUNNING BASELINE TEST FIRST")
    print("=" * 80)
    baseline_metrics = run_baseline_test(
        model, args.max_seconds, args.prompt_tokens, args.output_tokens, args.dataset,
        STUDY_DIR, VLLM_LOGS_DIR, GUIDELLM_LOGS_DIR, STUDY_ID
    )
    if baseline_metrics is not None:
        print(f"✓ Baseline performance: {baseline_metrics['output_tokens_per_second']:.2f} output tokens/second")
        print(f"✓ Baseline latency: {baseline_metrics['request_latency']:.2f} ms")
        if baseline_metrics.get('request_latency_p95'):
            print(f"✓ Baseline P95 latency: {baseline_metrics['request_latency_p95']:.2f} ms")
    else:
        print("⚠ Baseline test failed")
    print("=" * 80)
    
    db_path = os.path.join(STUDY_DIR, "optuna.db")
    storage_url = f"sqlite:///{db_path}"
    
    if args.mode == 'p95_latency':
        print(f"=== P95 LATENCY OPTIMIZATION MODE ===")
        print(f"Model: {model}")
        print(f"Trial duration: {args.max_seconds} seconds")
        if args.dataset:
            print(f"Dataset: {args.dataset}")
        else:
            print(f"Prompt tokens: {args.prompt_tokens}")
            print(f"Output tokens: {args.output_tokens}")
        n_trials = args.n_trials if args.n_trials else 100
        print(f"Number of trials: {n_trials}")
        print(f"Objective: Minimize P95 end-to-end latency")
        print("=" * 50)
        
        study = optuna.create_study(
            storage=storage_url,
            study_name=f"vllm_p95_latency_run{STUDY_ID}",
            direction="minimize",
            load_if_exists=True
        )
        
        def objective_function(trial):
            return p95_latency_objective_function(
                trial, model, args.max_seconds, args.prompt_tokens, args.output_tokens, args.dataset,
                vllm_config, STUDY_DIR, VLLM_LOGS_DIR, GUIDELLM_LOGS_DIR, STUDY_ID
            )
        
        print(f"Starting P95 latency optimization with {n_trials} trials...")
        study.optimize(objective_function, n_trials=n_trials)
        
        print("\n" + "="*80)
        print("P95 LATENCY OPTIMIZATION RESULTS")
        print("="*80)
        
        best_trial = study.best_trial
        best_p95_latency = best_trial.value
        
        print(f"Best P95 latency achieved: {best_p95_latency:.2f} ms")
        if baseline_metrics and baseline_metrics.get('request_latency_p95'):
            baseline_p95 = baseline_metrics['request_latency_p95']
            improvement = ((baseline_p95 - best_p95_latency) / baseline_p95) * 100
            print(f"P95 latency improvement over baseline: {improvement:.2f}%")
        
        print(f"Best trial parameters:")
        for param, value in best_trial.params.items():
            print(f"  --{param.replace('_', '-')}: {value}")
        
        if hasattr(best_trial, 'user_attrs'):
            print(f"\nAdditional metrics from best trial:")
            for attr, value in best_trial.user_attrs.items():
                if 'latency' in attr.lower() and value is not None:
                    print(f"  {attr}: {value}")
        
        print(f"\nTo use this configuration:")
        print(f"vllm serve {model} \\")
        for param, value in best_trial.params.items():
            print(f"  --{param.replace('_', '-')} {value} \\")
        print(f"  --port 8000")
        
        return
    
    optimization_config = vllm_config.get("optimization", {})
    optimization_approach = optimization_config.get("approach", "single_objective")
    
    print(f"=== CONFIG-BASED OPTIMIZATION MODE ===")
    print(f"Approach: {optimization_approach}")
    print(f"Model: {model}")
    print(f"Trial duration: {args.max_seconds} seconds")
    if args.dataset:
        print(f"Dataset: {args.dataset}")
    else:
        print(f"Prompt tokens: {args.prompt_tokens}")
        print(f"Output tokens: {args.output_tokens}")
    n_trials = args.n_trials if args.n_trials else optimization_config.get("n_trials", 200)
    print(f"Number of trials: {n_trials}")
    print("=" * 50)
    
    if optimization_approach == "multi_objective":
        study = optuna.create_study(
            storage=storage_url,
            study_name=f"vllm_tuning_run{STUDY_ID}_multi",
            directions=["maximize", "minimize"],
            load_if_exists=True
        )
        
        def objective_function(trial):
            return multi_objective_function(
                trial, model, args.max_seconds, args.prompt_tokens, args.output_tokens, args.dataset,
                vllm_config, STUDY_DIR, VLLM_LOGS_DIR, GUIDELLM_LOGS_DIR, STUDY_ID
            )
        
        print("Using multi-objective optimization (throughput vs latency)")
        print("Result: Multiple Pareto-optimal solutions showing trade-offs")
        
    else:
        study = optuna.create_study(
            storage=storage_url,
            study_name=f"vllm_tuning_run{STUDY_ID}_single",
            direction="maximize",
            load_if_exists=True
        )
        
        def objective_function(trial):
            return objective(
                trial, model, args.max_seconds, args.prompt_tokens, args.output_tokens, args.dataset,
                vllm_config, STUDY_DIR, VLLM_LOGS_DIR, GUIDELLM_LOGS_DIR, STUDY_ID
            )
        
        print("Using single-objective optimization (maximize throughput)")
        print("Result: Highest throughput configuration (latency not considered)")

    print(f"\nStarting Optuna trials with {optimization_approach} approach...")
    print(f"Will run {n_trials} optimization trials")
    
    study.optimize(objective_function, n_trials=n_trials)

    print("\nOptimization Results:")
    if optimization_approach == "multi_objective":
        print("Multi-objective results:")
        print(f"Number of Pareto-optimal solutions: {len(study.best_trials)}")
        print("\nTop solutions (showing trade-offs):")
        for i, trial in enumerate(study.best_trials[:5]):
            throughput, latency = trial.values
            print(f"  Solution {i+1}: {throughput:.2f} tokens/s, {latency:.2f} ms latency")
            print(f"    Use case: {'High throughput' if throughput > 50 else 'Low latency'}")
            print(f"    Parameters: {trial.params}")
            print()
        
        print("INTERPRETATION:")
        print("- Each solution represents a different throughput/latency trade-off")
        print("- Choose based on your use case (high throughput vs low latency)")
        print("- All solutions are mathematically optimal (Pareto-efficient)")
        
    else:
        print(f"Best trial value: {study.best_trial.value}")
        print(f"Best trial parameters: {study.best_trial.params}")
        if baseline_metrics is not None:
            improvement = ((study.best_trial.value - baseline_metrics["output_tokens_per_second"]) / baseline_metrics["output_tokens_per_second"]) * 100
            print(f"Improvement over baseline: {improvement:.2f}%")

    analyze_trial_results(study, baseline_metrics)
    get_optimization_recommendations(study, optimization_config)

if __name__ == "__main__":
    main()