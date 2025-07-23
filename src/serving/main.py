import os
import sys
import glob
import re
import optuna
import yaml
import argparse
from optuna.samplers import TPESampler, RandomSampler, GridSampler
from optuna.integration import BoTorchSampler

# Add project root to Python path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src.serving.utils import validate_huggingface_model
from src.serving.optimization import (
    objective, analyze_trial_results, 
    get_optimization_recommendations
)
from src.serving.run_baseline import run_baseline_test
from src.serving.vllm_server import cleanup_zombie_vllm_processes

SRC_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.abspath(os.path.join(SRC_DIR, ".."))
STUDIES_ROOT = os.path.join(PROJECT_DIR, "mlperf_studies")
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

os.makedirs(STUDY_DIR, exist_ok=True)

print(f"Logging this study's data to: {STUDY_DIR}")
print(f"Each trial will have its own folder: baseline/, trial_1/, trial_2/, etc.")


def build_grid_search_space(config):
    """Build search space for GridSampler from vLLM config"""
    search_space = {}
    parameters = config.get("parameters", {})
    
    total_combinations = 1
    
    print(f"DEBUG: Building grid search space from config...")
    
    for param_name, param_config in parameters.items():
        if not param_config.get("enabled", False):
            print(f"DEBUG: Skipping disabled parameter: {param_name}")
            continue
        
        param_key = param_name.replace("-", "_")
        print(f"DEBUG: Processing parameter: {param_name} -> {param_key}")
        
        if "options" in param_config:
            # Discrete options
            values = param_config["options"]
            search_space[param_key] = values
            total_combinations *= len(values)
            print(f"DEBUG: Options parameter {param_key}: {len(values)} values {values}")
            
        elif "range" in param_config:
            # Range parameters - convert to discrete list
            range_config = param_config["range"]
            start = range_config["start"]
            end = range_config["end"]
            step = range_config["step"]
            
            # Generate all values in range
            values = []
            current = start
            while current <= end:
                values.append(current)
                current += step
                # Handle floating point precision issues
                current = round(current, 10)
            
            search_space[param_key] = values
            total_combinations *= len(values)
            print(f"DEBUG: Range parameter {param_key}: {len(values)} values from {start} to {end} step {step}")
            
        elif "level" in param_config:
            # Level parameters (like compilation_config)
            # These are treated as categorical choices, not ranges
            levels = param_config["level"]
            if isinstance(levels, list):
                search_space[param_key] = levels
                total_combinations *= len(levels)
                print(f"DEBUG: Level parameter {param_key}: {len(levels)} values {levels}")
        else:
            print(f"WARNING: Unknown parameter type for {param_name}: {param_config}")
    
    print(f"DEBUG: Final search space: {search_space}")
    print(f"DEBUG: Total combinations: {total_combinations}")
    
    return search_space, total_combinations



def main():
    parser = argparse.ArgumentParser(description='vLLM Performance Optimization with MLPerf')
    parser.add_argument('--model', type=str, 
                        help='HuggingFace model name (default: meta-llama/Llama-3.1-8B-Instruct)')
    parser.add_argument('--n-trials', type=int, 
                        help='Number of optimization trials (overrides config)')
    parser.add_argument('--dataset', type=str,
                        help='Dataset for MLPerf: local path to dataset file (default: datasets/cnn_eval.json)')
    
    args = parser.parse_args()
    
    model = args.model if args.model else "meta-llama/Llama-3.1-8B-Instruct"
    dataset = args.dataset if args.dataset else "datasets/cnn_eval.json"
    
    if not validate_huggingface_model(model):
        print(f"Error: Invalid HuggingFace model: {model}")
        sys.exit(1)
    
    # Clean up any zombie vLLM processes before starting
    cleanup_zombie_vllm_processes()
    
    print("=" * 80)
    print("RUNNING MLPERF BASELINE TEST")
    print("=" * 80)
    
    baseline_metrics = run_baseline_test(
        model, dataset, STUDY_DIR, STUDY_ID
    )
    
    if baseline_metrics is not None:
        print(f"Baseline Performance: {baseline_metrics['tokens_per_second']:.2f} tokens/second")
        if baseline_metrics.get('mean_latency_ms'):
            print(f"Baseline Mean Latency: {baseline_metrics['mean_latency_ms']:.2f} ms")
        if baseline_metrics.get('p95_latency_ms'):
            print(f"Baseline P95 Latency: {baseline_metrics['p95_latency_ms']:.2f} ms")
    else:
        print("Baseline test failed")
    print("=" * 80)
    
    db_path = os.path.join(STUDY_DIR, "optuna.db")
    storage_url = f"sqlite:///{db_path}"
    
    optimization_config = vllm_config.get("optimization", {})
    
    print(f"=== MLPERF THROUGHPUT OPTIMIZATION ===")
    print(f"Model: {model}")
    print(f"Dataset: {dataset}")
    n_trials = args.n_trials if args.n_trials else optimization_config.get("n_trials", 200)
    print(f"Number of trials: {n_trials}")
    print(f"Objective: Maximize tokens per second")
    print("=" * 50)
    
    sampler_name = optimization_config.get("sampler", "tpe")
    
    if sampler_name == "botorch":
        sampler = BoTorchSampler(
            n_startup_trials=20
        )
    elif sampler_name == "tpe":
        sampler = TPESampler()
    elif sampler_name == "random":
        sampler = RandomSampler()
    elif sampler_name == "grid":
        search_space, total_combinations = build_grid_search_space(vllm_config)
        
        print(f"\n=== GRID SAMPLER CONFIGURATION ===")
        print(f"Parameters to optimize:")
        for param, values in search_space.items():
            print(f"  {param}: {len(values)} values {values}")
        print(f"Total possible combinations: {total_combinations:,}")
        print(f"GridSampler will test all {total_combinations:,} combinations")
        
        # Set seed for reproducible results
        sampler = GridSampler(search_space=search_space, seed=42)
    else:
        sampler = TPESampler()
    
    # Single-objective optimization only
    study = optuna.create_study(
        storage=storage_url,
        study_name=f"vllm_mlperf_run{STUDY_ID}",
        direction="maximize",
        sampler=sampler,
        load_if_exists=True
    )
    
    def objective_function(trial):
        return objective(
            trial, model, dataset, vllm_config,
            STUDY_DIR, STUDY_ID
        )
    
    print(f"Using single-objective optimization with {sampler_name} sampler")
    print("Result: Maximize tokens per second using MLPerf")

    print(f"\nStarting MLPerf optimization trials...")
    
    if sampler_name == "grid":
        print(f"Will run ALL grid combinations (no n_trials limit)")
        study.optimize(objective_function)
    else:
        print(f"Will run {n_trials} optimization trials")
        study.optimize(objective_function, n_trials=n_trials)

    print("\nMLPerf Optimization Results:")
    print(f"Best throughput achieved: {study.best_trial.value:.2f} tokens/s")
    print(f"Best trial parameters: {study.best_trial.params}")
    
    if baseline_metrics is not None:
        baseline_throughput = baseline_metrics.get("tokens_per_second", 0)
        if baseline_throughput > 0:
            improvement = ((study.best_trial.value - baseline_throughput) / baseline_throughput) * 100
            print(f"Improvement over baseline: {improvement:.2f}%")

    analyze_trial_results(study, baseline_metrics)
    get_optimization_recommendations(study)

if __name__ == "__main__":
    main()