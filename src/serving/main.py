import os
import sys
import glob
import re
import optuna
import yaml
import argparse
import logging
from optuna.samplers import TPESampler, RandomSampler, GridSampler
from optuna.integration import BoTorchSampler

# Add project root to Python path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src.serving.utils import validate_huggingface_model
from src.serving.optimization import (
    objective, 
)
from src.serving.run_baseline import run_baseline_test
from src.serving.vllm_server import cleanup_zombie_vllm_processes

SRC_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.abspath(os.path.join(SRC_DIR, ".."))
STUDIES_ROOT = os.path.join(PROJECT_DIR, "mlperf_studies")
os.makedirs(STUDIES_ROOT, exist_ok=True)

# Configure logging for the module
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

logger.info(f"Logging this study's data to: {STUDY_DIR}")
logger.info(f"Each trial will have its own folder: baseline/, trial_1/, trial_2/, etc.")


def build_grid_search_space(config):
    """Build search space for GridSampler from vLLM config"""
    logger = logging.getLogger(__name__)
    search_space = {}
    parameters = config.get("parameters", {})
    
    total_combinations = 1
    
    logger.debug(f"Building grid search space from config...")
    
    for param_name, param_config in parameters.items():
        if not param_config.get("enabled", False):
            logger.debug(f"Skipping disabled parameter: {param_name}")
            continue
        
        param_key = param_name.replace("-", "_")
        logger.debug(f"Processing parameter: {param_name} -> {param_key}")
        
        if "options" in param_config:
            # Discrete options
            values = param_config["options"]
            search_space[param_key] = values
            total_combinations *= len(values)
            logger.debug(f"Options parameter {param_key}: {len(values)} values {values}")
            
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
            logger.debug(f"Range parameter {param_key}: {len(values)} values from {start} to {end} step {step}")
            
        elif "level" in param_config:
            # Level parameters (like compilation_config)
            # These are treated as categorical choices, not ranges
            levels = param_config["level"]
            if isinstance(levels, list):
                search_space[param_key] = levels
                total_combinations *= len(levels)
                logger.debug(f"Level parameter {param_key}: {len(levels)} values {levels}")
        else:
            logger.warning(f"Unknown parameter type for {param_name}: {param_config}")
    
    logger.debug(f"Final search space: {search_space}")
    logger.debug(f"Total combinations: {total_combinations}")
    
    return search_space, total_combinations



def main():
    # Get logger instance
    logger = logging.getLogger(__name__)
    
    parser = argparse.ArgumentParser(
        description='vLLM Performance Optimization with MLPerf',
        epilog="""
            Examples:
            # Regular single-GPU optimization
            python src/serving/main.py --n-trials 50
            
            # Parallel optimization on GPU 0 and 1
            python src/serving/main.py --parallel --gpus "0,1" --n-trials 20
            
            # Parallel optimization on 4 GPUs
            python src/serving/main.py --parallel --gpus "0,1,2,3" --n-trials 40
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--model', type=str, 
                        help='HuggingFace model name (default: meta-llama/Llama-3.1-8B-Instruct)')
    parser.add_argument('--n-trials', type=int, 
                        help='Number of optimization trials (overrides config)')
    parser.add_argument('--dataset', type=str,
                        help='Dataset for MLPerf: local path to dataset file (default: datasets/cnn_eval.json)')
    parser.add_argument('--parallel', action='store_true',
                        help='Enable parallel optimization across multiple GPUs (works with BoTorch)')
    parser.add_argument('--gpus', type=str, default="0,1",
                        help='Comma-separated list of GPU IDs for parallel optimization (default: "0,1")')
    parser.add_argument('--baseline-gpu', type=int, default=0,
                        help='GPU ID to use for baseline test (default: 0)')
    
    args = parser.parse_args()
    
    model = args.model if args.model else "meta-llama/Llama-3.1-8B-Instruct"
    dataset = args.dataset if args.dataset else "datasets/cnn_eval.json"
    
    #Commenting this out since it does not seem to work when I point it to a local path
    #if not validate_huggingface_model(model):
    if False:
        logger.error(f"Invalid HuggingFace model: {model}")
        sys.exit(1)
    
    # Clean up any zombie vLLM processes before starting
    cleanup_zombie_vllm_processes()
    
    logger.info("Running the MLPERF Baseline Test")
    logger.info(f"Selected GPU for baseline: {args.baseline_gpu}")
    
    baseline_metrics = run_baseline_test(
        model, dataset, STUDY_DIR, STUDY_ID, gpu_id=args.baseline_gpu
    )
    
    if baseline_metrics is not None:
        logger.info(f"Baseline Performance: {baseline_metrics['tokens_per_second']:.2f} tokens/second")
        if baseline_metrics.get('mean_latency_ms'):
            logger.info(f"Baseline Mean Latency: {baseline_metrics['mean_latency_ms']:.2f} ms")
    else:
        logger.error("Baseline test failed")
    
    db_path = os.path.join(STUDY_DIR, "optuna.db")
    storage_url = f"sqlite:///{db_path}"
    
    optimization_config = vllm_config.get("optimization", {})
    
    logger.info(f"MLPerf Tokens/Second Optimization")
    logger.info(f"Model: {model}")
    logger.info(f"Dataset: {dataset}")
    n_trials = args.n_trials if args.n_trials else optimization_config.get("n_trials", 200)
    logger.info(f"Number of trials: {n_trials}")
    logger.info(f"Objective: Maximize tokens per second")
    
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
        
        logger.info(f"\n Grid Sampler Configuration")
        logger.info(f"Parameters to optimize:")
        for param, values in search_space.items():
            logger.info(f"{param}: {len(values)} values {values}")
        logger.info(f"Total possible combinations: {total_combinations:,}")
        logger.info(f"GridSampler will test all {total_combinations:,} combinations")
        
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
    
    # Check if parallel optimization is requested
    if args.parallel:
        gpu_ids = [int(gpu.strip()) for gpu in args.gpus.split(',')]
        logger.info(f"Using parallel optimization with {sampler_name} sampler on GPUs: {gpu_ids}")
        logger.info("Result: Maximize tokens per second using MLPerf with parallel trials")
        
        from src.serving.optimization import run_parallel_trials
        study = run_parallel_trials(
            study, model, dataset, vllm_config, 
            STUDY_DIR, STUDY_ID, gpu_ids, n_trials
        )
    else:
        logger.info(f"Using single-objective optimization with {sampler_name} sampler")
        logger.info("Result: Maximize tokens per second using MLPerf")

        logger.info(f"\nStarting MLPerf optimization trials...")
        
        if sampler_name == "grid":
            logger.info(f"Will run ALL grid combinations (no n_trials limit)")
            study.optimize(objective_function)
        else:
            logger.info(f"Will run {n_trials} optimization trials")
            study.optimize(objective_function, n_trials=n_trials)

    logger.info("\nMLPerf Optimization Results:")
    logger.info(f"Best throughput achieved: {study.best_trial.value:.2f} tokens/s")
    logger.info(f"Best trial parameters: {study.best_trial.params}")
    
    if baseline_metrics is not None:
        baseline_throughput = baseline_metrics.get("tokens_per_second", 0)
        if baseline_throughput > 0:
            improvement = ((study.best_trial.value - baseline_throughput) / baseline_throughput) * 100
            logger.info(f"Improvement over baseline: {improvement:.2f}%")

if __name__ == "__main__":
    main()