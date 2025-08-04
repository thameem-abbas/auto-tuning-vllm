import os
import sys
import logging
from src.serving.vllm_server import build_vllm_command, start_vllm_server, stop_vllm_server
from src.serving.benchmarking import run_mlperf, parse_benchmarks

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_baseline_test(model=None, dataset=None, study_dir=None, study_id=None, gpu_id=0):
    """Run baseline MLPerf test with default vLLM parameters"""
    port = 8000
    
    if model is None:
        model = "meta-llama/Llama-3.1-8B-Instruct"
    
    # Create baseline directory
    baseline_dir = os.path.join(study_dir, "baseline")
    os.makedirs(baseline_dir, exist_ok=True)
    
    vllm_log_file = os.path.join(baseline_dir, f"vllm_server_gpu_{gpu_id}.log")
    mlperf_log_file = os.path.join(baseline_dir, "mlperf_console.log")
    
    logger.info(f"\nRunning MLPerf baseline test")
    logger.info(f"Model: {model}")
    logger.info(f"Dataset: {dataset}")
    logger.info(f"GPU: {gpu_id}")
    logger.info(f"Baseline directory: {baseline_dir}")
    
    # Use default vLLM parameters (no optimization flags)
    vllm_cmd = build_vllm_command(model_name=model, port=port, candidate_flags=[], gpu_id=gpu_id)
    vllm_proc = start_vllm_server(vllm_cmd, log_file=vllm_log_file, gpu_id=gpu_id)

    try:
        logger.info("Starting baseline MLPerf benchmark...")
        # FIXME: need to pass qps=0 if this is offline
        summary_file = run_mlperf(model, dataset, baseline_dir, mlperf_log_file, qps=6)
        logger.info("Baseline MLPerf benchmark completed successfully")

        metrics = parse_benchmarks(summary_file)
        return metrics

    except Exception as e:
        logger.error(f"Error during baseline MLPerf test: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

    finally:
        stop_vllm_server(vllm_proc)
        logger.info("Baseline vLLM server stopped.") 