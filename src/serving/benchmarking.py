import subprocess
import os
import logging
from src.serving.utils import get_last_log_lines

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_mlperf(model_name, dataset_path, trial_dir, log_file, port=8000, qps=0):
    """
    Run MLPerf benchmark with the SUT_VLLM_SingleReplica.py script
    
    Args:
        model_name: HuggingFace model name (e.g., "meta-llama/Llama-3.1-8B")
        dataset_path: Path to dataset file
        trial_dir: Directory for this specific trial (baseline/, trial_1/, etc.)
        log_file: Path to log file for MLPerf console output
        port: Port number for the vLLM server (default: 8000)
    
    Returns:
        Path to the generated mlperf_log_summary.txt file
    """
    
    # Change to MLPerf directory
    mlperf_dir = os.path.abspath("mlperf-inference-5.1-redhat/language/llama3.1-8b")
    if not os.path.exists(mlperf_dir):
        raise RuntimeError(f"MLPerf directory not found: {mlperf_dir}")
    
    # Create metrics CSV path in trial directory
    metrics_csv_path = os.path.join(trial_dir, "mlperf_metrics.csv")
    
    # Build MLPerf command with output_log_dir
    # If qps is 0 then this is an offline run
    cmd = [
        "python3", "SUT_VLLM_SingleReplica.py" if qps == 0 else "SUT_VLLM_SingleReplica_Server.py",
        "--model-name", model_name,
        "--api-server-url", f"http://0.0.0.0:{port}",
        # "--print-timing",
        "--batch-size", "5000", #This is only used for offline, ignored for server
        # "--print-histogram",
        # "--sort-by-token-contents",
        # "--enable-metrics-csv",
        # "--metrics-csv-path", metrics_csv_path,
        "--output-log-dir", trial_dir,
        "--user-conf", "user.conf"
    ]
    if qps != 0:
        cmd.append("--scenario")
        cmd.append("Server")
        cmd.append("--target-qps")
        cmd.append(str(qps))
    
    # Add dataset path (convert to absolute path)
    if dataset_path:
        # Convert to absolute path since MLPerf runs from a different directory
        abs_dataset_path = os.path.abspath(dataset_path)
        cmd.extend(["--dataset-path", abs_dataset_path])
        logger.info(f"Using absolute dataset path: {abs_dataset_path}")
    
    logger.info(f"Running MLPerf benchmark in directory: {mlperf_dir}")
    logger.debug(f"MLPerf command: {' '.join(cmd)}")
    logger.info(f"MLPerf logs will be saved to: {trial_dir}")
    logger.info(f"Metrics CSV will be saved to: {metrics_csv_path}")
    
    try:
        with open(log_file, 'w') as f:
            proc = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=f,
                stdin=subprocess.PIPE,
                text=True,
                bufsize=1,
                cwd=mlperf_dir  # Run from MLPerf directory
            )
    except FileNotFoundError:
        raise RuntimeError(f"MLPerf script not found. Check that {mlperf_dir}/SUT_VLLM_SingleReplica.py exists")
    except OSError as e:
        raise RuntimeError(f"Failed to start MLPerf: {e.strerror}\nCommand: {' '.join(cmd)}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error starting MLPerf: {str(e)}")

    try:
        # MLPerf can take a long time, wait up to 40 minutes
        proc.wait(timeout=900)
        logger.info(f"MLPerf process completed with return code: {proc.returncode}")
    except subprocess.TimeoutExpired:
        proc.kill()
        raise RuntimeError("MLPerf process timed out after 40 minutes")
    except Exception as e:
        proc.kill()
        raise RuntimeError(f"Error waiting for MLPerf: {str(e)}")
    
    if proc.returncode != 0:
        last_logs = get_last_log_lines(log_file)
        raise RuntimeError(
            f"MLPerf failed with code {proc.returncode}\n"
            f"Last log lines:\n{last_logs}\n"
            f"Check the full log file for details: {log_file}"
        )

    # Return path to summary file in trial directory
    summary_file = os.path.join(trial_dir, "mlperf_log_summary.txt")
    if not os.path.exists(summary_file):
        raise RuntimeError(f"MLPerf summary file not found: {summary_file}")
    
    logger.info(f"MLPerf summary file available at: {summary_file}")
    
    return summary_file

import re
import optuna
def parse_latency_values(file_path):
    """
    Parse MLPerf log file to extract 99.90 percentile latency values and throughput.
    
    Args:
        file_path (str): Path to the log file to parse
        
    Returns:
        tuple: (first_token_latency, output_token_latency, throughput)
               Returns 0 for values that are not found
    """
    first_token_latency = 0
    output_token_latency = 0
    throughput = 0
    
    try:
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                
                # Look for first token latency line
                if "99.90 percentile first token latency (ns)" in line:
                    # Extract the number after the colon
                    match = re.search(r'99\.90 percentile first token latency \(ns\)\s*:\s*(\d+)', line)
                    if match:
                        first_token_latency = int(match.group(1))
                
                # Look for output token latency line
                elif "99.90 percentile time to output token (ns)" in line:
                    # Extract the number after the colon
                    match = re.search(r'99\.90 percentile time to output token \(ns\)\s*:\s*(\d+)', line)
                    if match:
                        output_token_latency = int(match.group(1))
                # Look for throughput
                elif "Completed tokens per second" in line:
                    # Extract the floating point number after the colon
                    match = re.search(r'Completed tokens per second\s*:\s*(\d+\.?\d*)', line)
                    if match:
                        throughput = float(match.group(1))
    
    except FileNotFoundError:
        logger.error(f"File '{file_path}' not found")
        return (0, 0, 0)
    except Exception as e:
        logger.error(f"Error reading file '{file_path}': {e}")
        return (0, 0, 0)
    
    return (first_token_latency, output_token_latency, throughput)

def parse_benchmarks(summary_file):
    """
    Parse MLPerf summary file to extract throughput from line 8
    Looking for: "Tokens per second: 612.447"
    """
    if not os.path.exists(summary_file):
        raise RuntimeError(f"MLPerf summary file missing: {summary_file}")

    try:
        with open(summary_file, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        raise RuntimeError(f"Error reading MLPerf summary file {summary_file}: {str(e)}")

    # Parse the summary file first to see if this is a server run
    # If the file contains latency information, then prune the trial if the latency constraints are violated
    lat = parse_latency_values(summary_file)
    ttft = lat[0]
    tpot = lat[1]
    throughput = lat[2]
    if ttft != 0:
        # This is a server run
        if ttft > 2000000000 or tpot > 200000000:
            raise optuna.TrialPruned(f"Latency violated ttft: {str(ttft)} tpot: {str(tpot)}")
        # Return simple metrics dict focused only on throughput
        result = {
            "tokens_per_second": throughput,
            "output_tokens_per_second": throughput,  # Alias for compatibility
        }
        return result 

    #If you are here then this was an offline run
    
    # Check if we have at least 8 lines
    if len(lines) < 8:
        raise RuntimeError(f"MLPerf summary file has fewer than 8 lines: {summary_file}")
    
    # Read line 8 (index 7) for completed tokens per second
    line_8 = lines[7].strip()
    logger.debug(f"Parsing MLPerf summary: {summary_file}")
    logger.debug(f"Reading line 8: {line_8}")
    
    # Extract tokens per second value using simple string parsing
    if "Completed tokens per second:" in line_8:
        try:
            # Split by ":" and get the number part
            tokens_per_second = float(line_8.split(":")[-1].strip())
        except (ValueError, IndexError):
            raise RuntimeError(f"Could not parse completed tokens per second from line 8: {line_8}")
    else:
        raise RuntimeError(f"Line 8 does not contain 'Completed tokens per second:': {line_8}")
    
    logger.info(f"Extracted completed tokens per second: {tokens_per_second}")
    
    # Return simple metrics dict focused only on throughput
    result = {
        "tokens_per_second": tokens_per_second,
        "output_tokens_per_second": tokens_per_second,  # Alias for compatibility
    }
    
    return result 