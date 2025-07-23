import subprocess
import os
from src.serving.utils import get_last_log_lines

def run_mlperf(model_name, dataset_path, trial_dir, log_file, port=8000):
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
    cmd = [
        "python3", "SUT_VLLM_SingleReplica.py",
        "--model_name", model_name,
        "--api-server-url", f"http://0.0.0.0:{port}",
        # "--print-timing",
        "--batch_size", "32",
        # "--print-histogram",
        # "--sort-by-token-contents",
        # "--enable-metrics-csv",
        # "--metrics-csv-path", metrics_csv_path,
        "--output-log-dir", trial_dir
    ]
    
    # Add dataset path (convert to absolute path)
    if dataset_path:
        # Convert to absolute path since MLPerf runs from a different directory
        abs_dataset_path = os.path.abspath(dataset_path)
        cmd.extend(["--dataset_path", abs_dataset_path])
        print(f"Using absolute dataset path: {abs_dataset_path}")
    
    print(f"Running MLPerf benchmark in directory: {mlperf_dir}")
    print(f"MLPerf command: {' '.join(cmd)}")
    print(f"MLPerf logs will be saved to: {trial_dir}")
    print(f"Metrics CSV will be saved to: {metrics_csv_path}")
    
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
        proc.wait(timeout=2400)
        print(f"MLPerf process completed with return code: {proc.returncode}")
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
    
    print(f"MLPerf summary file available at: {summary_file}")
    
    return summary_file

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
    
    # Check if we have at least 8 lines
    if len(lines) < 8:
        raise RuntimeError(f"MLPerf summary file has fewer than 8 lines: {summary_file}")
    
    # Read line 8 (index 7) for tokens per second
    line_8 = lines[7].strip()
    print(f"Parsing MLPerf summary: {summary_file}")
    print(f"Reading line 8: {line_8}")
    
    # Extract tokens per second value using simple string parsing
    if "Tokens per second:" in line_8:
        try:
            # Split by ":" and get the number part
            tokens_per_second = float(line_8.split(":")[-1].strip())
        except (ValueError, IndexError):
            raise RuntimeError(f"Could not parse tokens per second from line 8: {line_8}")
    else:
        raise RuntimeError(f"Line 8 does not contain 'Tokens per second:': {line_8}")
    
    print(f"Extracted throughput: {tokens_per_second} tokens/s")
    
    # Return simple metrics dict focused only on throughput
    result = {
        "tokens_per_second": tokens_per_second,
        "output_tokens_per_second": tokens_per_second,  # Alias for compatibility
    }
    
    return result 