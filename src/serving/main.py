import subprocess
import time
import requests
import os
import signal
import sys
import json
import glob
import re
import optuna
import yaml
import socket


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

def log_stream(stream, log_file, prefix):
    """
    Continuously read from a stream and write to a log file
    """
    with open(log_file, 'a') as f:
        while True:
            line = stream.readline()
            if not line:
                break
            print(f"[{prefix}] {line.strip()}")
            f.write(line)
            f.flush()

def build_vllm_command(model_name, port, candidate_flags):

    """
    Assembles a vLLM CLI command with both fix and candidate flags

    Args:
        --model: Qwen/Qwen3-32B-FP8 (smaller model: Qwen/Qwen3-1.7B)
        --max-model-len: 8192
        --disable-log-requests: True
        candidate_flags: List of candidate flags to be added to the command
    """

    cmd = [
        "vllm",
        "serve",
        model_name,
        "--max-model-len", "8192",
        "--port", str(port),
        "--disable-log-requests"
    ]

    cmd += candidate_flags

    return cmd

def check_port_available(port):
    """Check if a port is available before starting the server"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(('localhost', port))
        return True
    except socket.error:
        return False
    finally:
        sock.close()

def get_last_log_lines(log_file, n=20):
    """Get the last n lines from a log file"""
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            return ''.join(lines[-n:]) if lines else ''
    except Exception as e:
        return f"Could not read log file: {str(e)}"

def start_vllm_server(cmd, ready_pattern="Application startup complete", timeout=30000, log_file=None):
    """
    Launches the vLLM server and continuously logs its output
    
    Args:
        cmd: Command to run
        ready_pattern: Pattern to look for in logs to determine server is ready
        timeout: Timeout in seconds (if None, uses value from config)
        log_file: Path to log file
    """
    if not check_port_available(int(cmd[cmd.index('--port') + 1])):
        raise RuntimeError(f"Port {cmd[cmd.index('--port') + 1]} is already in use")

    print(f"Launching vLLM server {' '.join(cmd)}")
    
    try:
        with open(log_file, 'w') as f:
            proc = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=f,
                stdin=subprocess.PIPE,
                text=True,
                bufsize=1,
                preexec_fn=os.setsid
            )
    except FileNotFoundError:
        raise RuntimeError(f"vLLM binary not found. Is vllm installed and in PATH?")
    except OSError as e:
        raise RuntimeError(f"Failed to start vLLM server: {e.strerror}\nCommand: {' '.join(cmd)}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error starting vLLM server: {str(e)}")

    start_time = time.time()
    server_ready = False

    while True:
        if proc.poll() is not None:
            last_logs = get_last_log_lines(log_file)
            raise RuntimeError(
                f"vLLM server exited unexpectedly with code {proc.returncode}.\n"
                f"Last log lines:\n{last_logs}\n"
                f"Check the full log file for details: {log_file}"
            )
        
        elapsed = time.time() - start_time
        if elapsed > timeout:
            proc.kill()
            raise TimeoutError(f"vLLM server did not start within {timeout/1000} seconds")

        if not server_ready:
            try:
                with open(log_file, 'r') as f:
                    content = f.read()
                    if ready_pattern in content:
                        time.sleep(10)
                        server_ready = True
                        return proc
            except IOError as e:
                proc.kill()
                raise RuntimeError(f"Failed to read vLLM log file: {str(e)}")

        time.sleep(0.1)

def stop_vllm_server(proc, grace_period=5):
    """
    Terminates the vLLM server process
    
    Args:
        proc: Process to stop
        grace_period: Grace period in seconds (if None, uses value from config)
    """
    if proc is None:
        print("No vLLM process to stop")
        return

    print(f"Stopping vLLM process {proc.pid}...")
    killed = False

    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
    except ProcessLookupError:
        print("Process already terminated")
        return
    except Exception as e:
        print(f"Warning: SIGINT failed: {str(e)}")
    
    # Wait for graceful shutdown
    deadline = time.time() + grace_period
    while time.time() < deadline:
        if proc.poll() is not None:
            print("vLLM process exited cleanly")
            return
        time.sleep(0.1)

    # If still running, try SIGKILL
    if proc.poll() is None:
        print(f"Process {proc.pid} still alive after {grace_period}s; sending SIGKILL")
        try:
            proc.kill()
            proc.wait(timeout=5)  # Wait up to 5 more seconds for SIGKILL
            killed = True
        except Exception as e:
            print(f"Warning: SIGKILL failed: {str(e)}")
    
    if not killed and proc.poll() is None:
        print(f"Warning: Process {proc.pid} could not be terminated")
    else:
        print("vLLM process terminated")

def query_vllm_server(port, prompt, max_retries=3, retry_delay=1):
    """
    Queries the vLLM server with a given prompt
    
    Args:
        port: Server port number
        prompt: Input prompt text
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
    
    Returns:
        Generated text response
    
    Raises:
        RuntimeError: If server is unreachable or returns invalid response
    """
    url = f"http://localhost:{port}/v1/completions"
    payload = {
        "prompt": prompt
    }

    last_error = None
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            response_json = response.json()
            
            if not response_json.get("choices"):
                raise RuntimeError("Server returned empty choices array")
            
            return response_json["choices"][0]["text"]
            
        except requests.Timeout:
            last_error = "Request timed out"
        except requests.ConnectionError:
            last_error = "Failed to connect to server"
        except requests.HTTPError as e:
            last_error = f"Server returned HTTP {e.response.status_code}"
        except (KeyError, json.JSONDecodeError) as e:
            last_error = f"Invalid response format: {str(e)}"
        except Exception as e:
            last_error = f"Unexpected error: {str(e)}"
            
        if attempt < max_retries - 1:
            print(f"Attempt {attempt + 1} failed: {last_error}. Retrying in {retry_delay}s...")
            time.sleep(retry_delay)
        
    raise RuntimeError(f"Failed to query vLLM server after {max_retries} attempts. Last error: {last_error}")

def run_guidellm(guidellm_args, log_file):
    """
    Launches guidellm as a subprocess, give input text and capture output
    guidellm_args is a list of args for the guidellm command
    """

    cmd = ["guidellm"] + guidellm_args
    print(f"Launching guidellm: {' '.join(cmd)}")

    try:
        with open(log_file, 'w') as f:
            proc = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=f,
                stdin=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
    except FileNotFoundError:
        raise RuntimeError("guidellm binary not found. Is guidellm installed and in PATH?")
    except OSError as e:
        raise RuntimeError(f"Failed to start guidellm: {e.strerror}\nCommand: {' '.join(cmd)}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error starting guidellm: {str(e)}")

    try:
        proc.wait(timeout=600)  # 10 minute timeout
        print(f"guidellm process completed with return code: {proc.returncode}")
    except subprocess.TimeoutExpired:
        proc.kill()
        raise RuntimeError("guidellm process timed out after 10 minutes")
    except Exception as e:
        proc.kill()
        raise RuntimeError(f"Error waiting for guidellm: {str(e)}")
    
    if proc.returncode != 0:
        last_logs = get_last_log_lines(log_file)
        raise RuntimeError(
            f"guidellm failed with code {proc.returncode}\n"
            f"Last log lines:\n{last_logs}\n"
            f"Check the full log file for details: {log_file}"
        )

    return proc.returncode


# -----------------------------
# Optuna
# -----------------------------

def parse_benchmarks(bench_file):
    """
    Parse the benchmark results from the specified benchmark file
    
    Args:
        bench_file: Path to the benchmark results JSON file
    Returns:
        Dictionary containing all benchmark metrics
    Raises:
        RuntimeError: If the file is missing, invalid JSON, or missing required metrics
    """
    if not os.path.exists(bench_file):
        raise RuntimeError(f"Benchmark file missing: {bench_file}")

    try:
        with open(bench_file) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON in benchmark file {bench_file}: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Error reading benchmark file {bench_file}: {str(e)}")
    
    try:
        stats = data["benchmarks"][0]
        metrics = stats["metrics"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Invalid benchmark data structure in {bench_file}: {str(e)}")

    REQUIRED_METRICS = [
        "requests_per_second",
        "request_concurrency",
        "request_latency",
        "prompt_token_count",
        "output_token_count",
        "time_to_first_token_ms",
        "time_per_output_token_ms",
        "inter_token_latency_ms",
        "output_tokens_per_second",
        "tokens_per_second"
    ]

    result = {}
    for metric in REQUIRED_METRICS:
        try:
            result[metric] = metrics[metric]["successful"]["median"]
        except KeyError as e:
            raise RuntimeError(f"Missing required metric in {bench_file}: {metric}")
    
    return result

def objective(trial):
    """
    Objective function for Optuna
    
    Returns:
        float: The optimization metric (requests per second)
    
    Raises:
        optuna.TrialPruned: If the trial fails, after recording the error
    """
    try:
        port = 8000
        model = "Qwen/Qwen3-1.7B"

        # Get parameters from config
        candidate_flags = []
        for param_name, param_config in vllm_config["parameters"].items():
            if param_config["enabled"]:
                param_value = trial.suggest_int(
                    param_name,
                    param_config["range"]["start"],
                    param_config["range"]["end"],
                    step=param_config["range"]["step"]
                )
                candidate_flags.extend([f"--{param_name}", str(param_value)])

        trial_id = trial.number
        vllm_log_file = os.path.join(VLLM_LOGS_DIR, f"vllm_server_logs_{STUDY_ID}.{trial_id}.log")
        guidellm_log_file = os.path.join(GUIDELLM_LOGS_DIR, f"guidellm_logs_{STUDY_ID}.{trial_id}.log")
        
        print(f"\nStarting trial {trial_id}")
        print(f"vLLM log file: {vllm_log_file}")
        print(f"guidellm log file: {guidellm_log_file}")

        vllm_cmd = build_vllm_command(model_name=model, port=port, candidate_flags=candidate_flags)
        vllm_proc = start_vllm_server(vllm_cmd, log_file=vllm_log_file)

        bench_file = os.path.join(
            STUDY_DIR,
            f"benchmarks_{STUDY_ID}.{trial_id}.json"
        )

        try:
            print("Starting guidellm benchmark...")
            guidellm_args = [
                "benchmark",
                "--target",    "http://localhost:8000",
                "--model",     model,
                "--processor", model,
                "--data=" + '{"prompt_tokens":550,'
                            '"prompt_tokens_stdev":150,'
                            '"prompt_tokens_min":400,'
                            '"prompt_tokens_max":700,'
                            '"output_tokens":150,'
                            '"output_tokens_stdev":15,'
                            '"output_tokens_min":135,'
                            '"output_tokens_max":165}',
                "--rate-type",    "concurrent",
                "--max-requests", "100",
                "--rate",         "10",
                "--output-path",  bench_file
            ]

            run_guidellm(guidellm_args, guidellm_log_file)
            print("guidellm benchmark completed successfully")

            metrics = parse_benchmarks(bench_file)
            
            # Store all metrics as user attributes
            for metric_name, value in metrics.items():
                trial.set_user_attr(metric_name, value)

            return float(metrics["requests_per_second"])  # Ensure we return a float

        finally:
            stop_vllm_server(vllm_proc)
            print("vLLM server stopped.")
            # Wait between trials to allow port to be released
            interval = vllm_config["settings"].get("trial_interval", 30)
            print(f"Waiting {interval} seconds before next trial...")
            time.sleep(interval)

    except Exception as e:
        print(f"Error during trial {trial.number}:", str(e), file=sys.stderr)
        import traceback
        traceback.print_exc()
        trial.set_user_attr("error", str(e))
        trial.set_user_attr("traceback", traceback.format_exc())
        raise optuna.TrialPruned(f"Trial failed: {str(e)}")

def run_baseline_test():
    """
    Run a baseline test with default vLLM parameters
    """
    port = 8000
    model = "Qwen/Qwen3-32B-FP8"
    
    vllm_log_file = os.path.join(VLLM_LOGS_DIR, f"vllm_server_logs_{STUDY_ID}.baseline.log")
    guidellm_log_file = os.path.join(GUIDELLM_LOGS_DIR, f"guidellm_logs_{STUDY_ID}.baseline.log")
    
    print(f"\nRunning baseline test")
    print(f"vLLM log file: {vllm_log_file}")
    print(f"guidellm log file: {guidellm_log_file}")
    
    vllm_cmd = build_vllm_command(model_name=model, port=port, candidate_flags=[])
    vllm_proc = start_vllm_server(vllm_cmd, log_file=vllm_log_file)

    bench_file = os.path.join(STUDY_DIR, f"benchmarks_{STUDY_ID}.baseline.json")

    try:
        print("Starting guidellm benchmark for baseline...")
        guidellm_args = [
            "benchmark",
            "--target",    "http://localhost:8000",
            "--model",     model,
            "--processor", model,
            "--data=" + '{"prompt_tokens":550,'
                        '"prompt_tokens_stdev":150,'
                        '"prompt_tokens_min":400,'
                        '"prompt_tokens_max":700,'
                        '"output_tokens":150,'
                        '"output_tokens_stdev":15,'
                        '"output_tokens_min":135,'
                        '"output_tokens_max":165}',
            "--rate-type",    "concurrent",
            "--max-requests", "100",
            "--rate",         "10",
            "--output-path",  bench_file
        ]

        run_guidellm(guidellm_args, guidellm_log_file)
        print("Baseline guidellm benchmark completed successfully")

        metrics = parse_benchmarks(bench_file)
        return metrics

    except Exception as e:
        print("Error during baseline test:", str(e), file=sys.stderr)
        import traceback; traceback.print_exc()
        return None

    finally:
        stop_vllm_server(vllm_proc)
        print("Baseline vLLM server stopped.")

def main():
    db_path     = os.path.join(STUDY_DIR, "optuna.db")
    storage_url = f"sqlite:///{db_path}"
    study = optuna.create_study(
        storage=storage_url,
        study_name=f"vllm_tuning_run{STUDY_ID}",
        direction="maximize",
        load_if_exists=True
    )

    # Run baseline test first
    print("Running baseline test with default parameters...")
    baseline_metrics = run_baseline_test()
    if baseline_metrics is not None:
        print(f"Baseline performance: {baseline_metrics['requests_per_second']} requests/second")
        # trial = optuna.trial.create_trial(
        #     params={},
        #     value=baseline_metrics["requests_per_second"],
        #     user_attrs=baseline_metrics
        # )
        # study.add_trial(trial)
        # print("Baseline test completed successfully")

    else:
        print("Baseline test failed")

    # Run Optuna trials
    print("\nStarting Optuna trials...")
    study.optimize(objective, n_trials=10)

    print("\nOptimization Results:")
    print(f"Best trial value: {study.best_trial.value}")
    print(f"Best trial parameters: {study.best_trial.params}")
    if baseline_metrics is not None:
        improvement = ((study.best_trial.value - baseline_metrics["requests_per_second"]) / baseline_metrics["requests_per_second"]) * 100
        print(f"Improvement over baseline: {improvement:.2f}%")

if __name__ == "__main__":
    main()