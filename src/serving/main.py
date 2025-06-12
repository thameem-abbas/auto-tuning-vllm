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

# TODO
# [ ] max_num_batched_tokens


SRC_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.abspath(os.path.join(SRC_DIR, ".."))
STUDIES_ROOT = os.path.join(PROJECT_DIR, "studies")
os.makedirs(STUDIES_ROOT, exist_ok=True)

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
        --model: Qwen/Qwen3-1.7B
        --max-model-len: 8192
        --disable-log-requests: True
        candidate_flags: List of candidate flags to be added to the command
    """

    cmd = [
        "vllm",
        "serve",
        model_name,
        "--max-model-len", "4096",
        "--port", str(port),
        "--disable-log-requests"
    ]

    cmd += candidate_flags

    return cmd

def start_vllm_server(cmd, ready_pattern="Application startup complete", timeout=30000, log_file=None):
    """
    Lanches the vLLM server and continuously logs its output
    """

    print(f"Launching vLLM server {' '.join(cmd)}")
    
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

    start_time = time.time()
    server_ready = False

    while True:
        if proc.poll() is not None:
            raise RuntimeError(
                f"vLLM server exited unexpectedly with code {proc.returncode}. "
                f"Check the log file for details: {log_file}"
            )
        
        elapsed = time.time() - start_time
        if elapsed > timeout:
            proc.kill()
            raise TimeoutError("vLLM server did not start within the timeout period")

        if not server_ready and os.path.exists(log_file):
            with open(log_file, 'r') as f:
                content = f.read()
                if ready_pattern in content:
                    time.sleep(10)
                    server_ready = True
                    return proc

        time.sleep(0.1)

def stop_vllm_server(proc, grace_period=5):
    """
    Terminates the vLLM server process
    """
    if proc is None:
        print("No vLLM process to stop")
        return

    print(f"Stopping vLLM process {proc.pid}...")

    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        
        # Wait for graceful shutdown
        deadline = time.time() + grace_period
        while time.time() < deadline:
            if proc.poll() is not None:
                print("vLLM process exited cleanly")
                return
            time.sleep(0.1)

        if proc.poll() is None:
            print(f"Process {proc.pid} still alive after {grace_period}s; sending SIGKILL")
            proc.kill()
            proc.wait()
            print("vLLM process terminated")
    except Exception as e:
        print(f"Error stopping vLLM process: {str(e)}")
        try:
            if proc.poll() is None:
                proc.kill()
        except:
            pass
    finally:
        print("stop vllm completed")

def query_vllm_server(port, prompt):
    """
    Queries the vLLM server with a given prompt
    """

    url = f"http://localhost:{port}/v1/completions"
    payload = {
        "prompt": prompt
    }

    response = requests.post(url, json=payload)
    response.raise_for_status()
    response_json = response.json()
    
    if response_json["choices"] and len(response_json["choices"]) > 0:
        return response_json["choices"][0]["text"]
    return "No response generated"

def run_guidellm(guidellm_args, log_file):
    """
    Lauches guidellm as a subprocess, give input text and capture output
    guidellm_args is a list of args for the guidellm command
    """

    cmd = ["guidellm"] + guidellm_args
    print(f"Launching guidellm: {' '.join(cmd)}")

    with open(log_file, 'w') as f:
        proc = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=f,
            stdin=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    try:
        proc.wait(timeout=600)  # 10 minute timeout
        print(f"guidellm process completed with return code: {proc.returncode}")
    except subprocess.TimeoutExpired:
        print("guidellm process timed out after 5 minutes")
        proc.kill()
        raise RuntimeError("guidellm process timed out after 5 minutes")
    
    if proc.returncode != 0:
        print(f"guidellm failed with code {proc.returncode}")
        print("Last few lines of log file:")
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                for line in lines[-10:]:
                    print(line.strip())
        except Exception as e:
            print(f"Could not read log file: {e}")
        raise RuntimeError(f"guidellm failed with code {proc.returncode}. Check the log file for details: {log_file}")

    return proc.returncode


# -----------------------------
# Optuna
# -----------------------------

def parse_benchmarks(bench_file):
    """
    Parse the benchmark results from the specified benchmark file
    
    Args:
        bench_file: Path to the benchmark results JSON file
    """
    with open(bench_file) as f:
        data = json.load(f)
    
    stats = data["benchmarks"][0]
    median_rps = stats["metrics"]["requests_per_second"]["successful"]["median"]
    
    return median_rps

def objective(trial):
    """
    Objective function for Optuna
    """

    port = 8000
    model = "Qwen/Qwen3-1.7B"

    # Define the parameter to tune
    max_num_batched_tokens = trial.suggest_int("max_num_batched_tokens", 8192, 65536, step=4096)

    trial_id = trial.number + 1
    vllm_log_file = os.path.join(VLLM_LOGS_DIR, f"vllm_server_logs_{STUDY_ID}.{trial_id}.log")
    guidellm_log_file = os.path.join(GUIDELLM_LOGS_DIR, f"guidellm_logs_{STUDY_ID}.{trial_id}.log")
    
    print(f"\nStarting trial {trial_id}")
    print(f"vLLM log file: {vllm_log_file}")
    print(f"guidellm log file: {guidellm_log_file}")

    candidate_flags = [
        "--max-num-batched-tokens", str(max_num_batched_tokens)
    ]
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
            "--model",     "Qwen/Qwen3-1.7B",
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

        return parse_benchmarks(bench_file)

    except Exception as e:
        print("Error during vLLM to guidellm processing:", str(e), file=sys.stderr)
        import traceback; traceback.print_exc()
        return None

    finally:
        stop_vllm_server(vllm_proc)
        print("vLLM server stopped.")

def main():
    db_path     = os.path.join(STUDY_DIR, "optuna.db")
    storage_url = f"sqlite:///{db_path}"
    study = optuna.create_study(
        storage=storage_url,
        study_name=f"vllm_tuning_run{STUDY_ID}",
        direction="maximize",
        load_if_exists=True
    )
    study.optimize(objective, n_trials=10)

    print(study.best_trial.value, study.best_trial.params)

if __name__ == "__main__":
    main()