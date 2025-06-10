import vllm
import subprocess
import time
import requests
import os
import signal
import sys
import torch
import json
import glob
import re

from vllm import LLM, SamplingParams

import optuna

## TODO:
# - [X] BUILD Optuna around vllm server and guidellm
# - [ ] Log all the benchmark for each trial (name it benckmarks_1.1.json, benchmarks_1.2.json, etc.. For next benchmark run is it benchmarks_2.1.json, benchmarks_2.2.json, etc.)
# - [ ] Log the vllm server logs into a file that can be related to the benchmark run vllm_server_logs_1.1, vllm_server_logs_1.2, etc.

SRC_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.abspath(os.path.join(SRC_DIR, ".."))
BENCH_ROOT = os.path.join(PROJECT_DIR, "benchmarks")
os.makedirs(BENCH_ROOT, exist_ok=True)

# Initialize run tracking at module level
run_dirs = glob.glob(os.path.join(BENCH_ROOT, "benchmarks_*"))
run_ids = []
for d in run_dirs:
    m = re.match(r".*benchmarks_(\d+)$", d)
    if m:
        run_ids.append(int(m.group(1)))
RUN_ID = max(run_ids) + 1 if run_ids else 1
RUN_DIR = os.path.join(BENCH_ROOT, f"benchmarks_{RUN_ID}")
os.makedirs(RUN_DIR, exist_ok=True)
print(f"Logging this run's benchmarks to: {RUN_DIR}")


def build_vllm_command(model_name, port, candidate_flags):

    """
    Assembles a vLLM CLI command with both fix and candidate flags

    Args:
        --model: Qwen/Qwen3-32B-FP8
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

def start_vllm_server(cmd, ready_pattern="Application startup complete", timeout=10000): # TODO: fix timeout to instead not be hardcoded
    """
    Lanches the vLLM server
    """

    print(f"Launching vLLM server {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, preexec_fn=os.setsid)

    start_time = time.time()

    while True:
        if proc.poll() is not None:
            remaining = proc.stderr.read()
            raise RuntimeError(
                f"vLLM server exited unexpectedly with code {proc.returncode}. "
                f"Last stderr output: {remaining}"
            )
        
        elapsed = time.time() - start_time
        if elapsed > timeout:
            proc.kill()
            raise TimeoutError("vLLM server did not start within the timeout period")
        
        line = proc.stderr.readline()
        if not line:
            time.sleep(0.1)
            continue

        print(f"[vLLM stderr] {line.strip()}")

        if ready_pattern in line:
            print("vLLM server is ready")
            return proc
        
        
def stop_vllm_server(proc, grace_period=5):
    """
    Terminates the vLLM server process
    """
    if proc is None:
        print("No vLLM process to stop")
        return

    print(f"Stopping vLLM process {proc.pid}...")

    try:    
        # First try SIGINT for graceful shutdown
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        
        # Wait for graceful shutdown
        deadline = time.time() + grace_period
        while time.time() < deadline:
            if proc.poll() is not None:
                print("vLLM process exited cleanly")
                return
            time.sleep(0.1)

        # If still running force kill
        if proc.poll() is None:
            print(f"Process {proc.pid} still alive after {grace_period}s; sending SIGKILL")
            proc.kill()
            proc.wait()  # Wait
            print("vLLM process terminated")
    except Exception as e:
        print(f"Error stopping vLLM process: {str(e)}")
        # kill if still running
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
    # llm = LLM(model="Qwen/Qwen3-32B-FP8", port=port)
    # sampling_params = SamplingParams(temperature=0.1, top_p=0.95, max_tokens=512)
    # response = llm.generate(prompt, sampling_params=sampling_params)
    # return response[0].outputs[0].text

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

def run_guidellm(guidellm_args):
    """
    Lauches guidellm as a subprocess, give input text and capture output
    guidellm_args is a list of args for the guidellm command
    """

    cmd = ["guidellm"] + guidellm_args
    print(f"Launching guidellm: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    out, err = proc.communicate()  # no stdin
    if proc.returncode != 0:
        raise RuntimeError(f"guidellm failed (code {proc.returncode}):\n{err}")
    return out


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
    
    # Get the first benchmark result
    stats = data["benchmarks"][0]
    
    # Extract the median throughput from successful requests
    median_throughput = stats["metrics"]["requests_per_second"]["successful"]["median"]
    
    return median_throughput

def objective(trial):
    """
    Objective function for Optuna
    """

    max_gpus = torch.cuda.device_count()

    port = 8000
    model = "Qwen/Qwen3-32B-FP8"

    vllm_cmd = build_vllm_command(model_name=model, port=port, candidate_flags="")
    vllm_proc = start_vllm_server(vllm_cmd)

    trial_id   = trial.number + 1
    bench_file = os.path.join(
        RUN_DIR,
        f"benchmarks_{RUN_ID}.{trial_id}.json"
    )

    try:
        guidellm_args = [
            "benchmark",
            "--target", "http://localhost:8000",
            "--model",            "Qwen/Qwen3-32B-FP8",
            "--data",             '{"prompt_tokens":550,"prompt_tokens_stdev":150,"prompt_tokens_min":400,"prompt_tokens_max":700,"output_tokens":150,"output_tokens_stdev":15,"output_tokens_min":135,"output_tokens_max":165}',
            "--rate-type",        "concurrent",
            "--max-requests",     "100",
            "--rate",             "10",
            "--output-path", bench_file,
        ]

        run_guidellm(guidellm_args)

        # Parse the benchmark results from the specified file
        return parse_benchmarks(bench_file)

    except Exception as e:
        print("Error during vLLM to guidellm processing:", str(e), file=sys.stderr)
        import traceback; traceback.print_exc()

    finally:
        stop_vllm_server(vllm_proc)
        print("vLLM server stopped.")



def main():
    db_path     = os.path.join(RUN_DIR, "optuna.db")
    storage_url = f"sqlite:///{db_path}"
    study = optuna.create_study(
        storage=storage_url,
        study_name=f"vllm_tuning_run{RUN_ID}",
        direction="maximize",
        load_if_exists=True
    )
    study.optimize(objective, n_trials=10)

    print(study.best_trial.value, study.best_trial.params)

if __name__ == "__main__":
    main()