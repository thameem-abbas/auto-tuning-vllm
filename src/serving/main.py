import vllm
import subprocess
import time
import requests
import os
import signal
import sys
import torch

from vllm import LLM, SamplingParams

import optuna

## TODO:
# - [ ] BUILD Optuna around vllm server and guidellm


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
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

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
        # First try SIGTERM for graceful shutdown
        proc.terminate()
        
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

def run_guidellm(input_text, guidellm_args):
    """
    Lauches guidellm as a subprocess, give input text and capture output
    guidellm_args is a list of args for the guidellm command
    """

    cmd = ["guidellm"] + guidellm_args
    print(f"Launching guidellm: {' '.join(cmd)}")

    if input_text is None:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        out, err = proc.communicate()
    else:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        out, err = proc.communicate(input=input_text)

    if proc.returncode != 0:
        print("guidellm failed with error:", err, file=sys.stderr)
        return None
    return out


# -----------------------------
# Optuna
# -----------------------------
def objective(trial):
    """
    Objective function for Optuna
    """

    max_gpus = torch.cuda.device_count()

    port = 8000
    model = "Qwen/Qwen3-32B-FP8"

    vllm_cmd = build_vllm_command(model_name=model, port=port, candidate_flags="")
    vllm_proc = start_vllm_server(vllm_cmd)

    try:
        # sanity check
        response = query_vllm_server(port, "Hello from Red Hat")

        guidellm_args = [
            "benchmark",
            "--target", "http://localhost:8000",
            "--model",            "Qwen/Qwen3-32B-FP8",
            "--data",             '{"prompt_tokens":550,"prompt_tokens_stdev":150,"prompt_tokens_min":400,"prompt_tokens_max":700,"output_tokens":150,"output_tokens_stdev":15,"output_tokens_min":135,"output_tokens_max":165}',
            "--rate-type",        "concurrent",
            "--max-requests",     "100",
            "--rate",             "10",
        ]

        output = run_guidellm(guidellm_args)

        

    except Exception as e:
        print("Error during vLLM to guidellm processing:", str(e), file=sys.stderr)
        import traceback; traceback.print_exc()

    finally:
        stop_vllm_server(vllm_proc)
        print("vLLM server stopped.")



def main():
    return

if __name__ == "__main__":
    main()