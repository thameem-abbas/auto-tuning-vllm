import subprocess
import time
import requests
import os
import signal
import json
from src.serving.utils import check_port_available, get_last_log_lines

def build_vllm_command(model_name, port, candidate_flags):
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

def start_vllm_server(cmd, ready_pattern="Application startup complete", timeout=30000, log_file=None):
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
    
    deadline = time.time() + grace_period
    while time.time() < deadline:
        if proc.poll() is not None:
            print("vLLM process exited cleanly")
            return
        time.sleep(0.1)

    if proc.poll() is None:
        print(f"Process {proc.pid} still alive after {grace_period}s; sending SIGKILL")
        try:
            proc.kill()
            proc.wait(timeout=5)
            killed = True
        except Exception as e:
            print(f"Warning: SIGKILL failed: {str(e)}")
    
    if not killed and proc.poll() is None:
        print(f"Warning: Process {proc.pid} could not be terminated")
    else:
        print("vLLM process terminated")

def query_vllm_server(port, prompt, max_retries=3, retry_delay=1):
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