## Quick Start Guide

Follow these steps to install, configure, validate, and run an optimization study.

### 1) Create a Virtual Environment

Use either venv or uv (pick one):

```bash
# venv
python3 -m venv venv
source venv/bin/activate

# or uv (installs Python if needed and creates .venv)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate
```

### 2) Install Dependencies (from source)

Install the project in editable mode so CLI is available:

```bash
# pip
pip install -e .

# or uv
uv pip install -e .
```

### 3) Configure the Study

Start from the defaults at `examples/study_config_local_exec.yaml`.
- Modify the study name and model
- Choose optimization objectives (single-objective throughput or multi-objective throughput vs latency)
- Adjust parameter ranges for the tunable parameters

### 4) Validate Configuration

Always validate before launching optimization:

```bash
auto-tune-vllm validate --config examples/study_config_local_exec.yaml
```

### 5) Start Optimization

Provide a Python environment option for Ray workers. If you created `.venv`/`venv`, pass its path with `--venv-path`.
Use `--start-ray-head` if no Ray cluster is running; omit it if you have one.

```bash
auto-tune-vllm optimize \
  --config examples/study_config_local_exec.yaml \
  --venv-path .venv \
  --start-ray-head
```

Notes:
- Without `--start-ray-head`, the system tries to connect to an existing Ray cluster (address=auto) and fails if none is found.
- Alternatively, you can start Ray manually and skip `--start-ray-head`:
  ```bash
  ray start --head --dashboard-host=0.0.0.0
  ```

### 6) Monitor Logs

Open a separate terminal. After optimization starts, the CLI prints the exact logs command. For file logging, it looks like:

```bash
auto-tune-vllm logs --study-name <your_study_name> --log-path ./logs
```

If you configured PostgreSQL logging, use:

```bash
auto-tune-vllm logs --study-name <your_study_name> --database-url postgresql://user:pass@host:5432/db
```

### Next Steps

- Check available resources: `auto-tune-vllm check-env --ray-cluster`
- Tune concurrency realistically based on GPUs (e.g., 1 GPU per trial)

Reference: [Auto‑Tune vLLM (GitHub)](https://github.com/openshift-psap/auto-tuning-vllm/tree/main)


