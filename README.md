# vLLM Auto-tuning Redhat

Program finds a configuration (or multiple) that maximises output tokens/second (throughput) for a 32B MoE LLM served by vLLM.

## Requirements

- Python 3.8+
- vLLM server
- guidellm
- HuggingFace Hub access (for model validation)
- NVIDIA GPU with CUDA support

## Study Results

Results are stored in SQLite database at `src/studies/study_N/optuna.db` and can be analyzed using Optuna's visualization tools. 
