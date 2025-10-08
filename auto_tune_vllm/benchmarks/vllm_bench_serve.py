"""vLLM bench serve benchmark provider implementation."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlparse

from .config import BenchmarkConfig
from .providers import BenchmarkProvider

logger = logging.getLogger(__name__)


class VLLMBenchServeBenchmark(BenchmarkProvider):
    """vLLM bench serve benchmark provider implementation."""

    def run_benchmark(self, model_url: str, config: BenchmarkConfig) -> Dict[str, Any]:
        """Run vLLM bench serve benchmark."""
        self._logger.info(f"Starting vLLM bench serve benchmark for {config.model}")

        # Parse model URL to extract base URL and endpoint
        base_url, endpoint = self._parse_model_url(model_url)
        
        # Create results file path
        results_file = self._get_results_file_path()

        try:
            # Build vLLM bench serve command
            cmd = self._build_vllm_bench_command(base_url, endpoint, config, results_file)
            
            # Validate binary
            import shutil
            if shutil.which("vllm") is None:
                raise RuntimeError(
                    "vLLM CLI not found on PATH. "
                    "Please install vLLM or provide the full path."
                )

            # Run vLLM bench serve
            self._logger.info(f"Running: {' '.join(cmd)}")
            self._logger.info(f"Results will be saved to: {results_file}")
            
            # Use Popen so we can terminate if vLLM dies
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            stdout, stderr, returncode = None, None, None
            
            try:
                # Wait for completion with timeout
                stdout, stderr = self._process.communicate(
                    timeout=config.max_seconds * 1.5
                )
                returncode = self._process.returncode
            
            except subprocess.TimeoutExpired as e:
                self._logger.warning("vLLM bench serve timed out, terminating process")
                self.terminate_benchmark()
                timeout_seconds = config.max_seconds * 1.5
                raise RuntimeError(
                    f"vLLM bench serve benchmark timed out after {timeout_seconds} seconds"
                ) from e

            finally:
                self._process = None
            
            if stdout is not None:
                self._logger.info(f"vLLM bench serve stdout:\n{stdout}")
            if stderr is not None:
                self._logger.warning(f"vLLM bench serve stderr:\n{stderr}")
            if returncode is not None and returncode != 0:
                raise subprocess.CalledProcessError(
                    returncode, cmd, stdout, stderr
                )

            self._logger.debug(
                f"vLLM bench serve process completed with return code: {returncode}"
            )
            self._logger.info("vLLM bench serve completed successfully")
        
        except Exception as e:
            self._logger.error(f"Error during vLLM bench serve benchmark: {e}")
            raise e

        if not os.path.exists(results_file):
            raise RuntimeError(
                f"vLLM bench serve results file not found after completion: {results_file}"
            )
        
        # Parse results
        return self._parse_vllm_bench_results(results_file)

    def _parse_model_url(self, model_url: str) -> tuple[str, str]:
        """Parse model URL to extract base URL and endpoint."""
        if not (model_url.startswith("http://") or model_url.startswith("https://")):
            raise ValueError(
                f"Invalid model_url: {model_url!r} (expected http/https)"
            )
        
        parsed = urlparse(model_url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        # If ends with /v1, add /completions. Otherwise, use the path.
        if parsed.path.endswith("/v1"):
            endpoint = parsed.path + "/completions"
        else:
            endpoint = parsed.path
        
        self._logger.debug(f"Parsed URL - base_url: {base_url}, endpoint: {endpoint}")
        return base_url, endpoint


    def _get_results_file_path(self) -> str:
        """
        Get the permanent results file path, creating directory structure if needed.
        Reuses the same logic as GuideLLM for consistency.
        """
        if self._trial_context is None:
            # Fallback to temporary file if no trial context
            self._logger.warning("No trial context set, using temporary file")
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                return f.name

        try:
            # Create directory structure:
            # /tmp/auto-tune-vllm-local-run/logs/{study_name}/benchmark_results/
            study_name = self._trial_context["study_name"]
            trial_id = self._trial_context["trial_id"]

            # Use /tmp as base directory for consistency with existing log structure
            base_dir = Path("/tmp/auto-tune-vllm-local-run/logs")
            benchmark_dir = base_dir / study_name / "benchmark_results"

            # Create directory if it doesn't exist
            benchmark_dir.mkdir(parents=True, exist_ok=True)

            # Create permanent results file with trial-specific name
            permanent_file = benchmark_dir / f"{trial_id}_vllm_bench_results.json"

            return str(permanent_file)

        except Exception as e:
            self._logger.warning(
                f"Failed to create permanent results path: {e}, using temporary file"
            )
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                return f.name

    def _build_vllm_bench_command(
        self, base_url: str, endpoint: str, config: BenchmarkConfig, results_file: str
    ) -> list[str]:
        """Build vLLM bench serve command arguments."""
        cmd = [
            "vllm", "bench", "serve",
            "--dataset-name", "random",  # Always use synthetic data
            "--backend", "vllm",  # Fixed backend
            "--base-url", base_url,
            "--endpoint", endpoint,
            "--max-concurrency", str(config.rate),  # Map rate to max-concurrency
            "--request-rate", str(config.rate),  # Use request rate mode instead of num-prompts
            "--model", config.model,
            "--random-input-len", str(config.prompt_tokens),
            "--random-output-len", str(config.output_tokens),
            "--save-result",
            "--percentile-metrics", "ttft,tpot,itl,e2el",
            "--metric-percentiles", "25,50,75,90,95,99",
            "--save-detailed",
            "--result-filename", results_file,
        ]

        # Add dataset configuration if provided
        if config.dataset and not config.use_synthetic_data:
            if config.dataset.startswith("hf://"):
                # HuggingFace dataset
                dataset_name = config.dataset[5:]  # Remove "hf://" prefix
                cmd.extend(["--dataset-name", "hf", "--dataset-path", dataset_name])
            else:
                # Local file
                if not os.path.exists(config.dataset):
                    raise FileNotFoundError(f"Dataset file not found: {config.dataset}")
                cmd.extend(["--dataset-name", "custom", "--dataset-path", config.dataset])
        
        return cmd

    def _parse_vllm_bench_results(self, results_file: str) -> Dict[str, Any]:
        """Parse vLLM bench serve JSON results and map to standard format."""
        if not os.path.exists(results_file):
            raise RuntimeError(f"vLLM bench serve results file not found: {results_file}")

        try:
            with open(results_file) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON in results file: {e}")

        # Extract and map metrics to standard format
        result = {}

        # Core throughput metrics
        result["requests_per_second"] = data.get("request_throughput", 0.0)
        result["output_tokens_per_second"] = data.get("output_throughput", 0.0)
        result["tokens_per_second"] = data.get("total_token_throughput", 0.0)

        # Time to First Token (TTFT) metrics
        self._extract_metric_with_percentiles(
            result, data, "ttft", "time_to_first_token_ms"
        )

        # Time Per Output Token (TPOT) metrics
        self._extract_metric_with_percentiles(
            result, data, "tpot", "time_per_output_token_ms"
        )

        # Inter-Token Latency (ITL) metrics
        self._extract_metric_with_percentiles(
            result, data, "itl", "inter_token_latency_ms"
        )

        # End-to-End Latency (E2EL) metrics
        self._extract_metric_with_percentiles(
            result, data, "e2el", "request_latency_ms"
        )

        # Add backward compatibility metrics (use median as default)
        result["time_to_first_token_ms"] = result.get("time_to_first_token_ms_p50", 0.0)
        result["time_per_output_token_ms"] = result.get("time_per_output_token_ms_p50", 0.0)
        result["inter_token_latency_ms"] = result.get("inter_token_latency_ms_p50", 0.0)
        
        # Convert E2EL from ms to seconds for backward compatibility
        e2el_median_ms = result.get("request_latency_ms_p50", 0.0)
        result["request_latency"] = e2el_median_ms / 1000.0

        # Add additional useful metrics
        result["total_input_tokens"] = data.get("total_input_tokens", 0)
        result["total_output_tokens"] = data.get("total_output_tokens", 0)
        result["completed_requests"] = data.get("completed", 0)
        result["duration_seconds"] = data.get("duration", 0.0)

        self._logger.debug(f"Extracted {len(result)} metrics from vLLM bench serve results")
        return result

    def _extract_metric_with_percentiles(
        self, result: Dict[str, Any], data: Dict[str, Any], 
        metric_prefix: str, result_prefix: str
    ) -> None:
        """Extract a metric with all its percentiles and statistical measures."""
        # Extract mean, median, std, and percentiles
        mean_key = f"mean_{metric_prefix}_ms"
        median_key = f"median_{metric_prefix}_ms"
        std_key = f"std_{metric_prefix}_ms"
        
        if mean_key in data:
            result[f"{result_prefix}_mean"] = data[mean_key]
        if median_key in data:
            result[f"{result_prefix}_median"] = data[median_key]
        if std_key in data:
            result[f"{result_prefix}_std"] = data[std_key]

        # Extract percentiles (p25, p50, p75, p90, p95, p99)
        percentiles = ["p25", "p50", "p75", "p90", "p95", "p99"]
        for p in percentiles:
            p_key = f"{p}_{metric_prefix}_ms"
            if p_key in data:
                result[f"{result_prefix}_{p}"] = data[p_key]
