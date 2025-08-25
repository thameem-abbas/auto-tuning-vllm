"""Benchmark provider implementations."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from abc import ABC, abstractmethod
from typing import Any, Dict

from .config import BenchmarkConfig

logger = logging.getLogger(__name__)


class BenchmarkProvider(ABC):
    """Abstract benchmark provider interface."""
    
    @abstractmethod
    def run_benchmark(self, model_url: str, config: BenchmarkConfig) -> Dict[str, Any]:
        """
        Run benchmark against model server.
        
        Args:
            model_url: URL of the vLLM server (e.g., "http://localhost:8000/v1")
            config: Benchmark configuration
            
        Returns:
            Dictionary with benchmark results. Must include metrics that can be
            converted to objective values for Optuna.
        """
        pass


class GuideLLMBenchmark(BenchmarkProvider):
    """GuideLLM benchmark provider implementation."""
    
    def run_benchmark(self, model_url: str, config: BenchmarkConfig) -> Dict[str, Any]:
        """Run GuideLLM benchmark."""
        logger.info(f"Starting GuideLLM benchmark for {config.model}")
        
        # Create temporary file for results
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            results_file = f.name
        
        try:
            # Build GuideLLM command
            cmd = self._build_guidellm_command(model_url, config, results_file)
            
            # Run GuideLLM
            logger.info(f"Running: {' '.join(cmd)}")
            process = subprocess.run(
                cmd,
                timeout=config.max_seconds + 120,  # Add buffer for setup/teardown
                capture_output=True,
                text=True,
                check=True
            )
            
            # Log GuideLLM output for debugging
            if process.stdout:
                logger.info(f"GuideLLM stdout:\n{process.stdout}")
            if process.stderr:
                logger.warning(f"GuideLLM stderr:\n{process.stderr}")
            
            logger.debug(f"GuideLLM process completed with return code: {process.returncode}")
            
            logger.info("GuideLLM completed successfully")
            
            # Parse results
            return self._parse_guidellm_results(results_file)
            
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"GuideLLM benchmark timed out after {config.max_seconds + 120} seconds")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"GuideLLM failed: {e.stderr}")
        except Exception as e:
            raise RuntimeError(f"Benchmark execution failed: {e}")
        finally:
            # Cleanup results file
            if os.path.exists(results_file):
                os.unlink(results_file)
    
    def _build_guidellm_command(
        self, 
        model_url: str, 
        config: BenchmarkConfig, 
        results_file: str
    ) -> list[str]:
        """Build GuideLLM command arguments."""
        cmd = [
            "guidellm",
            "--url", model_url,
            "--model", config.model,
            "--max-seconds", str(config.max_seconds),
            "--concurrency", str(config.concurrency),
            "--output", results_file
        ]
        
        # Add dataset or synthetic data configuration
        if config.use_synthetic_data:
            cmd.extend([
                "--data-type", "synthetic",
                "--prompt-tokens", str(config.prompt_tokens),
                "--output-tokens", str(config.output_tokens)
            ])
        else:
            if config.dataset.startswith("hf://"):
                # HuggingFace dataset
                dataset_name = config.dataset[5:]  # Remove "hf://" prefix
                cmd.extend(["--data-type", "huggingface", "--dataset", dataset_name])
            else:
                # Local file
                cmd.extend(["--data-type", "file", "--dataset", config.dataset])
        
        return cmd
    
    def _parse_guidellm_results(self, results_file: str) -> Dict[str, Any]:
        """Parse GuideLLM JSON results."""
        if not os.path.exists(results_file):
            raise RuntimeError(f"GuideLLM results file not found: {results_file}")
        
        try:
            with open(results_file) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON in results file: {e}")
        
        # Extract benchmark data structure
        try:
            benchmarks = data["benchmarks"]
            if not benchmarks:
                raise RuntimeError("No benchmark data found in results")
            
            # Get first (and typically only) benchmark result
            benchmark_data = benchmarks[0]
            metrics = benchmark_data["metrics"]
            
            # Extract key metrics with percentiles
            result = {}
            
            required_metrics = [
                "requests_per_second",
                "request_latency", 
                "output_tokens_per_second",
                "time_to_first_token_ms",
                "inter_token_latency_ms"
            ]
            
            for metric_name in required_metrics:
                if metric_name not in metrics:
                    logger.warning(f"Missing metric: {metric_name}")
                    continue
                
                metric_data = metrics[metric_name]["successful"]
                
                # Store median value and percentiles
                result[metric_name] = metric_data["median"]
                
                percentiles = metric_data.get("percentiles", {})
                for percentile, value in percentiles.items():
                    result[f"{metric_name}_{percentile}"] = value
            
            return result
            
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"Invalid benchmark data structure: {e}")


class CustomBenchmarkTemplate(BenchmarkProvider):
    """Template for implementing custom benchmark providers."""
    
    def run_benchmark(self, model_url: str, config: BenchmarkConfig) -> Dict[str, Any]:
        """
        Template implementation for custom benchmarks.
        
        Override this method to implement your custom benchmark logic.
        The returned dictionary should contain metrics that will be used
        to compute objective values for Optuna optimization.
        """
        # Example custom benchmark implementation:
        
        # 1. Setup your benchmark client/tools
        # benchmark_client = YourBenchmarkTool(model_url)
        
        # 2. Run your benchmark logic
        # benchmark_results = benchmark_client.run_test(
        #     duration=config.max_seconds,
        #     concurrency=config.concurrency,
        #     # ... other config parameters
        # )
        
        # 3. Return structured results
        # The keys should be metric names, values should be numeric
        # Optuna will use these for optimization objectives
        
        return {
            "throughput": 0.0,  # requests/second or tokens/second
            "latency_p95": 0.0,  # 95th percentile latency in ms
            "latency_mean": 0.0,  # mean latency in ms
            "error_rate": 0.0,   # error percentage
            # Add any other metrics relevant to your benchmark
            # These will be stored in the detailed_metrics and can be
            # used for analysis and visualization
        }


# Registry for dynamic benchmark provider loading
BENCHMARK_PROVIDERS = {
    "guidellm": GuideLLMBenchmark,
    "custom_template": CustomBenchmarkTemplate,
}


def get_benchmark_provider(provider_name: str) -> BenchmarkProvider:
    """Get benchmark provider by name."""
    if provider_name not in BENCHMARK_PROVIDERS:
        raise ValueError(f"Unknown benchmark provider: {provider_name}. "
                        f"Available providers: {list(BENCHMARK_PROVIDERS.keys())}")
    
    provider_class = BENCHMARK_PROVIDERS[provider_name]
    return provider_class()