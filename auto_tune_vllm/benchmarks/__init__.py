"""Benchmark providers and interfaces."""

from .config import BenchmarkConfig
from .providers import BenchmarkProvider
from .guidellm_provider import GuideLLMBenchmark
from .custom_benchmark_template import CustomBenchmarkTemplate

# Registry for dynamic benchmark provider loading (for reference/documentation)
BENCHMARK_PROVIDERS = {
    "guidellm": GuideLLMBenchmark,
    "custom_template": CustomBenchmarkTemplate,
}

__all__ = [
    "BenchmarkProvider",
    "GuideLLMBenchmark",
    "CustomBenchmarkTemplate",
    "BenchmarkConfig",
    "BENCHMARK_PROVIDERS",
]